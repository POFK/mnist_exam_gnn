#!/usr/bin/env python3

import sys
import tempfile
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import os
import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid, GNNBenchmarkDataset


import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_max, scatter_mean

from dataset import CustomMNISTDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


class Net(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        self.num_node_features = kwargs['num_node_features'] if "num_node_features" in kwargs.keys() else 2
        self.num_classes = 10
        self.conv1 = GCNConv(self.num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 48)
        self.conv4 = GCNConv(48, 64)
        self.conv5 = GCNConv(64, 96)
        self.conv6 = GCNConv(96, 128)
        self.linear1 = torch.nn.Linear(128,64)
        self.linear2 = torch.nn.Linear(64,self.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.conv6(x, edge_index)
        x = F.relu(x)
        x, _ = scatter_max(x, data.batch, dim=0)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
#       x = F.log_softmax(x, dim=1)
        return x



#torch.set_num_threads(24)

#dataset = GNNBenchmarkDataset(root='/config/data/gnn-MNIST', name='MNIST')
dataset = CustomMNISTDataset(root=os.path.join(os.path.expanduser('~'),"data"))

def prepare(ds, rank, world_size, batch_size=32, pin_memory=True, num_workers=0, shuffle=False):
    dataset = ds
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader

#train_loader = DataLoader(dataset[:50000], batch_size=1024, shuffle=True, num_workers=4)
#test_loader = DataLoader(dataset[55000:], batch_size=1000, shuffle=False, num_workers=4)

def save_checkpoint(rank, ddp_model, opt, epoch, cpt_path=None):
    cpt_path = cpt_path if cpt_path else tempfile.gettempdir() + "/model.checkpoint"
    cpt_data = {
            'epoch': epoch,
            'model_state_dict': ddp_model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            }
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(cpt_data, cpt_path)
    dist.barrier()

    return cpt_path

def step(rank, epoch, model, optimizer, loss_fn, train_loader, test_loader):
    model.train()
    LOSS = 0
    CNT = 0
    t = tqdm.tqdm(train_loader)
    for data in t:
        data = data.to(rank)
        optimizer.zero_grad()
        out = model(data)
#       loss = F.nll_loss(out, data.y)
        loss = loss_fn(out, data.y)
        LOSS += loss*(len(data.y))
        CNT += len(data.y)
        loss.backward()
        optimizer.step()
        t.set_description(f"rank-{rank:02d}, epoch-{epoch:03d} loss: {loss:.5f}")
        t.refresh() # to show immediately the update
    model.eval()
    CORR = 0
    CNT_test = 0
    for data in test_loader:
        data = data.to(rank)
        pred = model(data).argmax(dim=1)
        correct = (pred == data.y).sum()
        CORR += int(correct)
        CNT_test += len(data.y)
    acc = CORR/CNT_test
    print(f'rank-{rank:02d}, Epoch {epoch} | Accuracy: {acc:.4f}, training loss: {LOSS/CNT}')

def demo_basic(rank, world_size, bs, init=True):
    print(f"Running basic DDP example on rank {rank} of {world_size}.")
    setup(rank, world_size)
    train_loader = prepare(dataset[:50000], rank, world_size, batch_size=bs, shuffle=True)
    test_loader = prepare(dataset[50000:], rank, world_size, batch_size=bs, shuffle=False)

    # create model and move it to GPU with id rank
    model = Net().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.CrossEntropyLoss()
#   optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-2)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-2, weight_decay=5e-4)

    EPOCH_BEGIN = 0

    if not init:
        CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=map_location)
        ddp_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        EPOCH_BEGIN += checkpoint['epoch'] + 1
        dist.barrier()

    for epoch in range(250):
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        step(rank, EPOCH_BEGIN+epoch, ddp_model, optimizer, loss_fn, train_loader, test_loader)
        CHECKPOINT_PATH = save_checkpoint(rank, ddp_model, optimizer, EPOCH_BEGIN+epoch)

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()


def run_demo(demo_fn, *args):
    mp.spawn(demo_fn,
             args=args,
             nprocs=args[0],
             join=True)

if __name__ == '__main__':
#   fname="model.pt"
#   model_path = os.path.join(os.path.expanduser('~'),f"data/models/{fname}")
    #run_demo(demo_basic, torch.cuda.device_count())
    run_demo(demo_basic, 1, 1000, False)
    #run_demo(demo_basic, 4, 4000)
