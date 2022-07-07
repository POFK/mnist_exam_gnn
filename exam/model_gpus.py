#!/usr/bin/env python3

import sys
import tempfile
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

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
train_loader = DataLoader(dataset[:50000], batch_size=1024, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset[55000:], batch_size=1000, shuffle=False, num_workers=4)

def step(model, optimizer, loss_fn):
    model.train()
    LOSS = 0
    CNT = 0
    t = tqdm.tqdm(train_loader)
    for data in t:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
#       loss = F.nll_loss(out, data.y)
        loss = loss_fn(out, data.y)
        LOSS += loss*(len(data.y))
        CNT += len(data.y)
        loss.backward()
        optimizer.step()
        t.set_description(f"epoch-{epoch:03d} loss: {LOSS/CNT:.5f}")
        t.refresh() # to show immediately the update


    model.eval()
    CORR = 0
    CNT = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data).argmax(dim=1)
        correct = (pred == data.y).sum()
        CORR += int(correct)
        CNT += len(data.y)
    acc = CORR/CNT
    print(f'Epoch {epoch} | Accuracy: {acc:.4f}, training loss: {LOSS/50000}')

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = Net().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-2, weight_decay=5e-4)


    for epoch in range(250):
        step(ddp_model, optimizer, loss_fn)

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    run_demo(demo_basic, 4)
