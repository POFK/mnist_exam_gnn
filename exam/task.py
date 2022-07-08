#!/usr/bin/env python3

import os
import tqdm
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from dataset import CustomMNISTDataset
from model_gpus import Net

class TaskManager(object):
    def __init__(self, *args, **kwargs):
        self.cpt_dir = None
        self.device = 'gpu'
        self.ds_train = None
        self.ds_test = None
        self.net = None

    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        return self

    def cleanup(self):
        dist.destroy_process_group()
        return self

    def prepare(self, dataset, rank, world_size, batch_size=32, pin_memory=True, num_workers=0, shuffle=False):
        sampler = DistributedSampler(dataset,
                                     num_replicas=world_size,
                                     rank=rank,
                                     shuffle=shuffle,
                                     drop_last=False)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                pin_memory=pin_memory,
                                num_workers=num_workers,
                                drop_last=False,
                                shuffle=False,
                                sampler=sampler)
        return dataloader

    def save_checkpoint(self, rank, ddp_model, opt, epoch, cpt_path=None):
        cpt_path = cpt_path if cpt_path else tempfile.gettempdir() + "/model.checkpoint"
        cpt_data = {
            'epoch': epoch,
            'model_state_dict': ddp_model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }
        if rank == 0:
            torch.save(cpt_data, cpt_path)
        dist.barrier()
        return cpt_path

    def step(self, rank, epoch, model, optimizer, loss_fn, train_loader, test_loader):
        model.train()
        LOSS = 0
        CNT = 0
        t = tqdm.tqdm(train_loader)
        for data in t:
            data = data.to(rank)
            optimizer.zero_grad()
            out = model(data)
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

    def demo_basic(self, rank, world_size, bs, init=True):
        self.setup(rank, world_size)
        dataset = CustomMNISTDataset(root=os.path.join(os.path.expanduser('~'),"data"))
        train_loader = self.prepare(dataset[:50000], rank, world_size, batch_size=bs, shuffle=True)
        test_loader = self.prepare(dataset[50000:], rank, world_size, batch_size=bs, shuffle=False)

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
            self.step(rank, EPOCH_BEGIN+epoch, ddp_model, optimizer, loss_fn, train_loader, test_loader)
            CHECKPOINT_PATH = self.save_checkpoint(rank, ddp_model, optimizer, EPOCH_BEGIN+epoch)

        if rank == 0:
            os.remove(CHECKPOINT_PATH)

        self.cleanup()

    def run(self, *args):
        mp.spawn(self.demo_basic,
                 args=args,
                 nprocs=args[0],
                 join=True,
                 )

if __name__ == '__main__':
    tm = Taskmanager()
    tm.run(1, 1000, False)
