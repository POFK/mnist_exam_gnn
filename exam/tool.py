#!/usr/bin/env python3

import os
from os.path import join as opj
import tqdm
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from dataset import CustomMNISTDataset
from model_gpus import Net
from db import DB

db = DB()

class TaskManager(object):
    def __init__(self, name="test_ui", *args, **kwargs):
        self.name = name
        self._rank = None
        self._world_size = None
        self._comm_mode = "gloo"
        self._epoch = 0
        self._model = None
        self._ddp_model = None
        self._optimizer = None

        self._cpt_dir = kwargs.get('cpt_dir',  tempfile.gettempdir())
        self._loss_fn = kwargs.get('loss_fn', None)
        self._lr = kwargs.get('lr', None)
        self._total_epoch = kwargs.get("EPOCH", 100)

        self.device = 'gpu'
        self._log_text = ""
        self.status = {"running": "True", "name": self.name, "epoch": 0}
        db.conn.hset("status", mapping=self.status)

    @property
    def rank(self):
        if self._rank is None:
            raise ValueError(f"rank should not be 'None'! {self._rank}")
        return self._rank

    @property
    def lr(self):
        if self._lr is None:
            raise ValueError(f"learning rate should not be 'None'! {self._lr}")
        return self._lr

    @property
    def world_size(self):
        return self._world_size

    @property
    def comm_mode(self):
        return self._comm_mode

    @property
    def epoch(self):
        return self._epoch

    @property
    def model(self):
        if not self._model:
            raise(ValueError, f"model should not be 'None'! {self._model}")
        return self._model

    @property
    def loss_fn(self):
        if not self._loss_fn:
            raise(ValueError, f"loss_fn should not be 'None'! {self._loss_fn}")
        return self._loss_fn

    @property
    def ddp_model(self):
        if not self._ddp_model:
            self._ddp_model = DDP(self._model, device_ids=[self.rank])
        return self._ddp_model

    @property
    def optimizer(self):
        if not self._optimizer:
            self._optimizer = torch.optim.Adam(self.ddp_model.parameters(), lr=self.lr)
        return self._optimizer

    def setup(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
                self.comm_mode, 
                rank=self.rank, 
                world_size=self.world_size)
        return self

    def cleanup(self):
        dist.destroy_process_group()
        return self

    def prepare(self, dataset, batch_size=32, pin_memory=True, num_workers=0, shuffle=False):
        sampler = DistributedSampler(dataset,
                                     num_replicas=self.world_size,
                                     rank=self.rank,
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

    def save_checkpoint(self, cpt_path=None):
        fp = cpt_path if cpt_path else "model.checkpoint"
        self.cpt_path = opj(self._cpt_dir, fp)
        cpt_data = {
            'epoch': self.epoch,
            'model_state_dict': self.ddp_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if self.rank == 0:
            torch.save(cpt_data, self.cpt_path)
#           print(f"Save checkpoint to {self.cpt_path}")
        dist.barrier()
        return self

    def load_checkpoint(self, path=None):
        fp = path if path else "model.checkpoint"
        self.cpt_path = opj(self._cpt_dir, fp)
        if self.rank==0:
            print(f"Load checkpoint from {self.cpt_path}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        checkpoint = torch.load(self.cpt_path, map_location=map_location)
        self.ddp_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._epoch = checkpoint['epoch'] + 1
        dist.barrier()
        return self

    def step_tr(self, train_loader):
        self.ddp_model.train()
        LOSS = 0
        CNT = 0
        for data in train_loader:
            data = data.to(self.rank)
            self.optimizer.zero_grad()
            out = self.ddp_model(data)
            loss = self.loss_fn(out, data.y)
            loss.backward()
            self.optimizer.step()
            LOSS += loss*(len(data.y))
            CNT += len(data.y)
        return CNT, LOSS

    def step_te(self, test_loader):
        self.ddp_model.eval()
        CORR = 0
        CNT = 0
        LOSS = 0
        for data in test_loader:
            data = data.to(self.rank)
            out = self.ddp_model(data)
            pred = out.argmax(dim=1)
            correct = (pred == data.y).sum()
            loss = self.loss_fn(out, data.y)
            CORR += int(correct)
            CNT += len(data.y)
            LOSS += loss*(len(data.y))
        return CNT, LOSS, CORR

    def step(self, train_loader, test_loader):
        tr_cnt, tr_loss = self.step_tr(train_loader)
        te_cnt, te_loss, te_corr = self.step_te(test_loader)
        data = {}
        data['epoch'] = self.epoch
        data['tr_loss'] = tr_loss/tr_cnt
        data['te_loss'] = te_loss/te_cnt
        data['acc'] = te_corr/te_cnt
        db.write(opj(self.name, self.rank), data)
#       if self.rank == 0:
#           self._log_text = f"Epoch {self.epoch:02}: train loss {tr_loss/tr_cnt:.5}, test loss {te_loss/te_cnt:.5}, acc {te_corr/te_cnt:.5}"
#       print(f"Epoch(rank {self.rank}) {self.epoch}: loss {te_loss/te_cnt}, acc {te_corr/te_cnt}")
        return self

    def set_ds(self, ds):
        self.dataset = ds
        return self

    def __call__(self, rank, world_size, bs, init=True):
        """
        Rewrite this method when use TaskManager
        """
        self._rank = rank
        self._world_size = world_size
        self.setup()
        dataset = self.dataset

        train_loader = self.prepare(dataset[:50000], batch_size=bs, shuffle=True)
        test_loader = self.prepare(dataset[50000:], batch_size=bs, shuffle=False)

        # create model and move it to GPU with id rank
        self._model = self.model.to(self.rank)
#       self._ddp_model = DDP(self.model, device_ids=[rank])
        self._optimizer = torch.optim.Adam(self.ddp_model.parameters(), lr=self.lr, weight_decay=5e-4)

        if not init:
            self.load_checkpoint(path=None)

        t = tqdm.tqdm(range(self._total_epoch))
        for i in t:
            train_loader.sampler.set_epoch(self.epoch)
            test_loader.sampler.set_epoch(self.epoch)
            self.step(train_loader, test_loader).save_checkpoint()
            if self.rank == 0:
                t.set_description(self._log_text)
                t.refresh() # to show immediately the update
                db.conn.hset("status","epoch",self.epoch)
            self._epoch += 1

        if self.rank == 0:
            db.conn.hset("running": "False")
            os.remove(self.cpt_path)
        self.cleanup()

    def run(self, world_size, bs, init):
        args = (world_size, bs, init)
        mp.spawn(self,
                 args=args,
                 nprocs=args[0],
                 join=True,
                 )



if __name__ == '__main__':
    tm = TaskManager(lr=1e-2)
    dataset = CustomMNISTDataset(root=os.path.join(os.path.expanduser('~'),"data"))
    tm._model = Net()
    tm._loss_fn = nn.CrossEntropyLoss()
    #tm.set_ds(dataset).run(4, 1000, False)
    tm.set_ds(dataset).run(1, 1000, True)
