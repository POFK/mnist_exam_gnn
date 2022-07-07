#!/usr/bin/env python3

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


#torch.set_num_threads(24)

#dataset = GNNBenchmarkDataset(root='/config/data/gnn-MNIST', name='MNIST')
dataset = CustomMNISTDataset(root=os.path.join(os.path.expanduser('~'),"data"))
train_loader = DataLoader(dataset[:50000], batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset[55000:], batch_size=5000, shuffle=False, num_workers=4)


class GCN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_node_features = kwargs['num_node_features'] if "num_node_features" in kwargs.keys() else 2
#       self.num_classes = kwargs['num_classes'] if "num_classes" in kwargs.keys() else 10
        self.num_classes = 10
        self.conv1 = GCNConv(self.num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.linear1 = torch.nn.Linear(16,self.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x, _ = scatter_max(x, data.batch, dim=0)
        x = self.linear1(x)
        x = F.dropout(x, training=self.training)

        return F.log_softmax(x, dim=1)

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = GCN().to(device)
model = Net().to(device)
#data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
#optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(250):
    model.train()
    LOSS = 0
    CNT = 0
    t = tqdm.tqdm(train_loader)
    for data in t:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
#       print('*'*80)
#       print("hhhhh",out.shape, data.y.shape)
#       print('*'*80)

#       loss = F.nll_loss(out, data.y)
        loss = criterion(out, data.y)
        LOSS += loss*(len(data.y))
        CNT += len(data.y)
        loss.backward()
        optimizer.step()
        t.set_description(f"epoch-{epoch:03d} loss: {LOSS/CNT:.5f}")
        t.refresh() # to show immediately the update


    model.eval()
    for data in test_loader:
        pred = model(data).argmax(dim=1)
        correct = (pred == data.y).sum()
        acc = int(correct) / len(data.y)
        print(f'Epoch {epoch} | Accuracy: {acc:.4f}, training loss: {LOSS/50000}')
