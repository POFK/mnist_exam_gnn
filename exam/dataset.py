#!/usr/bin/env python3
import tqdm
import os.path as osp
import numpy as np
import torch
from torchvision import datasets
from torch_geometric.loader import DataLoader
from torchvision.transforms import ToTensor
from torch_geometric.data import Dataset, Data
from torch_scatter import scatter_max, scatter_mean
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class CustomMNISTDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return 'rawdata.pt'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        data = datasets.MNIST(root=self.raw_dir,
                         train = True,
                         transform = ToTensor(),
                         download=True)
        torch.save(data, osp.join(self.raw_dir, self.raw_file_names))

    def process(self):
        ratio = 0.4
        dataset = torch.load(osp.join(self.raw_dir, self.raw_file_names))
        x = dataset.data.numpy()
        y_ = torch.tensor(dataset.targets.numpy())
        gdata = []

        x = np.where( x < (ratio*255), -1, 1000)

        if self.pre_filter is not None and not self.pre_filter(x):
            pass

        if self.pre_transform is not None:
            x = self.pre_transform(x)

        for i in tqdm.tqdm(range(len(x))):
#       for i in tqdm.tqdm(range(1000)):
            graph = self.__img_to_graph(x[i], y_[i])
#           print(i, graph)
            gdata.append(graph)
        print(graph)

#       print(gdata)
        self.data = gdata
        torch.save(gdata, osp.join(self.processed_dir, f'{self.processed_file_names}'))

    def len(self):
        try:
            return len(self.data)
        except AttributeError:
            self.data = torch.load(osp.join(self.processed_dir, f'{self.processed_file_names}'))
            return len(self.data)

    def get(self, idx):
        try:
            return self.data[idx]
        except AttributeError:
            self.data = torch.load(osp.join(self.processed_dir, f'{self.processed_file_names}'))
            return self.data[idx]

    def __img_to_graph(self, img, y):
        img = np.pad(img,[(2,2),(2,2)],"constant",constant_values=(-1))
        cnt = 0
        pos = []
        edges = []
        for i in range(2,30):
            for j in range(2,30):
                if img[i,j] == -1:
                    continue
                img[i,j] = cnt
                cnt += 1
        feature = np.zeros([cnt, 2])
        for i in range(2,30):
            for j in range(2,30):
                if img[i,j] == -1:
                    continue
                ind = img[i,j]
                pos.append([ind, i, j])
                im_fil = img[i-2:i+3,j-2:j+3].flatten()
                im_fil = np.delete(im_fil, 12)
                e_s = np.ones([np.sum(im_fil>-1)]) * ind
                e_s = e_s.astype(np.int64)
                e_t = im_fil[im_fil>-1]
                edges.append([e_s,e_t])
                feature[ind][0] = i
                feature[ind][1] = j
        edges = torch.tensor(np.hstack(edges), dtype=torch.long)
        x = torch.tensor(feature, dtype=torch.float)
        pos = torch.tensor(pos, dtype=torch.float)
        tg_data = Data(x=x, edge_index=edges, pos=pos, y=y, num_classes=10)
        return tg_data

if __name__ == '__main__':
    ds = CustomMNISTDataset(root="$HOME/data")
