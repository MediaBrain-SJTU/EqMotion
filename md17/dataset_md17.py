import numpy as np
import torch
import pickle as pkl
import os
# import networkx as nx
# from networkx.algorithms import tree


class MD17Dataset():
    def __init__(self, partition, max_samples, data_dir, molecule_type,past_length=25,future_length=25):

        full_dir = os.path.join(data_dir, molecule_type + '_' + partition + '.npy')

        self.max_samples = int(max_samples)
        self.data = self.load(data_dir,molecule_type,partition)
        self.past_length = past_length
        self.future_length = future_length
        self.partition = partition

    def load(self,data_dir,molecule_type,partition):

        loc_full_dir = os.path.join(data_dir, molecule_type + '_' + partition + '.npy')
        edge_full_dir = os.path.join(data_dir, molecule_type + '_' + 'structure.npy')

        loc = np.load(loc_full_dir) # (B,T,N,3)
        edge_attr = np.load(edge_full_dir)

        loc = loc[:self.max_samples]

        loc = torch.Tensor(loc)
        edge_attr = torch.Tensor(edge_attr)

        loc = loc.transpose(1,2)
        vel = torch.zeros_like(loc)
        vel[:,:,1:] = loc[:,:,1:] - loc[:,:,:-1]
        vel[:,:,0] = vel[:,:,1]

        edge_attr = edge_attr[None,:,:].repeat(loc.shape[0],1,1)
        # edge_attr = self.get_molecule_structure(loc[0])

        return (loc, vel, edge_attr)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr = self.data
        loc, vel, edge_attr = loc[i], vel[i], edge_attr[i]
        frame_0 = self.past_length
        frame_T = self.past_length + self.future_length

        # if self.training:
        return loc[:,0:frame_0], vel[:,0:frame_0], edge_attr, loc[:,frame_0:frame_T]
        # else:
        #     return loc[:,0:frame_0], vel[:,0:frame_0], edge_attr, loc[:,frame_0:frame_T]
        # if self.training:
        #     return loc[:,frame_0-2:frame_0], vel[:,frame_0-1:frame_0], edge_attr, charges, loc[:,frame_0:frame_T]
        # else:
        #     return loc[:,frame_0-2:frame_0], vel[:,frame_0-1:frame_0], edge_attr, charges, loc[:,frame_0:frame_T]

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges
