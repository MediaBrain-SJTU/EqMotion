import numpy as np
import torch
import random

class NBodyDataset():
    def __init__(self, partition='train', max_samples=1e8, dataset_name="nbody", past_length=20, future_length=20, training=True):
        self.partition = partition
        if self.partition == 'val':
            self.sufix = 'valid'
        else:
            self.sufix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.sufix += "_charged5_initvel1"

        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.sufix += "_charged5_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()
        self.past_length = past_length
        self.future_length = future_length
        self.training = training

    def load(self):
        loc = np.load('n_body_system/dataset/loc_' + self.sufix + '.npy')
        vel = np.load('n_body_system/dataset/vel_' + self.sufix + '.npy')
        edges = np.load('n_body_system/dataset/edges_' + self.sufix + '.npy')
        charges = np.load('n_body_system/dataset/charges_' + self.sufix + '.npy')

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        loc = loc.transpose(1,2)
        vel = vel.transpose(1,2)
        vel = torch.zeros_like(loc)
        vel[:,:,1:] = loc[:,:,1:] - loc[:,:,:-1]
        vel[:,:,0] = vel[:,:,1]
        return (loc, vel, edge_attr, charges), edges


    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = torch.Tensor(edges)

        #Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        # edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2) # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc), torch.Tensor(vel), edge_attr, edges, torch.Tensor(charges)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]
        frame_0 = self.past_length
        frame_T = self.past_length + self.future_length

        if self.training:
            return loc[:,0:frame_0], vel[:,0:frame_0], edge_attr, charges, loc[:,frame_0:frame_T]
        else:
            return loc[:,0:frame_0], vel[:,0:frame_0], edge_attr, charges, loc[:,frame_0:frame_T]

    def __len__(self):
        return len(self.data[0])


class NBodyDataset_reasoning2():
    def __init__(self, partition='train', max_samples=1e8, dataset_name="nbody", past_length=20, future_length=20, training=True):
        self.partition = partition
        if self.partition == 'val':
            self.sufix = 'valid'
        else:
            self.sufix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            # self.sufix += "_charged5_initvel1"
            self.sufix += "_charged_pos5_initvel1"

        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.sufix += "_charged5_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()
        self.past_length = past_length
        self.future_length = future_length
        self.training = training

    def load(self):
        loc = np.load('n_body_system/dataset/loc_' + self.sufix + '.npy')
        vel = np.load('n_body_system/dataset/vel_' + self.sufix + '.npy')
        edges = np.load('n_body_system/dataset/edges_' + self.sufix + '.npy')
        charges = np.load('n_body_system/dataset/charges_' + self.sufix + '.npy')

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        loc = loc.transpose(1,2)
        vel = vel.transpose(1,2)
        vel = torch.zeros_like(loc)
        vel[:,:,1:] = loc[:,:,1:] - loc[:,:,:-1]
        vel[:,:,0] = vel[:,:,1]
        return (loc, vel, edge_attr, charges), edges


    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0:self.max_samples]
        edge_attr = torch.Tensor(edges)

        #Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        # edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2) # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc), torch.Tensor(vel), edge_attr, edges, torch.Tensor(charges)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]
        frame_0 = self.past_length
        frame_T = self.past_length + self.future_length

        if self.training:
            return loc[:,0:frame_0], vel[:,0:frame_0], edge_attr, charges, loc[:,frame_0:frame_T]
        else:
            return loc[:,0:frame_0], vel[:,0:frame_0], edge_attr, charges, loc[:,frame_0:frame_T]

    def __len__(self):
        return len(self.data[0])


class NBodyDataset_reasoning():
    def __init__(self, partition='train', max_samples=1e8, dataset_name="nbody", past_length=20, future_length=20, training=True):
        self.partition = partition
        if self.partition == 'val':
            self.sufix = 'valid'
        else:
            self.sufix = self.partition
        self.dataset_name = dataset_name
        self.sufix += '_springs5_initvel1'

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()
        self.past_length = past_length
        self.future_length = future_length
        self.training = training

    def load(self):
        loc = np.load('n_body_system/dataset/loc_' + self.sufix + '.npy')
        vel = np.load('n_body_system/dataset/vel_' + self.sufix + '.npy')
        edges = np.load('n_body_system/dataset/edges_' + self.sufix + '.npy')
        

        loc, vel, edge_attr, edges = self.preprocess(loc, vel, edges)
        loc = loc.transpose(1,2)
        vel = vel.transpose(1,2)
        vel = torch.zeros_like(loc)
        vel[:,:,1:] = loc[:,:,1:] - loc[:,:,:-1]
        vel[:,:,0] = vel[:,:,1]

        return (loc, vel, edge_attr), edges


    def preprocess(self, loc, vel, edges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        # loc, vel = torch.Tensor(loc), torch.Tensor(vel)

        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        edge_attr = torch.Tensor(edges)

        #Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        # edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2) # swap n_nodes <--> batch_size and add nf dimension

        return torch.Tensor(loc), torch.Tensor(vel), edge_attr, edges

    def __getitem__(self, i):
        loc, vel, edge_attr = self.data
        loc, vel, edge_attr = loc[i], vel[i], edge_attr[i]
        frame_0 = self.past_length
        frame_T = self.past_length + self.future_length

        if self.training:
            return loc[:,0:frame_0], vel[:,0:frame_0], edge_attr, edge_attr, loc[:,frame_0:frame_T]
        else:
            return loc[:,0:frame_0], vel[:,0:frame_0], edge_attr, edge_attr, loc[:,frame_0:frame_T]

    def __len__(self):
        return len(self.data[0])


if __name__ == "__main__":
    NBodyDataset()