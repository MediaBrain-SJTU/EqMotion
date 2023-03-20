from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from h36m import data_utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

class H36motion3D(Dataset):
    def __init__(self, actions='all', input_n=10, output_n=10, split=0, scale=100, sample_rate=2):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        path_to_data = 'h36m/dataset'
        # print('loading from', path_to_data)
        self.path_to_data = path_to_data
        self.split = split
        self.input_n = input_n
        self.output_n = output_n

        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        # print(subs)

        acts = data_utils.define_actions(actions)
        
        self.path_to_data = path_to_data

        subjs = subs[split]
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate, input_n + output_n)

        all_seqs = all_seqs / scale
        self.all_seqs_ori = all_seqs.copy()
        self.dim_used = dim_used
        all_seqs = all_seqs[:, :, dim_used] # （B,T,N*3）
        all_seqs = all_seqs.reshape(all_seqs.shape[0],all_seqs.shape[1],-1,3) #(B,T,N,3)
        all_seqs = all_seqs.transpose(0,2,1,3) #(B,N,T,3)
        all_seqs_vel = np.zeros_like(all_seqs)
        all_seqs_vel[:,:,1:] = all_seqs[:,:,1:] - all_seqs[:,:,:-1]
        all_seqs_vel[:,:,0] = all_seqs_vel[:,:,1]

        self.all_seqs = all_seqs
        self.all_seqs_vel = all_seqs_vel

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        loc_data = self.all_seqs[item]
        vel_data = self.all_seqs_vel[item]
        loc_data_ori = self.all_seqs_ori[item] #(T,N3)

        return loc_data[:,:self.input_n], vel_data[:,:self.input_n], loc_data[:,self.input_n:self.input_n+self.output_n],loc_data_ori[self.input_n:self.input_n+self.output_n],item
