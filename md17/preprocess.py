import numpy as np
import torch
import pickle as pkl
import os
import networkx as nx
from networkx.algorithms import tree


molecule_list = ['aspirin', 'benzene', 'ethanol', 'malonaldehyde']
# train_period_length = 5000
# val_period_length = 2000
# test_period_length = 2000
# total_period_length = train_period_length + val_period_length + test_period_length

train_proportion = 5
val_proportion = 1
test_proportion = 2

past_frames = 10
future_frames = 10
total_frames = past_frames + future_frames
framegap = 20
traj_length = total_frames * framegap

atom_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

for molecule_type in molecule_list:
    print(molecule_type)
    data_dir = 'dataset/' + molecule_type + '_dft.npz'
    data = np.load(data_dir)
    if molecule_type == 'uracil':
        sample_frequency = 10
    else:
        sample_frequency = 20

    x = data['R']
    v = x[1:] - x[:-1]
    x = x[:-1] # (T,N,3)
    z = data['z']
    x = x[:, z > 1, ...]
    v = v[:, z > 1, ...]
    z = z[z > 1]

    _lambda = 1.6

    def d(_i, _j, _t):
        return np.sqrt(np.sum((x[_t][_i] - x[_t][_j]) ** 2))

    n = x.shape[1]

    atom_edges = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                _d = d(i, j, 0)
                if _d < _lambda:
                    atom_edges[i][j] = 1
    # print(atom_edges)
    np.save('processed_dataset/'+ molecule_type + '_structure.npy', atom_edges)

    train_data = []
    test_data = []
    val_data = []
    total_length = x.shape[0]
    train_period_length = int(total_length * train_proportion / (train_proportion + val_proportion +test_proportion))
    val_period_length = int(total_length * val_proportion / (train_proportion + val_proportion +test_proportion))
    test_period_length = int(total_length * test_proportion / (train_proportion + val_proportion +test_proportion))
    # print(x.shape)
    total_period_length = train_period_length + val_period_length + test_period_length
    num_period_segment = int(total_length/total_period_length)
    assert num_period_segment == 1
    # print(num_period_segment)
    for i in range(num_period_segment):
        x_period = x[i*total_period_length:(i+1)*total_period_length]
        x_period_train = x_period[:train_period_length]
        x_period_val = x_period[train_period_length:train_period_length+val_period_length]
        x_period_test = x_period[train_period_length+val_period_length:]

        num_train = int((train_period_length-traj_length) / sample_frequency)
        for j in range(num_train):
            traj = x_period_train[sample_frequency*j:sample_frequency*j+traj_length:framegap]
            train_data.append(traj)

        num_val = int((val_period_length-traj_length) / sample_frequency)
        for j in range(num_val):
            traj = x_period_val[sample_frequency*j:sample_frequency*j+traj_length:framegap]
            val_data.append(traj)

        num_test = int((test_period_length-traj_length) / sample_frequency)
        for j in range(num_test):
            traj = x_period_test[sample_frequency*j:sample_frequency*j+traj_length:framegap]
            test_data.append(traj)
    train_data = np.stack(train_data,axis=0)
    val_data = np.stack(val_data,axis=0)
    test_data = np.stack(test_data,axis=0)

    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)
    print(train_data.shape)
    print(test_data.shape)
    print(val_data.shape)
    np.save('processed_dataset/'+ molecule_type + '_train.npy', train_data)
    np.save('processed_dataset/'+ molecule_type + '_val.npy', val_data)
    np.save('processed_dataset/'+ molecule_type + '_test.npy', test_data)

