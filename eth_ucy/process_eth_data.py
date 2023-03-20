import os, random, numpy as np, copy

from eth_ucy.dataloader import data_generator

import torch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--subset', type=str, default='eth',
                    help='Name of the subset.')
    parser.add_argument('--total_num', type=int, default=5,
                help='Name of the subset.')
    args = parser.parse_args()
    past_length = 8
    future_length = 12
    scale = 1

    generator_train = data_generator(args.subset, past_length, future_length, scale, split='train', phase='training')
    generator_test = data_generator(args.subset, past_length, future_length, scale, split='test', phase='testing')
    
    total_num = args.total_num
    all_past_data = [] #(B,N,T,2)
    all_future_data = [] #(B,N,T,2)
    all_valid_num = []
    print('start process training data:')
    while not generator_train.is_epoch_end():
        data = generator_train()
        if data is not None:
            loc = torch.stack(data['pre_motion_3D'],dim=0)
            loc_end = torch.stack(data['fut_motion_3D'],dim=0)
            length = loc.shape[1]
            length_f = loc_end.shape[1]
            agent_num = loc.shape[0]
            loc = np.array(loc)
            loc_end = np.array(loc_end)
            if loc.shape[0] < total_num:
                for i in range(loc.shape[0]):
                    temp = np.zeros((total_num,length,2))
                    temp[0] = loc[i]
                    temp[1:agent_num] = np.delete(loc,i,axis=0)
                    all_past_data.append(temp[None])
                    
                    temp = np.zeros((total_num,length_f,2))
                    temp[0] = loc_end[i]
                    temp[1:agent_num] = np.delete(loc_end,i,axis=0)
                    all_future_data.append(temp[None])
                    all_valid_num.append(agent_num)
            else:
                for i in range(loc.shape[0]):
                    distance_i = np.linalg.norm(loc[:,-1] - loc[i:i+1,-1],axis=-1)
                    neighbors_idx = np.argsort(distance_i)
                    assert neighbors_idx[0] == i
                    neighbors_idx = neighbors_idx[:total_num]
                    temp = loc[neighbors_idx]
                    all_past_data.append(temp[None])
                    temp = loc_end[neighbors_idx]
                    all_future_data.append(temp[None])
                    all_valid_num.append(total_num)

    all_past_data = np.concatenate(all_past_data,axis=0)
    all_future_data = np.concatenate(all_future_data,axis=0)
    print(all_past_data.shape)
    print(all_future_data.shape)
    all_data = np.concatenate([all_past_data,all_future_data],axis=2)
    all_valid_num = np.array(all_valid_num)
    print(all_data.shape)
    print(all_valid_num.shape)
    np.save('processed_data/'+ args.subset +'_data_train.npy',all_data)
    np.save('processed_data/'+ args.subset +'_num_train.npy',all_valid_num)

                    
    all_past_data = [] #(B,N,T,2)
    all_future_data = [] #(B,N,T,2)
    all_valid_num = []
    print('start process testing data:')
    while not generator_test.is_epoch_end():
        data = generator_test()
        if data is not None:
            loc = torch.stack(data['pre_motion_3D'],dim=0)
            loc_end = torch.stack(data['fut_motion_3D'],dim=0)
            length = loc.shape[1]
            length_f = loc_end.shape[1]
            agent_num = loc.shape[0]
            loc = np.array(loc)
            loc_end = np.array(loc_end)
            if loc.shape[0] < total_num:
                for i in range(loc.shape[0]):
                    temp = np.zeros((total_num,length,2))
                    temp[0] = loc[i]
                    temp[1:agent_num] = np.delete(loc,i,axis=0)
                    all_past_data.append(temp[None])
                    
                    temp = np.zeros((total_num,length_f,2))
                    temp[0] = loc_end[i]
                    temp[1:agent_num] = np.delete(loc_end,i,axis=0)
                    all_future_data.append(temp[None])
                    all_valid_num.append(agent_num)
            else:
                for i in range(loc.shape[0]):
                    distance_i = np.linalg.norm(loc[:,-1] - loc[i:i+1,-1],axis=-1)
                    neighbors_idx = np.argsort(distance_i)
                    assert neighbors_idx[0] == i
                    neighbors_idx = neighbors_idx[:total_num]
                    temp = loc[neighbors_idx]
                    all_past_data.append(temp[None])
                    temp = loc_end[neighbors_idx]
                    all_future_data.append(temp[None])
                    all_valid_num.append(total_num)

    all_past_data = np.concatenate(all_past_data,axis=0)
    all_future_data = np.concatenate(all_future_data,axis=0)
    print(all_past_data.shape)
    print(all_future_data.shape)
    all_data = np.concatenate([all_past_data,all_future_data],axis=2)
    all_valid_num = np.array(all_valid_num)
    print(all_data.shape)
    print(all_valid_num.shape)
    np.save('processed_data/'+ args.subset +'_data_test.npy',all_data)
    np.save('processed_data/'+ args.subset +'_num_test.npy',all_valid_num)





