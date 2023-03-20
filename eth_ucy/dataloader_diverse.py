import os, random, numpy as np, copy

import sys 
sys.path.append("..")
from eth_ucy.preprocessor import preprocess
import torch

def get_ethucy_split(dataset):
     seqs = [
          'biwi_eth',
          'biwi_hotel',
          'crowds_zara01',
          'crowds_zara02',
          'crowds_zara03',
          'students001',
          'students003',
          'uni_examples'
     ]
     if dataset == 'eth':
          test = ['biwi_eth']
     elif dataset == 'hotel':
          test = ['biwi_hotel']
     elif dataset == 'zara1':
          test = ['crowds_zara01']
     elif dataset == 'zara2':
          test = ['crowds_zara02']
     elif dataset == 'univ':
          test = ['students001', 'students003']

     train, val = [], []
     for seq in seqs:
          if seq in test:
               continue
          train.append(f'{seq}_train')
          val.append(f'{seq}_val')
     return train, val, test

def print_log(print_str, log, same_line=False, display=True):
	'''
	print a string to a log file
	parameters:
		print_str:          a string to print
		log:                a opened file to save the log
		same_line:          True if we want to print the string without a new next line
		display:            False if we want to disable to print the string onto the terminal
	'''
	if display:
		if same_line: print('{}'.format(print_str), end='')
		else: print('{}'.format(print_str))

	# if same_line: log.write('{}'.format(print_str))
	# else: log.write('{}\n'.format(print_str))
	# log.flush()


class eth_dataset(object):
    def __init__(self, dataset, past_frames, future_frames, traj_scale, split='train', phase='training'):
        # file_dir = 'eth_ucy/processed_data_diverse/'
        file_dir = '/GPFS/data/cxxu/trajectory_prediction/equivariant/egnn-main/eth_ucy/processed_data_diverse/'
        if phase == 'training':
            data_file_path = file_dir + dataset +'_data_train.npy'
            num_file_path = file_dir + dataset +'_num_train.npy'
        elif phase == 'testing':
            data_file_path = file_dir + dataset +'_data_test.npy'
            num_file_path = file_dir + dataset +'_num_test.npy'
        all_data = np.load(data_file_path)
        all_num = np.load(num_file_path)
        self.all_data = torch.Tensor(all_data)
        self.all_num = torch.Tensor(all_num)
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.traj_scale = traj_scale

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self,item):
        all_seq = self.all_data[item] / self.traj_scale
        num = self.all_num[item]
        past_seq = all_seq[:,:self.past_frames]
        future_seq = all_seq[:,self.past_frames:]
        return past_seq, future_seq, num

class data_generator(object):
    def __init__(self, dataset, past_frames, future_frames, traj_scale, split='train', phase='training'):
        self.past_frames = past_frames
        self.min_past_frames = past_frames
        self.future_frames = future_frames
        self.min_future_frames = future_frames
        self.frame_skip = 1
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = 'eth_ucy/data'          
            seq_train, seq_val, seq_test = get_ethucy_split(dataset)
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        process_func = preprocess
        self.data_root = data_root

        log = 'log'
        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        process_config = {}
        process_config['dataset'] = dataset
        process_config['past_frames'] = past_frames
        process_config['future_frames'] = future_frames
        process_config['frame_skip'] = self.frame_skip
        process_config['min_past_frames'] = past_frames
        process_config['min_future_frames'] = future_frames
        process_config['traj_scale'] = traj_scale
        
        for seq_name in self.sequence_to_load:
            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, process_config, log, self.split, self.phase)

            num_seq_samples = preprocessor.num_fr - (self.min_past_frames - 1) * self.frame_skip - self.min_future_frames * self.frame_skip + 1
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
            
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)
        
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):
        sample_index = self.sample_list[self.index]
        seq_index, frame = self.get_seq_and_frame(sample_index)
        seq = self.sequence[seq_index]
        self.index += 1
        
        data = seq(frame)
        return data      

    def __call__(self):
        return self.next_sample()


class data_generator_new(object):
    def __init__(self, dataset, past_frames, future_frames, traj_scale, split='train', phase='training'):
        self.past_frames = past_frames
        self.min_past_frames = past_frames
        self.future_frames = future_frames
        self.min_future_frames = future_frames
        self.frame_skip = 1
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = 'eth_ucy/data'          
            seq_train, seq_val, seq_test = get_ethucy_split(dataset)
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        process_func = preprocess
        self.data_root = data_root

        log = 'log'
        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        process_config = {}
        process_config['dataset'] = dataset
        process_config['past_frames'] = past_frames
        process_config['future_frames'] = future_frames
        process_config['frame_skip'] = self.frame_skip
        process_config['min_past_frames'] = past_frames
        process_config['min_future_frames'] = future_frames
        process_config['traj_scale'] = traj_scale
        

        for seq_name in self.sequence_to_load:
            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, process_config, log, self.split, self.phase)

            num_seq_samples = preprocessor.num_fr - (self.min_past_frames - 1) * self.frame_skip - self.min_future_frames * self.frame_skip + 1
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
            
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        self.stack_size = 16
        self.max_scene_size = 8
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)
        
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):
        cnt = 0
        seq_start_end = []
        all_loc = []
        all_loc_end = []
        while cnt < self.stack_size:
            if self.index >= self.num_total_samples:
                break
            sample_index = self.sample_list[self.index]
            seq_index, frame = self.get_seq_and_frame(sample_index)
            seq = self.sequence[seq_index]
            data = seq(frame)
            self.index += 1
            if data is not None:
                loc = torch.stack(data['pre_motion_3D'],dim=0)
                loc_end = torch.stack(data['fut_motion_3D'],dim=0)
                seq_start_end.append((cnt,cnt+loc.shape[0]))
                cnt += loc.shape[0]
                all_loc.append(loc)
                all_loc_end.append(loc_end)
        if len(all_loc) == 0:
            return None
        all_loc = torch.cat(all_loc,dim=0)
        all_loc_end = torch.cat(all_loc_end,dim=0)

        return all_loc, all_loc_end, seq_start_end

    def __call__(self):
        return self.next_sample()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--subset', type=str, default='eth',
                    help='Name of the subset.')
    args = parser.parse_args()
    past_length = 8
    future_length = 12
    scale = 1

    generator_train = data_generator(args.subset, past_length, future_length, scale, split='train', phase='training')
    generator_test = data_generator(args.subset, past_length, future_length, scale, split='test', phase='testing')
    
    total_num = 5
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
                    all_future_data.append(temp)
                    all_valid_num.append(agent_num)
            else:
                for i in range(loc.shape[0]):
                    distance_i = np.linalg.norm(loc[:,-1] - loc[i:i+1,-1],dim=-1)
                    neighbors_idx = np.argsort(distance_i)
                    assert neighbors_idx[0] == i
                    neighbors_idx = neighbors_idx[:total_num]
                    temp = loc[neighbors_idx]
                    all_past_data.append(temp[None])
                    temp = loc_end[neighbors_idx]
                    all_future_data.append(temp[None])
                    all_valid_num.append(total_num)

    all_past_data = np.concatenate(all_past_data,dim=0)
    all_future_data = np.concatenate(all_future_data,dim=0)
    print(all_past_data.shape)
    print(all_future_data.shape)
    all_data = np.concatenate([all_past_data,all_future_data],dim=1)
    all_valid_num = np.array(all_valid_num)
    print(all_data.shape)
    print(all_valid_num.shape)
    np.save('processed_data/'+ args.subset +'_data_train.npy',all_data)
    np.save('processed_data/'+ args.subset +'_num_train.npy',all_valid_num)







