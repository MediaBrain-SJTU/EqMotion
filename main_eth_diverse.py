import argparse
import torch
from eth_ucy.dataloader_diverse import eth_dataset
from eth_ucy.model_t import EqMotion
import os
from torch import nn, optim
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import random


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--past_length', type=int, default=8, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--future_length', type=int, default=12, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=-1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--epoch_decay', type=int, default=2, metavar='N',
                    help='number of epochs for the lr decay')
parser.add_argument('--lr_gamma', type=float, default=0.8, metavar='N',
                    help='the lr decay ratio')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='learning rate')
parser.add_argument('--model', type=str, default='egnn_vel', metavar='N',
                    help='available models: gnn, baseline, linear, linear_vel, se3_transformer, egnn_vel, rf_vel, tfn')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--degree', type=int, default=2, metavar='N',
                    help='degree of the TFN and SE3')
parser.add_argument('--channels', type=int, default=64, metavar='N',
                    help='number of channels')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--sweep_training', type=int, default=0, metavar='N',
                    help='0 nor sweep, 1 sweep, 2 sweep small')
parser.add_argument('--time_exp', type=int, default=0, metavar='N',
                    help='timing experiment')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--div', type=float, default=1, metavar='N',
                    help='timing experiment')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--subset', type=str, default='eth',
                    help='Name of the subset.')
parser.add_argument('--model_save_dir', type=str, default='eth_ucy/saved_models',
                    help='Name of the subset.')
parser.add_argument('--scale', type=float, default=1, metavar='N',
                    help='dataset scale')
parser.add_argument("--apply_decay",action='store_true')
parser.add_argument("--res_pred",action='store_true')
parser.add_argument("--supervise_all",action='store_true')
parser.add_argument('--model_name', type=str, default='eth_ckpt_best', metavar='N',
                    help='dataset scale')
parser.add_argument('--test_scale', type=float, default=1, metavar='N',
                    help='dataset scale')
parser.add_argument("--test",action='store_true')
parser.add_argument("--vis",action='store_true')

time_exp_dic = {'time': 0, 'counter': 0}


args = parser.parse_args()
args.cuda = True


device = torch.device("cuda" if args.cuda else "cpu")
# loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass

if args.subset == 'zara1':
    args.channels = 128
else:
    args.channels = 64

if args.subset == 'hotel':
    args.lr = 5e-4
else:
    args.lr = 1e-3

if args.subset == 'eth':
    args.test_scale = 1.6

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def lr_decay(optimizer, lr_now, gamma):
    lr_new = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new

def main():
    # seed = 861
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0,1000)
        setup_seed(seed)

    print('The seed is :',seed)

    past_length = args.past_length
    future_length = args.future_length

    dataset_train = eth_dataset(args.subset, args.past_length, args.future_length, args.scale, split='train', phase='training')
    dataset_test = eth_dataset(args.subset, args.past_length, args.future_length, args.test_scale, split='test', phase='testing')

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=8)


    model = EqMotion(in_node_nf=args.past_length, in_edge_nf=2, hidden_nf=args.nf, in_channel=args.past_length, hid_channel=args.channels, out_channel=args.future_length,device=device, n_layers=args.n_layers, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)    

    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.test:
        model_path = args.model_save_dir + '/' + args.model_name +'.pth.tar'
        print('Loading model from:', model_path)
        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt['state_dict'], strict=False)
        test_loss, ade = test(model, optimizer, 0, loader_test, backprop=False)
        print('ade:',ade,'fde:',test_loss)

    # if args.vis:
    #     model_path = args.model_save_dir + '/' + args.model_name +'.pth.tar'
    #     print('Loading model from:', model_path)
    #     model_ckpt = torch.load(model_path)
    #     model.load_state_dict(model_ckpt['state_dict'], strict=False)
    #     test_loss, ade = vis(model, optimizer, 0, loader_test, backprop=False)

    results = {'epochs': [], 'losess': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_ade = 1e8
    best_epoch = 0
    lr_now = args.lr
    for epoch in range(0, args.epochs):
        if args.apply_decay:
            if epoch % args.epoch_decay == 0 and epoch > 0:
                lr_now = lr_decay(optimizer, lr_now, args.lr_gamma)
        train(model, optimizer, epoch, loader_train)
        if epoch % args.test_interval == 0:
            test_loss, ade = test(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['losess'].append(test_loss)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_ade = ade
                best_epoch = epoch

                state = {'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()}
                file_path = os.path.join(args.model_save_dir, str(args.subset)+'_ckpt_best.pth.tar')
                torch.save(state, file_path)
            print("Best Test Loss: %.5f \t Best ade: %.5f \t Best epoch %d" % (best_test_loss, best_ade, best_epoch))
            print('The seed is :',seed)

            state = {'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}

            file_path = os.path.join(args.model_save_dir, str(args.subset)+'_ckpt_'+str(epoch)+'.pth.tar')
            torch.save(state, file_path)

    return best_val_loss, best_test_loss, best_epoch

constant = 1

def get_valid_mask2(num_valid,agent_num):
    batch_size = num_valid.shape[0]
    valid_mask = torch.zeros((batch_size,agent_num))
    for i in range(batch_size):
        valid_mask[i,:num_valid[i]] = 1
    return valid_mask.unsqueeze(-1).unsqueeze(-1)

def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        if data is not None:
            loc, loc_end, num_valid = data
            loc = loc.cuda()
            loc_end = loc_end.cuda()
            num_valid = num_valid.cuda()
            num_valid = num_valid.type(torch.int)

            vel = torch.zeros_like(loc)
            vel[:,:,1:] = loc[:,:,1:] - loc[:,:,:-1]
            vel[:,:,0] = vel[:,:,1]

            batch_size, agent_num, length, _ = loc.size()

            optimizer.zero_grad()

            vel = vel * constant
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
            loc_pred, category = model(nodes, loc.detach(), vel, num_valid)

            loc_end = loc_end[:,:,None,:,:]
            if args.supervise_all:
                mask = get_valid_mask2(num_valid,agent_num)
                mask = mask.cuda()
                mask = mask[:,:,None,:,:]
                loss = torch.mean(torch.min(torch.mean(torch.norm(mask*(loc_pred-loc_end),dim=-1),dim=3),dim=2)[0]) # only for ego agent
            else:
                loss = torch.mean(torch.min(torch.mean(torch.norm(loc_pred[:,0:1]-loc_end[:,0:1],dim=-1),dim=3),dim=2)[0]) # only for ego agent

            if backprop:
                loss.backward()
                optimizer.step()
            res['loss'] += loss.item() * batch_size
            res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']

def test(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    validate_reasoning = False
    if validate_reasoning:
        acc_list = [0]*args.n_layers
    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'ade': 0}

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if data is not None:
                loc, loc_end, num_valid = data
                loc = loc.cuda()
                loc_end = loc_end.cuda()
                num_valid = num_valid.cuda()
                num_valid = num_valid.type(torch.int)

                vel = torch.zeros_like(loc)
                vel[:,:,1:] = loc[:,:,1:] - loc[:,:,:-1]
                vel[:,:,0] = vel[:,:,1]

                batch_size, agent_num, length, _ = loc.size()

                optimizer.zero_grad()

                vel = vel * constant
                nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
                loc_pred, category_list = model(nodes, loc.detach(), vel, num_valid)

                loc_pred = np.array(loc_pred.cpu()) # B,N,20,T,2 [:,0,:,:,:]
                loc_end = np.array(loc_end.cpu()) # B,N,T,2 [:,0,:,:]
                loc_end = loc_end[:,:,None,:,:]
                ade = np.mean(np.min(np.mean(np.linalg.norm(loc_pred[:,0:1,:,:,:]-loc_end[:,0:1,:,:,:],axis=-1),axis=3),axis=2))
                fde = np.mean(np.min(np.mean(np.linalg.norm(loc_pred[:,0:1,:,-1:,:]-loc_end[:,0:1,:,-1:,:],axis=-1),axis=3),axis=2))

                res['loss'] += fde*batch_size
                res['ade'] += ade*batch_size
                res['counter'] += batch_size
    res['ade'] *= args.test_scale
    res['loss'] *= args.test_scale

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f ade: %.5f' % (prefix+'test', epoch, res['loss'] / res['counter'], res['ade'] / res['counter']))
    
    return res['loss'] / res['counter'], res['ade'] / res['counter']


if __name__ == "__main__":
    main()




