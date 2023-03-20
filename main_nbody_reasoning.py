import argparse
import torch
from n_body_system.dataset_nbody import NBodyDataset_reasoning
from n_body_system.model_t_reason import EqMotion
import os
from torch import nn, optim
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='EqMotion')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--past_length', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--future_length', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=-1, metavar='N',
                    help='the rand seed')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                    help='folder to output')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='learning rate')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, default=5000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody", metavar='N',
                    help='nbody')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument("--apply_decay",action='store_true')
parser.add_argument("--vis",action='store_true')
parser.add_argument("--special",action='store_true')

time_exp_dic = {'time': 0, 'counter': 0}


args = parser.parse_args()
args.cuda = True


device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass

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
    if args.seed < 0:
        seed = random.randint(0,1000)
    else:
        seed = args.seed
    # seed = 475
    setup_seed(seed)
    print('The seed is :',seed)

    past_length = args.past_length
    future_length = args.future_length
    dataset_train = NBodyDataset_reasoning(partition='train', dataset_name=args.dataset,
                                 max_samples=args.max_training_samples, past_length=past_length, future_length=future_length)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    dataset_val = NBodyDataset_reasoning(partition='val', dataset_name=args.dataset, past_length=past_length, future_length=future_length,training=False)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    dataset_test = NBodyDataset_reasoning(partition='test', dataset_name=args.dataset, past_length=past_length, future_length=future_length,training=False)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)


    model = EqMotion(in_node_nf=args.past_length, in_edge_nf=2, hidden_nf=args.nf, in_channel=args.past_length, hid_channel=64, out_channel=args.future_length,device=device, n_layers=args.n_layers, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)    

    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            val_loss, _ = test(model, optimizer, epoch, loader_val, backprop=False)
            test_loss, ade = test(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['losess'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_ade = ade
                best_epoch = epoch
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best ade: %.5f \t Best epoch %d" % (best_val_loss, best_test_loss, best_ade, best_epoch))
            print('The seed is :',seed)

    return best_val_loss, best_test_loss, best_epoch

constant = 1


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, length, _ = data[0].size()
        data = [d.to(device) for d in data]
        # data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, edge_attr, charges, loc_end = data

        optimizer.zero_grad()


        vel = vel * constant
        nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()

        loc_pred, category = model(nodes, loc.detach(), vel,edge_attr)

        loss = torch.mean(torch.norm(loc_pred-loc_end,dim=-1))

        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']

def evaluate_accuracy(pred_logit, gt, log=False):
    assert pred_logit.shape[-1] == 2
    if pred_logit is None:
        return 0
    pred_type = torch.max(pred_logit,dim=-1)[1]
    # gt[gt<0] = 0
    if args.special:
        gt = (gt + 1) / 2
    else:
        gt = torch.abs(gt)
    mask = torch.ones_like(gt)
    for i in range(mask.shape[1]):
        mask[:,i,i] = 0
    acc = torch.sum(mask*(torch.abs(pred_type-gt)))/torch.sum(mask)
    acc = np.array(acc.cpu())
    # if log:
    #     print(gt[0:3])
    #     print(pred_type[0:3])
    #     print(torch.sum(mask[0:3]*(torch.abs(pred_type[0:3]-gt[0:3])))/torch.sum(mask[0:3]))
    return acc

def test(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    validate_reasoning = True
    if validate_reasoning:
        acc_list = [0]*args.n_layers
    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'ade': 0}

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            batch_size, n_nodes, length, _ = data[0].size()
            data = [d.to(device) for d in data]
            loc, vel, edge_attr, charges, loc_end = data


            optimizer.zero_grad()

            vel = vel * constant
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
            loc_pred, category_list = model(nodes, loc.detach(), vel,edge_attr)

            if validate_reasoning:
                for l in range(0,len(category_list)):
                    acc_list[l]+=(evaluate_accuracy(category_list[l],edge_attr,log=(batch_idx==0 and l == 0)))*batch_size

            loc_pred = np.array(loc_pred.cpu())
            loc_end = np.array(loc_end.cpu())
            ade = np.mean(np.linalg.norm(loc_pred[:,:,:,:]-loc_end[:,:,:,:],axis=-1))
            fde = np.mean(np.linalg.norm(loc_pred[:,:,-1:,:]-loc_end[:,:,-1:,:],axis=-1))

            res['loss'] += fde*batch_size
            res['ade'] += ade*batch_size
            res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f ade: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter'], res['ade'] / res['counter']))
    if validate_reasoning:
            print('acc:',max(acc_list[0]/res['counter'],1-acc_list[0]/res['counter']))
    return res['loss'] / res['counter'], res['ade'] / res['counter']

def draw_sample(trajs,pre=False,rot_id=0):
    print('drawing')
    for idx in range(50):
        plt.clf()
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # set figure information
        ax.set_title("3D_Curve")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        traj = trajs[idx]
        xmax = np.max(traj[:,:,0])
        xmin = np.min(traj[:,:,0])
        ymax = np.max(traj[:,:,1])
        ymin = np.min(traj[:,:,1])
        zmax = np.max(traj[:,:,2])
        zmin = np.min(traj[:,:,2])
        ax.set_xlim(xmin-0.2, xmax+0.2)
        ax.set_ylim(ymin-0.2, ymax+0.2)
        ax.set_zlim(zmin-0.2, zmax+0.2)

        color_list = ['orange', 'red', 'chocolate','dodgerblue','turquoise', 'blueviolet', 'cyan', 'navy', 'darkolivegreen', 'firebrick',
                    'crimson', 'orange', 'blue', 'steelblue', 'slategray', 'darkorange', 'lightcoral', 'tomato', 'darkviolet', 'fuchsia']
        # fig, ax = plt.subplots()
        length = traj.shape[1]
        for i in range(traj.shape[0]):
            figure = ax.plot(traj[i,:20,0],traj[i,:20,1],traj[i,:20,2], c=color_list[i],alpha=0.5)
            figure = ax.plot(traj[i,19:,0],traj[i,19:,1],traj[i,19:,2], c=color_list[i],alpha=1)

        if pre:
            plt.savefig('vis/physical/sample_'+str(idx)+'_'+str(rot_id)+'_pre.png')
        else:
            plt.savefig('vis/physical/sample_'+str(idx)+'_'+str(rot_id)+'_gt.png')

        plt.close()


def draw_heatmap(pred,gt):
    if args.special:
        gt = (gt + 1) / 2
    else:
        gt = np.abs(gt)
    pred = pred[...,0]
    for i in range(20):
        heatmap1 = pred[i]
        heatmap2 = gt[i]
        for j in range(5):
            heatmap1[j,j] = 0
            heatmap2[j,j] = 0
        sns.set()

        plt.clf()
        plt.figure(figsize=(5,5))
        ax = sns.heatmap(heatmap1,vmin=-0.2, vmax=1.2,cmap="Blues",linewidths=0.1,cbar=False,yticklabels=False,xticklabels=False) # 生成热力图
        plt.savefig('vis/physical/sample_'+str(i)+'_pre_heatmap1.png')
        plt.close()

        sns.set()
        plt.clf()
        plt.figure(figsize=(5,5))
        ax = sns.heatmap(1-heatmap1,vmin=-0.2, vmax=1.2,cmap="Blues",linewidths=0.1,cbar=False,yticklabels=False,xticklabels=False) # 生成热力图
        plt.savefig('vis/physical/sample_'+str(i)+'_pre_heatmap2.png')
        plt.close()

        sns.set()
        plt.clf()
        plt.figure(figsize=(5,5))
        ax = sns.heatmap(heatmap2,vmin=-0.2, vmax=1.2,cmap="Blues",linewidths=0.1,cbar=False,yticklabels=False,xticklabels=False) # 生成热力图
        plt.savefig('vis/physical/sample_'+str(i)+'_gt_heatmap.png')
        plt.close()
    return

if __name__ == "__main__":
    main()




