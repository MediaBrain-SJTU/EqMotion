import argparse
import torch
from md17.dataset_md17 import MD17Dataset
from n_body_system.model_t import EqMotion
import os
from torch import nn, optim
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import random


parser = argparse.ArgumentParser(description='EqMotion')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--past_length', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--future_length', type=int, default=10, metavar='N',
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
parser.add_argument('--mol', type=str, default='aspirin',
                    help='Name of the molecule.')
parser.add_argument("--vis",action='store_true')

# aspirin, benzene, ethanol, malonaldehyde, naphthalene, salicylic, toluene, uracil
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


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():
    seed = random.randint(0,1000)
    # seed = 861
    setup_seed(seed)
    print('The seed is :',seed)

    past_length = args.past_length
    future_length = args.future_length

    args.data_dir = 'md17/processed_dataset'
    args.delta_frame = 50

    dataset_train = MD17Dataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir,
                                molecule_type=args.mol, past_length=args.past_length, future_length=args.future_length)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)

    dataset_val = MD17Dataset(partition='val', max_samples=2000, data_dir=args.data_dir,
                                molecule_type=args.mol, past_length=args.past_length, future_length=args.future_length)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                             num_workers=8)

    dataset_test = MD17Dataset(partition='test', max_samples=2000, data_dir=args.data_dir,
                                molecule_type=args.mol, past_length=args.past_length, future_length=args.future_length)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=8)


    model = EqMotion(in_node_nf=args.past_length, in_edge_nf=2, hidden_nf=args.nf, in_channel=args.past_length, hid_channel=32, out_channel=args.future_length,device=device, n_layers=args.n_layers, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)    

    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results = {'epochs': [], 'losess': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_ade = 1e8
    best_epoch = 0
    for epoch in range(0, args.epochs):
        train(model, optimizer, epoch, loader_train)
        if epoch % args.test_interval == 0:
            #train(epoch, loader_train, backprop=False)
            val_loss, _ = test(model, optimizer, epoch, loader_val, backprop=False)
            test_loss, ade = test(model, optimizer, epoch, loader_test, backprop=False)
            # _, _ = check_equivariant(model, optimizer, epoch, loader_test, backprop=False)
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
        loc, vel, edge_attr, loc_end = data

        optimizer.zero_grad()
 
        vel = vel * constant
        nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
        loc_pred, category = model(nodes, loc.detach(), vel)

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
            batch_size, n_nodes, length, _ = data[0].size()
            data = [d.to(device) for d in data]
            loc, vel, edge_attr, loc_end = data

            optimizer.zero_grad()

            vel = vel * constant
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
            loc_pred, category_list = model(nodes, loc.detach(), vel)
            
            loc_pred = np.array(loc_pred.cpu())
            loc_end = np.array(loc_end.cpu())
            loc = np.array(loc.cpu())
            ade = np.mean(np.linalg.norm(loc_pred[:,:,:,:]-loc_end[:,:,:,:],axis=-1))
            fde = np.mean(np.linalg.norm(loc_pred[:,:,-1:,:]-loc_end[:,:,-1:,:],axis=-1))

            # if batch_idx == 0 and args.vis:
            #     loc_d = np.reshape(loc,(batch_size,n_nodes,-1,3))
            #     loc_pred_d = np.reshape(loc_pred,(batch_size,n_nodes,-1,3))
            #     loc_end_d = np.reshape(loc_end,(batch_size,n_nodes,-1,3))
            #     trajs = np.concatenate([loc_d,loc_end_d],axis=-2)
            #     draw_sample(trajs,pre=False)

            res['loss'] += fde*batch_size
            res['ade'] += ade*batch_size
            res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f ade: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter'], res['ade'] / res['counter']))
    return res['loss'] / res['counter'], res['ade'] / res['counter']  

def draw_sample(trajs,pre=False,rot_id=0):
    # print('drawing')
    import warnings
    warnings.filterwarnings('ignore')

    for idx in range(15):
        plt.cla()
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
        # length = args.past_frames+args.future_frames
        length = traj.shape[1]
        for i in range(traj.shape[0]):
            figure = ax.plot(traj[i,:,0],traj[i,:,1],traj[i,:,2], c=color_list[i])
            # for j in range(length):
            #     plt.scatter(traj[i,j,0],traj[i,j,1], c=color_list[i], alpha=np.arange(0.1, 1.0, 0.9/length)[j], s=50)
            # for j in range(length-1):
            #     plt.plot([traj[i,j,0],traj[i,j+1,0]],[traj[i,j,1],traj[i,j+1,1]], c=color_list[i])

        # ax.set_aspect('equal', 'box')
        if pre:
            plt.savefig('vis/md17/sample_'+str(idx)+'_pre.png')
        else:
            plt.savefig('vis/md17/sample_'+str(idx)+'_gt.png')

        plt.close()


if __name__ == "__main__":
    main()




