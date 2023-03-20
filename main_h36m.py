import argparse
import torch
from h36m.dataloader import H36motion3D
from h36m.model_t import EqMotion
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
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--past_length', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--future_length', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=-1, metavar='S',
                    help='random seed (default: -1)')
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
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--channels', type=int, default=72, metavar='N',
                    help='number of channels')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--div', type=float, default=1, metavar='N',
                    help='timing experiment')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--model_save_dir', type=str, default='h36m/saved_models',
                    help='Path to save models')
parser.add_argument('--scale', type=float, default=100, metavar='N',
                    help='data scale')
parser.add_argument('--model_name', type=str, default='ckpt_short',
                    help='Name of the model.')
parser.add_argument("--weighted_loss",action='store_true')
parser.add_argument("--apply_decay",action='store_true')
parser.add_argument("--debug",action='store_true')
parser.add_argument("--add_agent_token",action='store_true')
parser.add_argument('--category_num', type=int, default=4)
parser.add_argument("--test",action='store_true')

time_exp_dic = {'time': 0, 'counter': 0}

args = parser.parse_args()
args.cuda = True
args.add_agent_token = True
if args.future_length == 25:
    args.weighted_loss = True

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
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0,1000)
        setup_seed(seed)
    print('The seed is :',seed)

    past_length = args.past_length
    future_length = args.future_length

    if args.debug:
        dataset_train = H36motion3D(actions='walking', input_n=args.past_length, output_n=args.future_length, split=0, scale=args.scale)
    else:
        dataset_train = H36motion3D(actions='all', input_n=args.past_length, output_n=args.future_length, split=0, scale=args.scale)
   
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)

    acts = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]
    loaders_test = {}
    for act in acts:
        dataset_test = H36motion3D(actions=act, input_n=args.past_length, output_n=args.future_length, split=1, scale=args.scale)
        loaders_test[act] = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                num_workers=8)

    dim_used = dataset_train.dim_used

    model = EqMotion(in_node_nf=args.past_length, in_edge_nf=2, hidden_nf=args.nf, in_channel=args.past_length, hid_channel=args.channels, out_channel=args.future_length,device=device, n_layers=args.n_layers, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh,add_agent_token=args.add_agent_token,category_num=args.category_num)    

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    print(get_parameter_number(model))
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.test:
        model_path = args.model_save_dir + '/' + args.model_name +'.pth.tar'
        print('Loading model from:', model_path)
        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt['state_dict'], strict=False)
        if args.future_length == 25:
            avg_mpjpe = np.zeros((6))
        elif args.future_length == 15:
            avg_mpjpe = np.zeros((2))
        else:
            avg_mpjpe = np.zeros((4))

        for act in acts:
            mpjpe = test(model, optimizer, 0, (act, loaders_test[act]), dim_used, backprop=False)
            avg_mpjpe += mpjpe
        avg_mpjpe = avg_mpjpe / len(acts)
        print('avg mpjpe:',avg_mpjpe)
        return

    results = {'epochs': [], 'losess': []}
    best_test_loss = 1e8
    best_ade = 1e8
    best_epoch = 0
    lr_now = args.lr

    for epoch in range(0, args.epochs):
        if args.apply_decay:
            if epoch % args.epoch_decay == 0 and epoch > 0:
                lr_now = lr_decay(optimizer, lr_now, args.lr_gamma)
        train(model, optimizer, epoch, loader_train, dim_used)
        if epoch % args.test_interval == 0:
            if args.future_length == 25:
                avg_mpjpe = np.zeros((6))
            elif args.future_length == 15:
                avg_mpjpe = np.zeros((2))
            else:
                avg_mpjpe = np.zeros((4))
            for act in acts:
                mpjpe = test(model, optimizer, epoch, (act, loaders_test[act]), dim_used, backprop=False)
                avg_mpjpe += mpjpe
            avg_mpjpe = avg_mpjpe / len(acts)
            print('avg mpjpe:',avg_mpjpe)
            avg_mpjpe = np.mean(avg_mpjpe)

            if avg_mpjpe < best_test_loss:
                best_test_loss = avg_mpjpe
                best_epoch = epoch
                state = {'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()}
                if args.future_length == 25:
                    file_path = os.path.join(args.model_save_dir, 'ckpt_long_best.pth.tar')
                else:
                    file_path = os.path.join(args.model_save_dir, 'ckpt_best.pth.tar')
                torch.save(state, file_path)

            state = {'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()}
            if args.future_length == 25:
                file_path = os.path.join(args.model_save_dir, 'ckpt_long_'+str(epoch)+'.pth.tar')
            else:
                file_path = os.path.join(args.model_save_dir, 'ckpt_'+str(epoch)+'.pth.tar')
            torch.save(state, file_path)

            print("Best Test Loss: %.5f \t Best epoch %d" % (best_test_loss, best_epoch))
            print('The seed is :',seed)

    return


def train(model, optimizer, epoch, loader, dim_used=[], backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, length, _ = data[0].size()
        data = [d.to(device) for d in data]
        loc, vel, loc_end, _, item = data
        loc_start = loc[:,:,-1:]

        optimizer.zero_grad()

        nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
        loc_pred, category = model(nodes, loc.detach(), vel)

        if args.weighted_loss:
            weight = np.arange(1,5,(4/args.future_length))
            weight = args.future_length / weight
            # weight = weight / np.sum(weight)
            weight = torch.from_numpy(weight).type_as(loc_end)
            weight = weight[None,None]
            loss = torch.mean(weight*torch.norm(loc_pred-loc_end,dim=-1))
        else:
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
    print('%s epoch %d avg loss: %.5f' % (prefix+'train', epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']

def test(model, optimizer, epoch, act_loader,dim_used=[],backprop=False):

    act, loader = act_loader[0], act_loader[1]

    model.eval()

    validate_reasoning = False
    if validate_reasoning:
        acc_list = [0]*args.n_layers
    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'ade': 0}

    output_n = args.future_length
    if output_n == 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 15:
        eval_frame = [3, 14]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
    t_3d = np.zeros(len(eval_frame))

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            batch_size, n_nodes, length, _ = data[0].size()
            data = [d.to(device) for d in data]
            loc, vel, loc_end, loc_end_ori,_ = data
            loc_start = loc[:,:,-1:]
            pred_length = loc_end.shape[2]

            optimizer.zero_grad()

            nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
            loc_pred, _ = model(nodes, loc.detach(), vel)

            pred_3d = loc_end_ori.clone()
            loc_pred = loc_pred.transpose(1,2)
            loc_pred = loc_pred.contiguous().view(batch_size,loc_end.shape[2],n_nodes*3)
            
            joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
            index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
            joint_equal = np.array([13, 19, 22, 13, 27, 30])
            index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

            pred_3d[:,:,dim_used] = loc_pred
            pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
            pred_p3d = pred_3d.contiguous().view(batch_size, pred_length, -1, 3)#[:, input_n:, :, :]
            targ_p3d = loc_end_ori.contiguous().view(batch_size, pred_length, -1, 3)#[:, input_n:, :, :]

            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                t_3d[k] += torch.mean(torch.norm(targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2, 1)).item() * batch_size
            
            res['counter'] += batch_size
    t_3d *= args.scale
    N = res['counter']
    actname = "{0: <14} |".format(act)
    if args.future_length == 25:
        print('Act: {},  ErrT: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}, TestError {:.4f}'\
            .format(actname, 
                    float(t_3d[0])/N, float(t_3d[1])/N, float(t_3d[2])/N, float(t_3d[3])/N, float(t_3d[4])/N, float(t_3d[5])/N, 
                    float(t_3d.mean())/N))
    elif args.future_length == 15:
        print('Act: {},  ErrT: {:.3f} {:.3f}, TestError {:.4f}'\
            .format(actname, 
                    float(t_3d[0])/N, float(t_3d[1])/N, 
                    float(t_3d.mean())/N))
    else:
        print('Act: {},  ErrT: {:.3f} {:.3f} {:.3f} {:.3f}, TestError {:.4f}'\
            .format(actname, 
                    float(t_3d[0])/N, float(t_3d[1])/N, float(t_3d[2])/N, float(t_3d[3])/N, 
                    float(t_3d.mean())/N))

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    # print('%s epoch %d avg loss: %.5f ade: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter'], res['ade'] / res['counter']))
    return t_3d / N

if __name__ == "__main__":
    main()




