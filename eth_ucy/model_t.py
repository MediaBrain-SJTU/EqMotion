import torch
from torch import nn
from eth_ucy.gcl_t import Feature_learning_layer
import numpy as np
from torch.nn import functional as F


class EqMotion(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, in_channel, hid_channel, out_channel, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, recurrent=False, norm_diff=False, tanh=False):
        super(EqMotion, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        self.embedding = nn.Linear(in_node_nf, int(self.hidden_nf/2))
        self.embedding2 = nn.Linear(in_node_nf, int(self.hidden_nf/2))

        self.coord_trans = nn.Linear(in_channel, int(hid_channel), bias=False)
        self.vel_trans = nn.Linear(in_channel, int(hid_channel), bias=False)

        self.apply_dct = True
        self.validate_reasoning = True
        self.in_channel = in_channel
        self.out_channel = out_channel

        category_num = 4
        self.category_num = category_num
        self.tao = 1

        self.given_category = False
        if not self.given_category:
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_nf*2+hid_channel*2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn)
            
            self.coord_mlp = nn.Sequential(
                nn.Linear(hid_channel*2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hid_channel*2),
                act_fn)

            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf+hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn)

            self.category_mlp = nn.Sequential(
                nn.Linear(hidden_nf*2+hid_channel*2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, category_num),
                act_fn)

        for i in range(0, n_layers-1):
            self.add_module("gcl_%d" % i, Feature_learning_layer(self.hidden_nf, self.hidden_nf, self.hidden_nf, in_channel, hid_channel, out_channel, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh, apply_reasoning=False,input_reasoning=True,category_num=category_num))

        self.predict_head = []
        for i in range(20):
            self.add_module("head_%d" % i, Feature_learning_layer(self.hidden_nf, self.hidden_nf, self.hidden_nf, in_channel, hid_channel, out_channel, edges_in_d=in_edge_nf, act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent, norm_diff=norm_diff, tanh=tanh, apply_reasoning=False,input_reasoning=True,category_num=category_num))
            self.predict_head.append(nn.Linear(hid_channel, out_channel, bias=False))
        self.predict_head = nn.ModuleList(self.predict_head)

        self.to(self.device)

    def get_dct_matrix(self,N,x):
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        dct_m = torch.from_numpy(dct_m).type_as(x)
        idct_m = torch.from_numpy(idct_m).type_as(x)
        return dct_m, idct_m
    
    def transform_edge_attr(self,edge_attr):
        edge_attr = (edge_attr / 2) + 1
        interaction_category = F.one_hot(edge_attr.long(),num_classes=self.category_num)
        return interaction_category

    def calc_category(self,h,coord,valid_mask):
        import torch.nn.functional as F
        batch_size, agent_num, channels = coord.shape[0], coord.shape[1], coord.shape[2]
        h1 = h[:,:,None,:].repeat(1,1,agent_num,1)
        h2 = h[:,None,:,:].repeat(1,agent_num,1,1)
        coord_diff = coord[:,:,None,:,:] - coord[:,None,:,:,:]
        coord_dist = torch.norm(coord_diff,dim=-1)
        coord_dist = self.coord_mlp(coord_dist)
        edge_feat_input = torch.cat([h1,h2,coord_dist],dim=-1)
        # edge_feat_input = coord_dist
        edge_feat = self.edge_mlp(edge_feat_input)
        mask = (torch.ones((agent_num,agent_num)) - torch.eye(agent_num)).type_as(edge_feat)
        mask = mask[None,:,:,None].repeat(batch_size,1,1,1)
        node_new = self.node_mlp(torch.cat([h,torch.sum(valid_mask*mask*edge_feat,dim=2)],dim=-1))
        node_new1 = node_new[:,:,None,:].repeat(1,1,agent_num,1)
        node_new2 = node_new[:,None,:,:].repeat(1,agent_num,1,1)

        edge_feat_input_new = torch.cat([node_new1,node_new2,coord_dist],dim=-1)
        interaction_category = F.softmax(self.category_mlp(edge_feat_input_new)/self.tao,dim=-1)

        return interaction_category

    def get_valid_mask(self,num_valid,agent_num):
        batch_size = num_valid.shape[0]
        valid_mask = torch.zeros((batch_size,agent_num,agent_num))
        for i in range(batch_size):
            valid_mask[i,:num_valid[i],:num_valid[i]] = 1
        return valid_mask.unsqueeze(-1)

    def get_valid_mask2(self,num_valid,agent_num):
        batch_size = num_valid.shape[0]
        valid_mask = torch.zeros((batch_size,agent_num))
        for i in range(batch_size):
            valid_mask[i,:num_valid[i]] = 1
        return valid_mask.unsqueeze(-1).unsqueeze(-1)

    def forward(self, h, x, vel, num_valid, edge_attr=None):
        # hinit = torch.zeros()
        vel_pre = torch.zeros_like(vel)
        vel_pre[:,:,1:] = vel[:,:,:-1] 
        vel_pre[:,:,0] = vel[:,:,0]
        EPS = 1e-6
        vel_cosangle = torch.sum(vel_pre*vel,dim=-1)/((torch.norm(vel_pre,dim=-1)+EPS)*(torch.norm(vel,dim=-1)+EPS))

        vel_angle = torch.acos(torch.clamp(vel_cosangle,-1,1))

        batch_size, agent_num, length = x.shape[0], x.shape[1], x.shape[2]

        valid_agent_mask = self.get_valid_mask2(num_valid,agent_num)
        valid_agent_mask = valid_agent_mask.type_as(h) # (B,N,1,1)

        if self.apply_dct:
            x_center = torch.mean(x*valid_agent_mask,dim=(1,2),keepdim=True) * (agent_num/num_valid[:,None,None,None])
            x = x - x_center
            dct_m,_ = self.get_dct_matrix(self.in_channel,x)
            _,idct_m = self.get_dct_matrix(self.out_channel,x)
            dct_m = dct_m[None,None,:,:].repeat(batch_size,agent_num,1,1)
            idct_m = idct_m[None,None,:,:].repeat(batch_size,agent_num,1,1)
            x = torch.matmul(dct_m,x)
            vel = torch.matmul(dct_m,vel)

        h = self.embedding(h)
        vel_angle_embedding = self.embedding2(vel_angle)
        h = torch.cat([h,vel_angle_embedding],dim=-1)

        x_mean = torch.mean(torch.mean(x*valid_agent_mask,dim=-2,keepdim=True),dim=-3,keepdim=True) * (agent_num/num_valid[:,None,None,None])
        x = self.coord_trans((x-x_mean).transpose(2,3)).transpose(2,3) + x_mean
        vel = self.vel_trans(vel.transpose(2,3)).transpose(2,3)
        x_cat = torch.cat([x,vel],dim=-2)

        valid_mask = self.get_valid_mask(num_valid,agent_num)
        valid_mask = valid_mask.type_as(h)

        cagegory_per_layer = []
        if self.given_category:
            category = self.transform_edge_attr(edge_attr)
        else:
            category = self.calc_category(h,x_cat,valid_mask)
        for i in range(0, self.n_layers-1):
            h, x, _ = self._modules["gcl_%d" % i](h, x, vel,valid_mask,valid_agent_mask,num_valid, edge_attr=edge_attr, category=category)
            # h, x, category = self._modules["gcl_%d" % i](h, x, vel, edge_attr=edge_attr)
            cagegory_per_layer.append(category)

        all_out = []
        for i in range(20):
            _, out, _ = self._modules["head_%d" % i](h, x, vel,valid_mask,valid_agent_mask,num_valid, edge_attr=edge_attr, category=category)
            out_mean = torch.mean(torch.mean(out*valid_agent_mask,dim=-2,keepdim=True),dim=-3,keepdim=True) * (agent_num/num_valid[:,None,None,None])
            out = self.predict_head[i]((out-out_mean).transpose(2,3)).transpose(2,3) + out_mean
            all_out.append(out[:,:,None,:,:])

        x = torch.cat(all_out,dim=2)
        x = x.view(batch_size,agent_num,20,self.out_channel,-1)

        if self.apply_dct:
            idct_m = idct_m[:,:,None,:,:]
            x = torch.matmul(idct_m,x)
            x = x + x_center.unsqueeze(2)
        if self.validate_reasoning:
            return x, cagegory_per_layer
        else:
            return x, None

class RF_vel(nn.Module):
    def __init__(self, hidden_nf, edge_attr_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4):
        super(RF_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        #self.reg = reg
        ### Encoder
        #self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL_rf_vel(nf=hidden_nf, edge_attr_nf=edge_attr_nf, act_fn=act_fn))
        self.to(self.device)


    def forward(self, vel_norm, x, edges, vel, edge_attr):
        for i in range(0, self.n_layers):
            x, _ = self._modules["gcl_%d" % i](x, vel_norm, vel, edges, edge_attr)
        return x

class Baseline(nn.Module):
    def __init__(self, device='cpu'):
        super(Baseline, self).__init__()
        self.dummy = nn.Linear(1, 1)
        self.device = device
        self.to(self.device)

    def forward(self, loc):
        return loc

class Linear(nn.Module):
    def __init__(self, input_nf, output_nf, device='cpu'):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_nf, output_nf)
        self.device = device
        self.to(self.device)

    def forward(self, input):
        return self.linear(input)

class Linear_dynamics(nn.Module):
    def __init__(self, device='cpu'):
        super(Linear_dynamics, self).__init__()
        self.time = nn.Parameter(torch.ones(1)*0.7)
        self.device = device
        self.to(self.device)

    def forward(self, x, v):
        return x + v*self.time
    

class RecurrentBaseline(nn.Module):
    """LSTM model for joint trajectory prediction."""
    def __init__(self, n_in, n_hid, n_out, n_atoms, n_layers, do_prob=0.):
        super(RecurrentBaseline, self).__init__()
        self.fc1_1 = nn.Linear(n_in, n_hid)
        self.fc1_2 = nn.Linear(n_hid, n_hid)
        self.n_hid = n_hid
        self.num_layers = n_layers
        self.rnn = nn.LSTM(n_hid, n_hid, n_layers,batch_first=True)
        self.fc2_1 = nn.Linear(n_hid, n_hid)
        self.fc2_2 = nn.Linear(n_hid, n_out*20)

        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)


    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_timesteps, n_in]

        batch = inputs.shape[0]
        actors = inputs.shape[1]
        inputs = inputs.view(-1,inputs.shape[2],inputs.shape[3])
        # h0 = torch.zeros(self.num_layers, inputs.size(0), self.n_hid).cuda()
        # c0 = torch.zeros(self.num_layers, inputs.size(0), self.n_hid).cuda()

        x = F.relu(self.fc1_1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.relu(self.fc1_2(x))
        # x = x.view(ins.size(0), -1)

        # out, (h_n, h_c) = self.rnn(x, (h0, c0))
        out, (h_n, h_c) = self.rnn(x)

        hidden = out[:, -1, :]
        x = F.relu(self.fc2_1(hidden))
        x = self.fc2_2(x)
        x = x.view(batch,actors,-1,3)

        return x,hidden
