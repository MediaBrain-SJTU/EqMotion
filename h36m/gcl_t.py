from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math

class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class Feature_learning_layer(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, input_c, hidden_c, output_c, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, norm_diff=False, tanh=False,apply_reasoning=True, output_reasoning=False, input_reasoning=False,category_num=2):
        super(Feature_learning_layer, self).__init__()
        self.norm_diff = norm_diff
        self.coord_vel = nn.Linear(hidden_c,hidden_c,bias=False)
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.hidden_c = hidden_c
        edge_coords_nf = hidden_c
        self.hidden_nf = hidden_nf

        one_coord_weight = False
        if one_coord_weight:
            layer = nn.Linear(hidden_nf, 1, bias=False)
        else:
            layer = nn.Linear(hidden_nf, hidden_c, bias=False)

        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.clamp = False
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        self.category_num = category_num

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge +edge_coords_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.category_mlp = []
        for i in range(category_num-2):
            self.category_mlp.append(nn.Sequential(
                nn.Linear(input_edge +edge_coords_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_c),
                act_fn))
        self.category_mlp = nn.ModuleList(self.category_mlp)

        self.factor_mlp = nn.Sequential(
            nn.Linear(hidden_c, hidden_c),
            act_fn,
            nn.Linear(hidden_c, hidden_c),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
        
        self.add_non_linear = True

        if self.add_non_linear:
            self.layer_q = nn.Linear(hidden_c,hidden_c,bias=False)
            self.layer_k = nn.Linear(hidden_c,hidden_c,bias=False)
        
        self.add_inner_agent_attention = True

        if self.add_inner_agent_attention:
            self.mlp_q = nn.Sequential(
                        nn.Linear(hidden_nf, int(hidden_c)),
                        act_fn)

    def edge_model(self, h, coord, edge_attr=None):
        batch_size, agent_num, channels = coord.shape[0], coord.shape[1], coord.shape[2]
        h1 = h[:,:,None,:].repeat(1,1,agent_num,1)
        h2 = h[:,None,:,:].repeat(1,agent_num,1,1)
        coord_diff = coord[:,:,None,:,:] - coord[:,None,:,:,:]
        coord_dist = torch.norm(coord_diff,dim=-1)
        edge_feat = torch.cat([h1,h2,coord_dist],dim=-1)
        edge_feat = self.edge_mlp(edge_feat)
        return edge_feat, coord_diff #(B,N,N,D)

    def aggregate_coord_reasoning(self, coord, edge_feat, coord_diff, category,h):
        batch_size, agent_num, channels = coord.shape[0], coord.shape[1], coord.shape[2]
        h1 = h[:,:,None,:].repeat(1,1,agent_num,1)
        h2 = h[:,None,:,:].repeat(1,agent_num,1,1)
        coord_dist = torch.norm(coord_diff,dim=-1)
        edge_h =  torch.cat([h1,h2,coord_dist],dim=-1)
        factors = torch.zeros(batch_size,agent_num,agent_num,channels).type_as(coord)
        for i in range(self.category_num-2):
            factors += (category[:,:,:,i:i+1]*self.category_mlp[i](edge_h))
        factors = self.factor_mlp(factors)

        factors = factors.unsqueeze(-1)
        neighbor_effect = torch.sum(factors * coord_diff, dim=2)
        coord = coord + neighbor_effect
        return coord

    def node_model(self, x, edge_feat):
        batch_size, agent_num = edge_feat.shape[0], edge_feat.shape[1]
        mask = (torch.ones((agent_num,agent_num)) - torch.eye(agent_num)).type_as(edge_feat)
        mask = mask[None,:,:,None].repeat(batch_size,1,1,1)
        aggregated_edge = torch.sum(mask*edge_feat,dim=2)
        out = self.node_mlp(torch.cat([x,aggregated_edge],dim=-1))

        if self.recurrent:
            out = x + out
        return out

    def inner_agent_attention(self,coord,h):
        att = self.mlp_q(h).unsqueeze(-1)
        v = coord - torch.mean(coord,dim=(1,2),keepdim=True)
        out = att * v
        apply_res = True
        if apply_res:
            out = out + coord
        return out

    def non_linear(self, coord):
        coord_mean = torch.mean(coord,dim=(1,2),keepdim=True)
        coord = coord - coord_mean
        q = self.layer_q(coord.transpose(2,3)).transpose(2,3)
        k = self.layer_k(coord.transpose(2,3)).transpose(2,3)
        product = torch.matmul(q.unsqueeze(-2),k.unsqueeze(-1)).squeeze(-1) # (B,N,C,1)
        mask = (product >= 0).float() # (B,N,C,1)
        EPS = 1e-4
        k_norm_sq = torch.sum(k*k,dim=-1,keepdim=True) # (B,N,C,1)
        coord = mask * q + (1-mask) * (q-(product/(k_norm_sq+EPS))*k)
        coord = coord + coord_mean
        return coord

    def forward(self, h, coord, vel, edge_attr=None, node_attr=None,category=None): 
        edge_feat, coord_diff = self.edge_model(h,coord,edge_attr)

        if self.add_inner_agent_attention:
            coord = self.inner_agent_attention(coord,h)
        coord = self.aggregate_coord_reasoning(coord, edge_feat, coord_diff, category,h)
        coord += self.coord_vel(vel.transpose(2,3)).transpose(2,3)

        if self.add_non_linear:
            coord = self.non_linear(coord)

        h = self.node_model(h, edge_feat)

        return h, coord, category
