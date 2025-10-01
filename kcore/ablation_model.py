#!/usr/bin/env -S uv run

import torch
import metta

import os
import copy
import argparse

from kcore.coloring.networks import ColorPropagator

import torch.nn as nn
# --------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Collapse Model',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=''' Example: collapse_model.py -m model_path -f archivo.txt''')

parser.add_argument('-path', '--PATH', type=str, dest='path', help='Training path')
parser.add_argument('-exp' , '--exp_name', type=str, dest='exp_name', help='ExpName')
parser.add_argument('-cfgstr','--cfgstr', type=str, dest='cfgstr', help='Epoch-Time-Steps')
parser.add_argument('-epoch' , '--epoch_idx', type=int, dest='epoch', help='Epoch of collapse')
parser.add_argument('-lin' , '--linear_thr', type=float, dest='linear_thr', help='Threshold Lineal')
parser.add_argument('-cnn' , '--cnn_thr', type=float, dest='cnn_thr', help='Threshold CNN')
parser.add_argument('-lstm', '--lstm_thr', type=float, dest='lstm_thr', help='Threshold LSTM')
parser.add_argument('-crit', '--critic_thr', type=float, dest='critic_thr', help='Threshold Criticd')

args = parser.parse_args()

# --------------------------------------------------------
folder = args.path + args.exp_name + '/' + args.exp_name + '/checkpoints/'
model_path =  folder + args.exp_name + args.cfgstr + '.pt'
model = torch.load(model_path , weights_only=False)
num_params = sum(p.numel() for p in model.parameters())
print('Num_Params:', num_params)

# # --------------------------------------------------------

inputs_net = {'td':{'env_obs':torch.randn(1, 3, 24, 200)},
            'state':torch.randn(4, 1, 128),
            'getitem_5': torch.randn(1, 24, 11, 11)}

# --------------------------------------------------------

# # If you want to check, the colors:
# reduction_colors_model = {
#     'policy_components_cnn1__net_0':[], 
#     'policy_components_cnn2__net_0':[], 
#     'policy_components_fc1__net_0':[], 
#     'policy_components_encoded_obs__net_0':[],
#     'policy_components__core___net':[], 
#     'policy_components_critic_1__net_0':[], 
#     'policy_components_actor_1__net_0':[]}

# --------------------------------------------------------

for kk, cc in model.policy.components.items():
    cc._traced=False

# Coloring -------------------------------------------
propagator= ColorPropagator(model)
propagator.coloring(inputs_net,
            threshold={'linear':args.linear_thr, 
                        'cnn': args.cnn_thr, 
                        'lstm':args.lstm_thr,
                        'critic':args.critic_thr}, 
            origin_node='policy_components_cnn1__net_0')

num_colors_cnn1 = len(torch.unique(propagator.colors['policy_components_cnn1__net_0'])) 
num_colors_fc1 = len(torch.unique(propagator.colors['policy_components_fc1__net_0']))
num_colors_encoded = len(torch.unique(propagator.colors['policy_components_encoded_obs__net_0']))
num_colors_core = len(torch.unique(propagator.colors['policy_components__core___net']))
num_colors_critic = len(torch.unique(propagator.colors['policy_components_critic_1__net_0']))

per_nodes_red = (num_colors_cnn1 + num_colors_fc1 + num_colors_encoded +num_colors_core +num_colors_critic)/1472

# Ablation -------------------------------------------
print('Collapsing: ', model_path)

base_model = copy.deepcopy(model)

# -----------------------------------------------------

W = model.policy.components.cnn1._net[0].weight
bias = model.policy.components.cnn1._net[0].bias

N, M, _, _ = W.shape

num_nodes_out = int(per_nodes_red * N)
out_indices = torch.randperm(N)[:num_nodes_out]

base_model.policy.components.cnn1._net[0] = nn.Conv2d(
        in_channels=24,
        out_channels=num_nodes_out,
        kernel_size=model.policy.components.cnn1._net[0].kernel_size,
        stride=model.policy.components.cnn1._net[0].stride,
        padding=model.policy.components.cnn1._net[0].padding,
        dilation=model.policy.components.cnn1._net[0].dilation,
        groups=model.policy.components.cnn1._net[0].groups,
        bias=model.policy.components.cnn1._net[0].bias is not None,
        padding_mode=model.policy.components.cnn1._net[0].padding_mode, device=model.policy.components.cnn1._net[0].weight.device)

base_model.policy.components.cnn1._net[0].weight.data = W.data[out_indices]
base_model.policy.components.cnn1._net[0].bias.data = bias.data[out_indices]

# -----------------------------------------------------

W = model.policy.components.cnn2._net[0].weight
bias = model.policy.components.cnn2._net[0].bias

N, M, _, _ = W.shape

in_indices = out_indices

base_model.policy.components.cnn2._net[0] = nn.Conv2d(
        in_channels=len(in_indices),
        out_channels=64,
        kernel_size=model.policy.components.cnn2._net[0].kernel_size,
        stride=model.policy.components.cnn2._net[0].stride,
        padding=model.policy.components.cnn2._net[0].padding,
        dilation=model.policy.components.cnn2._net[0].dilation,
        groups=model.policy.components.cnn2._net[0].groups,
        bias=model.policy.components.cnn2._net[0].bias is not None,
        padding_mode=model.policy.components.cnn2._net[0].padding_mode, device=model.policy.components.cnn2._net[0].weight.device)

base_model.policy.components.cnn2._net[0].weight.data = W.data[:,in_indices]
base_model.policy.components.cnn2._net[0].bias.data = bias.data

# -----------------------------------------------------

W = model.policy.components.fc1._net[0].weight
bias = model.policy.components.fc1._net[0].bias

N, M = W.shape
num_nodes_out = int(N * per_nodes_red) 
out_indices = torch.randperm(N)[:num_nodes_out]

base_model.policy.components.fc1._net[0] = nn.Linear(64, num_nodes_out, device=base_model.policy.components.fc1._net[0].weight.device)

base_model.policy.components.fc1._net[0].weight.data = W.data[out_indices]
base_model.policy.components.fc1._net[0].bias.data = bias.data[out_indices]

# -----------------------------------------------------

W = model.policy.components.encoded_obs._net[0].weight
bias = model.policy.components.encoded_obs._net[0].bias

N, M = W.shape
num_nodes_out = int(per_nodes_red * N)

in_indices = out_indices
out_indices = torch.randperm(N)[:num_nodes_out]

base_model.policy.components.encoded_obs._net[0] = nn.Linear(len(in_indices), num_nodes_out, device=base_model.policy.components.encoded_obs._net[0].weight.device)
base_model.policy.components.encoded_obs._net[0].weight.data = W.data[out_indices][:, in_indices]
base_model.policy.components.encoded_obs._net[0].bias.data = bias.data[out_indices]

# -----------------------------------------------------

in_indices = out_indices
num_nodes_out = int(128 * per_nodes_red)
out_indices_core = torch.randperm(128)[:num_nodes_out]

vector_ = torch.cat([out_indices_core, out_indices_core + 128, out_indices_core + 256, out_indices_core + 384 ])

base_model.policy.components._core_._net = nn.LSTM(len(in_indices), num_nodes_out, num_layers=2)

for layer_idx in range(2):

    w_ih = getattr(model.policy.components._core_._net, f'weight_ih_l{layer_idx}').data
    w_hh = getattr(model.policy.components._core_._net, f'weight_hh_l{layer_idx}').data        
    b_ih = getattr(model.policy.components._core_._net, f'bias_ih_l{layer_idx}').data
    b_hh = getattr(model.policy.components._core_._net, f'bias_hh_l{layer_idx}').data

    w_ih_coll = w_ih.data[vector_][:,in_indices]
    w_hh_coll = w_hh.data[vector_][:,out_indices_core]
    b_ih_coll = b_ih.data[vector_]
    b_hh_coll = b_ih.data[vector_]

    setattr(base_model.policy.components._core_._net, f'weight_ih_l{layer_idx}', torch.nn.Parameter(w_ih_coll, requires_grad=True))
    setattr(base_model.policy.components._core_._net, f'weight_hh_l{layer_idx}', torch.nn.Parameter(w_hh_coll, requires_grad=True))
    setattr(base_model.policy.components._core_._net, f'bias_ih_l{layer_idx}'  , torch.nn.Parameter(b_ih_coll, requires_grad=True))
    setattr(base_model.policy.components._core_._net, f'bias_hh_l{layer_idx}'  , torch.nn.Parameter(b_hh_coll, requires_grad=True))

    in_indices = out_indices_core

# -----------------------------------------------------

W = model.policy.components.critic_1._net[0].weight
bias = model.policy.components.critic_1._net[0].bias

N, M = W.shape
num_nodes_out = int(per_nodes_red * N)

in_indices = out_indices_core
out_indices = torch.randperm(N)[:num_nodes_out]

base_model.policy.components.critic_1._net[0] = nn.Linear(len(in_indices), num_nodes_out, device=base_model.policy.components.critic_1._net[0].weight.device)
base_model.policy.components.critic_1._net[0].weight.data = W.data[out_indices][:,in_indices]
base_model.policy.components.critic_1._net[0].bias.data = bias.data[out_indices]

# -----------------------------------------------------

W = model.policy.components._value_._net.weight
bias = model.policy.components._value_._net.bias

N, M = W.shape
in_indices = out_indices

base_model.policy.components._value_._net = nn.Linear(len(in_indices), 1, device=base_model.policy.components._value_._net.weight.device)
base_model.policy.components._value_._net.weight.data = W.data[:,in_indices]
base_model.policy.components._value_._net.bias.data = bias.data

# -----------------------------------------------------

W = model.policy.components.actor_1._net[0].weight
bias = model.policy.components.actor_1._net[0].bias

N, M = W.shape
in_indices = out_indices_core

base_model.policy.components.actor_1._net[0] = nn.Linear(len(in_indices), 512, device=base_model.policy.components.actor_1._net[0].weight.device)
base_model.policy.components.actor_1._net[0].weight.data = W.data[:,in_indices]
base_model.policy.components.actor_1._net[0].bias.data = bias.data


# -----------------------------------------------------

print(base_model)
num_params_base = sum(p.numel() for p in base_model.parameters())
per_reduction = num_params_base/num_params

print('Thresholds: ', args.linear_thr,args.cnn_thr,args.lstm_thr,args.critic_thr, \
    'Reduction: ', per_reduction)

# Transfer Extra Information (MettaAgent) -----------
base_model.policy.components._core_.hidden_size = int(128 * per_nodes_red)
base_model.reduction = per_reduction
base_model.threshold = (args.linear_thr,args.cnn_thr,args.lstm_thr,args.critic_thr)

# Saving data ---------------------------------------
new_exp_name = args.exp_name + '_epoch_' + str(args.epoch) + '_ablationsrandom_' + \
                str(args.linear_thr) + '_' +\
                str(args.cnn_thr) + '_' +\
                str(args.lstm_thr) + '_' +\
                str(args.critic_thr)

new_folder = args.path + new_exp_name + '/' + new_exp_name + '/checkpoints/'
base_model_path =  new_folder + new_exp_name + args.cfgstr + '.pt'
base_model.uri = 'file://' + base_model_path

print('Base: ', base_model)
print('Base Path: ', base_model_path)

if not os.path.exists(new_folder): os.makedirs(new_folder)
torch.save(base_model, base_model_path)

# # =======================================================================