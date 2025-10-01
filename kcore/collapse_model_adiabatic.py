#!/usr/bin/env -S uv run

import torch
import metta

import os
import copy
import argparse

from kcore.coloring.networks_adiabatic import ColorPropagator

# --------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Collapse Model',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=''' Example: collapse_model.py -m model_path -f archivo.txt''')

parser.add_argument('-path', '--PATH', type=str, dest='path', help='Training path')
parser.add_argument('-exp' , '--exp_name', type=str, dest='exp_name', help='ExpName')
parser.add_argument('-cfgstr','--cfgstr', type=str, dest='cfgstr', help='Epoch-Time-Steps')
parser.add_argument('-epoch' , '--epoch_idx', type=int, dest='epoch', help='Epoch of collapse')
parser.add_argument('-step' , '--step_red', type=int, dest='step', help='Steps')
parser.add_argument('-lin' , '--linear_red', type=int, dest='linear_red', help='Num Nodes to reduce Lineal')
parser.add_argument('-cnn' , '--cnn_red', type=int, dest='cnn_red', help='Num Nodes to reduce CNN')
parser.add_argument('-lstm', '--lstm_red', type=int, dest='lstm_red', help='Num Nodes to reduce LSTM')
parser.add_argument('-crit', '--critic_red', type=int, dest='critic_red', help='Num Nodes to reduce Critic')

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
            threshold={'linear':args.linear_red, 
                        'cnn': args.cnn_red, 
                        'lstm':args.lstm_red,
                        'critic':args.critic_red}, 
            origin_node='policy_components_cnn1__net_0')
            
# for ll in reduction_colors_model.keys():
#     colors_ll = propagator.colors[ll]
#     size_ll = len(colors_ll)
#     num_colors_ll = len(torch.unique(colors_ll))
#     reduction_ll = num_colors_ll/size_ll

#     print(ll,reduction_ll)

# Collapse -------------------------------------------
print('Collapsing: ', model_path)

base_model = propagator.collapse_model()
num_params_base = sum(p.numel() for p in base_model.parameters())
per_reduction = num_params_base/num_params

print('Thresholds: ', args.linear_red,args.cnn_red,args.lstm_red,args.critic_red, \
    'Reduction: ', per_reduction)

# Transfer Extra Information (MettaAgent) -----------
base_model.policy.components._core_.hidden_size = len(torch.unique(propagator.colors['policy_components__core___net']))
base_model.reduction = per_reduction
base_model.threshold = (args.linear_red,args.cnn_red,args.lstm_red,args.critic_red)

# Saving data ---------------------------------------

if 'time' not in args.exp_name:
    new_exp_name = args.exp_name + '_epoch_' + str(args.epoch) + '_base_' + \
                    str(args.linear_red) + '_' +\
                    str(args.cnn_red) + '_' +\
                    str(args.lstm_red) + '_' +\
                    str(args.critic_red) +'_time_' + str(args.step)
else:
    new_exp_name = f"{args.exp_name.split('_time_')[0]}_time_{args.step}"

new_folder = args.path + new_exp_name + '/' + new_exp_name + '/checkpoints/'
base_model_path =  new_folder + new_exp_name + args.cfgstr + '.pt'
base_model.uri = 'file://' + base_model_path

print('Base: ', base_model)
print('Base Path: ', base_model_path)

if not os.path.exists(new_folder): os.makedirs(new_folder)
torch.save(base_model, base_model_path)

# =======================================================================