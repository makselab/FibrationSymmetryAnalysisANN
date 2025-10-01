#!/usr/bin/env -S uv run
import os
import copy
import glob

import torch
import metta

from kcore.coloring.networks import ColorPropagator

import matplotlib.pyplot as plt
import re

# ali@ubuntu22:~/Codes/metta/kcore$ export PYTHONPATH=/home/ali/Codes/metta

# --------------------------------------------------------

#osva_sept_8_final__e340__s332928000__t1013__sc0.pt
PATH = './train_dir/'
resPATH = './results_kcore/'
# exp_name = 'osva_sept_8_final/'
exp_name = 'osva_sept_8_final_epoch_15730_base_0.7_0.5_1.0_0.8'

# --------------------------------------------------------

inputs_net = {'td':{'env_obs':torch.randn(1, 3, 24, 200)},
            'state':torch.randn(4, 1, 128),
            'getitem_5': torch.randn(1, 24, 11, 11)}

thr_lstm = torch.Tensor([0.8]) # torch.linspace(0, 3.0, steps=101)
thr_cnn = torch.Tensor([0.5]) # torch.linspace(0, 3.0, steps=11)
thr_l = torch.Tensor([0.7]) # torch.linspace(0, 3.0, steps=4)

results = {'epoch':[], 'reduction':[], 'steps':[]}

# --------------------------------------------------------
epochs = []
reduction = []

for idx_epoch in range(15730,30190,10):
    # osva_sept_8_final_epoch_15730_base_0.7_0.5_1.0_0.8__e30190__s29562048000__t97464__sc0.pt

    pattern_filename = os.path.join(PATH,exp_name,exp_name,'checkpoints', f"{exp_name}__e{idx_epoch}__*__sc0.pt")
    matches = glob.glob(pattern_filename)

    filename = matches[0]

    match_step = re.search(r"__s(\d+)__", filename)
    time_steps = int(match_step.group(1))

    model = torch.load(filename , weights_only=False)
    num_params = sum(p.numel() for p in model.parameters())
    
    for kk, cc in model.policy.components.items():
        cc._traced=False

    # Coloring -------------------------------------------
    propagator= ColorPropagator(model)
    propagator.coloring(inputs_net,
                threshold={'linear':0.7, 
                'cnn': 0.5, 
                'lstm':1.0,
                'critic':0.8}, 
                origin_node='policy_components_cnn1__net_0')            

    # Collapse -------------------------------------------

    base_model = propagator.collapse_model()

    num_params_base = sum(p.numel() for p in base_model.parameters())
    per_reduction = num_params_base/num_params

    results['epoch'].append(idx_epoch)
    results['steps'].append(time_steps)
    results['reduction'].append(per_reduction)

    print(idx_epoch, per_reduction)

torch.save(results, 'symm_vs_time_steps_final_collapsed.pth')