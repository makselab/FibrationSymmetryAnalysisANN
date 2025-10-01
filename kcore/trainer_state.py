#!/usr/bin/env -S uv run

import torch
import argparse

parser = argparse.ArgumentParser(
    description='Optimizer',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=''' Example: collapse_model.py -m model_path -f archivo.txt''')

parser.add_argument('-base', '--base_path', type=str, dest='base_path', help='Base path')
parser.add_argument('-opt' , '--opt_path', type=str, dest='opt_path', help='Optimizer')
parser.add_argument('-epoch', '--epoch_idx', type=int, dest='epoch_idx', help='Epoch')
parser.add_argument('-step' , '--step_idx', type=int, dest='step_idx', help='Step')

args = parser.parse_args()

# ===============================================
model = torch.load(args.base_path, weights_only=False)
trainer_state = torch.load(args.opt_path)

# ===============================================

trainer_state['epoch'] = args.epoch_idx
trainer_state['agent_step'] = args.step_idx

# ===============================================
#CheckInitial
# for param_idx, param in enumerate(model.parameters()):
# 	print(param_idx, param.shape, \
# 		trainer_state['optimizer']['state'][param_idx]['exp_avg'].shape, \
# 		trainer_state['optimizer']['state'][param_idx]['exp_avg_sq'].shape)

# ===============================================
for param_idx, param in enumerate(model.parameters()):
	if param.dim() == 1:
		new_shape = param.shape[0]
		trainer_state['optimizer']['state'][param_idx]['exp_avg'] = \
			trainer_state['optimizer']['state'][param_idx]['exp_avg'][:new_shape]
		trainer_state['optimizer']['state'][param_idx]['exp_avg_sq'] = \
			trainer_state['optimizer']['state'][param_idx]['exp_avg_sq'][:new_shape]
	if param.dim() == 2:
		new_shape_x, new_shape_y = param.shape
		trainer_state['optimizer']['state'][param_idx]['exp_avg'] = \
			trainer_state['optimizer']['state'][param_idx]['exp_avg'][:new_shape_x,:new_shape_y]
		trainer_state['optimizer']['state'][param_idx]['exp_avg_sq'] = \
			trainer_state['optimizer']['state'][param_idx]['exp_avg_sq'][:new_shape_x,:new_shape_y]
	if param.dim() == 4:
		new_shape_x, new_shape_y = param.shape[:2]
		trainer_state['optimizer']['state'][param_idx]['exp_avg'] = \
			trainer_state['optimizer']['state'][param_idx]['exp_avg'][:new_shape_x,:new_shape_y]
		trainer_state['optimizer']['state'][param_idx]['exp_avg_sq'] = \
			trainer_state['optimizer']['state'][param_idx]['exp_avg_sq'][:new_shape_x,:new_shape_y]

# ============================================
#CheckFinal
# for param_idx, param in enumerate(model.parameters()):
# 	print(param_idx, param.shape, \
# 		trainer_state['optimizer']['state'][param_idx]['exp_avg'].shape, \
# 		trainer_state['optimizer']['state'][param_idx]['exp_avg_sq'].shape)

torch.save(trainer_state,args.opt_path)