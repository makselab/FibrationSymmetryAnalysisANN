#!/bin/bash

export PYTHONPATH=/home/ali/Codes/metta

# ===========================================================
# Initial Configuration - Parameters
PATHtrain='./train_dir/'
exp_name='Prueba'
interval_epoch=20 

linear_thr=0.7
cnn_thr=0.5
lstm_thr=1.0
critic_thr=0.8

exp_name=$exp_name'_'$linear_thr'_'$cnn_thr'_'$lstm_thr'_'$critic_thr

# interval_epochs=(50 50 50 50 50 50 50 50 50 50 200 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000)
# linear_num_nodes_per_epoch=5 #4
# cnn_num_nodes_per_epoch=3
# lstm_num_nodes_pero_epoch=2
# critic_num_nodes_pero_epoch=46

# ===========================================================
step_idx=0
bk_epoch=0

for step_bk in {0..0}; do

	base_name=$exp_name'_'$step_bk

	# RunTraining
	num_agent_steps=$(( (interval_epoch + 5) * 1000000 ))
	end_step_idx=$(( step_idx + num_agent_steps))

	# ./tools/run.py experiments.recipes.arena_basic_easy_shaped.train\
	# 			--args run="${base_name}"\
	# 			--overrides trainer.evaluation.evaluate_interval=0\
	# 						trainer.checkpoint.checkpoint_interval=10\
	# 						trainer.checkpoint.wandb_checkpoint_interval=10\
	# 						trainer.evaluation.replay_dir="${base_name}"\
	# 						trainer.evaluation.skip_git_check=true\
	# 						trainer.total_timesteps=$end_step_idx
	
	bk_epoch=$((bk_epoch+interval_epoch))

	# Set up - Paths/Folders/Files
	checkpoints_folder=$PATHtrain$base_name'/'$base_name'/checkpoints/'
	model_checkpoint=$(find "$checkpoints_folder" -name "${base_name}__e${bk_epoch}__s*__t*__sc0.pt" | head -n 1)
	checkpoint=$(basename "$model_checkpoint")

	pattern=$(echo "$checkpoint" | grep -o '__e[0-9]\+__s[0-9]\+__t[0-9]\+__sc0')
	step_idx=$(echo "$checkpoint" | grep -o '__s[0-9]\+' | cut -d's' -f2)
	t_idx=$(echo "$checkpoint" | grep -o '__t[0-9]\+' | cut -d't' -f2)

	trainer_state=$checkpoints_folder'trainer_state.pt'

	# ===========================================================	
	# Collapse Model
	./kcore/breaking_symmetry.py -path $PATHtrain\
							  -exp $base_name\
							  -cfgstr $pattern\
							  -step $step_bk\
							  -lin $linear_thr\
							  -cnn $cnn_thr\
							  -lstm $lstm_thr\
							  -crit $critic_thr

	# base_folder=$PATHtrain$base_name'/'$base_name'/checkpoints/'
	# base_path=$base_folder$base_name$pattern'.pt'

# # 	# ===========================================================

# # 	# Optimizer

# # 	cp "$trainer_state" "$base_folder"
# # 	optimizer_path=$base_folder'trainer_state.pt'

# # 	./kcore/trainer_state.py -base $base_path -opt $optimizer_path \
# # 							-epoch $collapse_epoch -step $step_idx

# # 	# ===========================================================


done