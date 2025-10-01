#!/bin/bash

export PYTHONPATH=/home/ali/Codes/metta

# ===========================================================
# Initial Configuration - Parameters
PATHtrain='./train_dir/'
exp_name='osva_sept_8_final'
epoch_idx=15020 #14980 #15730 #3160
#interval_epochs=200
# running_epochs=205
# num_agent_steps=$(($running_epochs * 1000000))

interval_epochs=(50 50 50 50 50 50 50 50 50 50 200 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000)

linear_num_nodes_per_epoch=5 #4
cnn_num_nodes_per_epoch=3
lstm_num_nodes_pero_epoch=2
critic_num_nodes_pero_epoch=46

# linear_thr=0.7
# cnn_thr=0.5
# lstm_thr=1.0
# critic_thr=0.8

# ===========================================================

base_name=$exp_name
collapse_epoch=$epoch_idx

for step_reduction in {1..25}; do

	int_epochs=${interval_epochs[step_reduction]}

	running_epochs=$(($int_epochs +5))
	num_agent_steps=$(($running_epochs * 1000000))

	# Set up - Paths/Folders/Files
	checkpoints_folder=$PATHtrain$base_name'/'$base_name'/checkpoints/'
	model_checkpoint=$(find "$checkpoints_folder" -name "${base_name}__e${collapse_epoch}__s*__t*__sc0.pt" | head -n 1)
	checkpoint=$(basename "$model_checkpoint")

	pattern=$(echo "$checkpoint" | grep -o '__e[0-9]\+__s[0-9]\+__t[0-9]\+__sc0')
	step_idx=$(echo "$checkpoint" | grep -o '__s[0-9]\+' | cut -d's' -f2)
	t_idx=$(echo "$checkpoint" | grep -o '__t[0-9]\+' | cut -d't' -f2)

	trainer_state=$checkpoints_folder'trainer_state.pt'

	end_step_idx=$(( step_idx + num_agent_steps))

	# ===========================================================	
	# Collapse Model
	./kcore/collapse_model_adiabatic.py -path $PATHtrain\
							  -exp $base_name\
							  -cfgstr $pattern\
							  -epoch $epoch_idx\
							  -step $step_reduction\
							  -lin $linear_num_nodes_per_epoch\
							  -cnn $cnn_num_nodes_per_epoch\
							  -lstm $lstm_num_nodes_pero_epoch\
							  -crit $critic_num_nodes_pero_epoch

	base_name=$exp_name'_epoch_'$epoch_idx'_base_'$linear_num_nodes_per_epoch'_'$cnn_num_nodes_per_epoch'_'$lstm_num_nodes_pero_epoch'_'$critic_num_nodes_pero_epoch'_time_'$step_reduction
	base_folder=$PATHtrain$base_name'/'$base_name'/checkpoints/'
	base_path=$base_folder$base_name$pattern'.pt'

	# ===========================================================

	# Optimizer

	cp "$trainer_state" "$base_folder"
	optimizer_path=$base_folder'trainer_state.pt'

	./kcore/trainer_state.py -base $base_path -opt $optimizer_path \
							-epoch $collapse_epoch -step $step_idx

	# ===========================================================

	# RunTraining

	./tools/run.py experiments.recipes.arena_basic_easy_shaped.train\
				--args run="${base_name}"\
				--overrides trainer.evaluation.evaluate_interval=0\
							trainer.checkpoint.checkpoint_interval=10\
							trainer.checkpoint.wandb_checkpoint_interval=10\
							trainer.evaluation.replay_dir="${base_name}"\
							trainer.evaluation.skip_git_check=true\
							trainer.total_timesteps=$end_step_idx

	collapse_epoch=$((collapse_epoch+int_epochs))

done