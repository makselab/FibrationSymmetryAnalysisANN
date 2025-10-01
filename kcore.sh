#!/bin/bash

export PYTHONPATH=/home/ali/Codes/metta

# ===========================================================
# Initial Configuration - Parameters
PATHtrain='./train_dir/'
exp_name='osva_sept_8_final'
epoch_idx=15730
#3160

linear_thr=0.7
cnn_thr=0.5
lstm_thr=1.0
critic_thr=0.8

# ===========================================================

# Set up - Paths/Folders/Files
checkpoints_folder=$PATHtrain$exp_name'/'$exp_name'/checkpoints/'
model_checkpoint=$(find "$checkpoints_folder" -name "${exp_name}__e${epoch_idx}__s*__t*__sc0.pt" | head -n 1)
checkpoint=$(basename "$model_checkpoint")

pattern=$(echo "$checkpoint" | grep -o '__e[0-9]\+__s[0-9]\+__t[0-9]\+__sc0')
step_idx=$(echo "$checkpoint" | grep -o '__s[0-9]\+' | cut -d's' -f2)
t_idx=$(echo "$checkpoint" | grep -o '__t[0-9]\+' | cut -d't' -f2)

trainer_state=$checkpoints_folder'trainer_state.pt'

# ===========================================================

# Collapse Model (Arguments explained inside of the script)

./kcore/collapse_model.py -path $PATHtrain\
						  -exp $exp_name\
						  -cfgstr $pattern\
						  -epoch $epoch_idx\
						  -lin $linear_thr\
						  -cnn $cnn_thr\
						  -lstm $lstm_thr\
						  -crit $critic_thr

# ===========================================================

# Optimizer: Adapting the optimizer dimension to the new model dimension

base_name=$exp_name'_epoch_'$epoch_idx'_base_'$linear_thr'_'$cnn_thr'_'$lstm_thr'_'$critic_thr
base_folder=$PATHtrain$base_name'/'$base_name'/checkpoints/'
base_path=$base_folder$base_name$pattern'.pt'

cp "$trainer_state" "$base_folder"
optimizer_path=$base_folder'trainer_state.pt'

./kcore/trainer_state.py -base $base_path -opt $optimizer_path \
						-epoch $epoch_idx -step $step_idx


# ===========================================================

# Run Training of Collapse Model (base_name)

./tools/run.py experiments.recipes.arena_basic_easy_shaped.train\
			--args run="${base_name}"\
			--overrides trainer.evaluation.evaluate_interval=0\
						trainer.checkpoint.checkpoint_interval=10\
						trainer.checkpoint.wandb_checkpoint_interval=10\
						trainer.evaluation.replay_dir="${base_name}"\
						trainer.evaluation.skip_git_check=true