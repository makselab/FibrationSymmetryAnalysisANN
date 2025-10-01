MODEL COLLAPSING AND SYMMETRY BREAKING IN METTA AGENTS: STEP-BY-STEP GUIDE
=================================================

This guide walks through how to use the fibration-based model collapsing features 
built into Metta agents to compress neural network layers and compare performance 
before and after compression. You'll run two training sessions and visualize 
the effects of collapsing using fibrations.


STEP 1: RUN BASELINE TRAINING
-----------------------------
Run the original training using commands of Metta. For example,

    ./tools/run.py experiments.recipes.arena_basic_easy_shaped.train --args run='exp_name' --overrides policy_architecture.name='fast' trainer.evaluation.evaluate_interval=0 trainer.checkpoint.checkpoint_interval=10 trainer.checkpoint.wandb_checkpoint_interval=10 trainer.evaluation.replay_dir=‘exp_name’ trainer.evaluation.skip_git_check=true

This command:
- Trains the agent in the "arena_basic_easy_shaped" environment
- Stores model checkpoints under a subdirectory in `train_dir/exp_name` (checkpoint_interval = 10)
- Stores metrics in WANDB under the name `exp_name` (wandb_checkpoint_interval = 10)


STEP 2: RUN COLLAPSED TRAINING
------------------------------

Run a second training session with the collapse mechanism enabled:

    bash ./kcore.sh

Set up the next parameters:

-exp_name (str).                            Name of the model to collapse
-epoch_idx (int).                           Epoch of training when you want to collapse
-linear_thr, cnn_thr, 
 lstm_thr, critic_thr (floats 0-2)          Threshold for the fibration calculations

This code automatically creates a collapsed version of the model and retrains it.
The output will be saved as a Metta experiment with the name 'base_name':

    exp_name_epoch_epoch_idx_base_linear_thr_cnn_thr_lstm_thr_critic_thr

More information about this script is inside the functions/scripts.

STEP 3: VISUALIZE AND COMPARE RESULTS
-------------------------------------
Use the interface in WANDB for compare metrics and results. 


FOLDER STRUCTURE EXAMPLE
-------------------------
train_dir/
├── exp_name/
│   ├── config.json
│   └── exp/
│       └── checkpoints/
│           ├── exp_name__eXX__sSS__tTT__sc0.pt


DISCLAIMER
----------
- The --collapse feature must be implemented in your local fork of PufferLib.
- Choose meaningful thresholds.
- Always verify WANDB and fiber output to validate results.
