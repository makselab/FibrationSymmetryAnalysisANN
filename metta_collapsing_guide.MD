FIBRATION SYMMETRIES IN METTA AGENTS: STEP-BY-STEP GUIDE
=================================================

This guide explains how to use the fibration-based model collapsing features 
built into Metta agents to compress neural network layers and compare performance 
before and after compression. You'll run training sessions and visualize 
the effects of collapsing based on fibration symmetries.

STEP 1: RUN BASELINE TRAINING
-----------------------------
Run the original training using commands of Metta. For example,

    ./tools/run.py experiments.recipes.arena_basic_easy_shaped.train --args run='exp_name' --overrides policy_architecture.name='fast' trainer.evaluation.evaluate_interval=0 trainer.checkpoint.checkpoint_interval=10 trainer.checkpoint.wandb_checkpoint_interval=10 trainer.evaluation.replay_dir=‘exp_name’ trainer.evaluation.skip_git_check=true

This command:
- Trains the agent in the "arena_basic_easy_shaped" environment
- Stores model checkpoints under a subdirectory in `train_dir/exp_name` (checkpoint_interval = 10)
- Stores metrics in WANDB under the name `exp_name` (wandb_checkpoint_interval = 10)


STEP 2: RUN COLLAPSE AND RE-TRAINING
------------------------------

Run a second training session with the collapse mechanism enabled:

    bash ./kcore.sh

Set up the next parameters:

- exp_name (str).                            Name of the model to collapse
- epoch_idx (int).                           Epoch of training when you want to collapse
- linear_thr, cnn_thr, 
  lstm_thr, critic_thr (floats 0-2)          Threshold for the fibration calculations
                                            (default = 0.7, 0.5, 1.0, 0.8)

This code automatically creates a collapsed version of the model and retrains it.
The output will be saved as a Metta experiment with the name 'base_name':

    exp_name_epoch_epoch_idx_base_linear_thr_cnn_thr_lstm_thr_critic_thr

More information about this script is inside the functions/scripts.

STEP 3: TEST 1 - RUN RANDOM ABLATION AND RE-TRAINING
------------------------------

Use the script 

    bash ./kcore_ablation.sh

and set up the same parameters from the step 2:

- exp_name (str).                            Name of the model to collapse
- epoch_idx (int).                           Epoch of training when you want to collapse
- linear_thr, cnn_thr, 
 lstm_thr, critic_thr (floats 0-2)          Threshold for the fibration calculations
                                            (default = 0.7, 0.5, 1.0, 0.8)

to apply ablation to the original model at epoch 'epoch_idx'.

This new model has the same number of nodes of the collapsed model (using the thresholds defined).
The output will be saved as a Metta experiment with the name 'ablation_name':

    exp_name_epoch_epoch_idx_ablationsrandomlinear_thr_cnn_thr_lstm_thr_critic_thr

More information about this script is inside the functions/scripts.

STEP 4: TEST 2 - RUN RANDOM MODEL FROM SCRATCH
------------------------------

Once you obtain the collapsed model, you will know its dimensions. 
You can define a new agent in the  './agent/src/metta/agent/component_policies/'; e.g, 'fast_small.py'.

Then, you can run the training using commands of Metta:

    ./tools/run.py experiments.recipes.arena_basic_easy_shaped.train --args run='random_small' --overrides policy_architecture.name='fast_small' trainer.evaluation.evaluate_interval=0 trainer.checkpoint.checkpoint_interval=10 trainer.checkpoint.wandb_checkpoint_interval=10 trainer.evaluation.replay_dir=‘exp_name’ trainer.evaluation.skip_git_check=true


VALIDATION RESULTS (VISUALIZE AND COMPARE RESULTS BETWEEN ORIGINAL MODEL, COLLAPSED MODEL AND ABLATION MODEL)
------------------------------

Steps 3 and 4 should reproduce the results in the file './validation.pdf'.

1. The green curve is the original model (default) trained up to 15G agent steps (50k train_time).

2. The red curve is the collapsed model via fibration to 19% (reduction of 81%) of the original size of the model. 
We then perform the two randomization protocols. 

3. Orange curve: We ablate 81% nodes at random, and keep the weights of the original 
model trained at 15G. We keep training. 

4. Purple curve: We train from scratch the small model, 19% of the original size, by initially randomizing all weights.


Use the interface in WANDB for compare metrics and results. 


FOLDER STRUCTURE EXAMPLE
-------------------------

    train_dir/
    ├── exp_name/                    # Original model (1st step)
    │   ├── config.json             
    │   └── exp_name/
    │       └── checkpoints/        
    │           ├── exp_name__eXX__sSS__tTT__sc0.pt
    │           └── ...
    │
    ├── base_name/                  # Collapsed model (2nd step)
    │   ├── config.json
    │   └── base_name/
    │       └── checkpoints/
    │           └── ...
    │
    ├── ablation_name/              # Ablation model (3rd step)
    │   ├── config.json
    │   └── ablation_name/
    │       └── checkpoints/
    │           └── ...
    └── ...


DISCLAIMER
----------
- Choose meaningful thresholds.
- Always verify WANDB and fiber output to validate results.
