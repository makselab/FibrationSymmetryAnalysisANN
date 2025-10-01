FIBRATION SYMMETRIES IN METTA AGENTS: STEP-BY-STEP GUIDE
=================================================

This guide explains how to use the fibration-based model collapsing features 
built into Metta agents to compress neural network layers and compare performance 
before and after compression and perform validation against random models. You'll run training sessions and visualize 
the effects of collapsing based on fibration symmetries.

STEP 1: RUN DEFAULT TRAINING MODEL
-----------------------------
Run the original training using Metta's commands. For example,

    ./tools/run.py experiments.recipes.arena_basic_easy_shaped.train --args run='exp_name' --overrides policy_architecture.name='fast' trainer.evaluation.evaluate_interval=0 trainer.checkpoint.checkpoint_interval=10 trainer.checkpoint.wandb_checkpoint_interval=10 trainer.evaluation.replay_dir=‘exp_name’ trainer.evaluation.skip_git_check=true

This command:
- Trains the agent in the "arena_basic_easy_shaped" environment
- Stores model checkpoints under a subdirectory in `train_dir/exp_name` (checkpoint_interval = 10)
- Stores metrics in WANDB under the name `exp_name` (wandb_checkpoint_interval = 10)


STEP 2: RUN COLLAPSE AND RE-TRAINING
------------------------------

This code automatically creates a collapsed version of the model and retrains it.

Run a second session with the collapse mechanism enabled:

    bash ./kcore.sh

Set up the following parameters to collapse the original default model:

- exp_name (str).                            Name of the model to collapse
- epoch_idx (int).                           Epoch of training when you want to collapse
- linear_thr, cnn_thr, 
  lstm_thr, critic_thr (floats 0-2)          Threshold for the fibration calculations. Threshold defines how much to collapse. thr=0 means no collapse. thr=2 maximum collapse into one node 
                                            (default = 0.7, 0.5, 1.0, 0.8)


The output will be saved as a Metta experiment with the name 'base_name':

    exp_name_epoch_epoch_idx_base_linear_thr_cnn_thr_lstm_thr_critic_thr

More information about this script is inside the functions/scripts.

STEP 3: TEST 1 - RUN RANDOM ABLATION MODEL AND RE-TRAINING
------------------------------

This script compares the fibration collapse to a random null model of ablation.
This script applies a random ablation to the original default model trained at epoch 'epoch_idx'. That is, it takes the default model trained 
at epoch 'epoch_idx' and ablates the model at random to the size of the collapse model, while keeping the weights of the trained default model.

The script calculates the dimensions of the collapsed model based on fibrations (using the thresholds defined) and then 
randomly ablates to the size of the collapsed model.

The calculation of the ablation model's dimensions is automatic, and the script returns the updated model.
This code reproduces the results explained in STEP 5 and shown in the file './validation.pdf' [pdf file](validation.pdf)

Use the script 

    bash ./kcore_ablation.sh

and set up the same parameters from STEP 2:

- exp_name (str).                            Name of the model to collapse
- epoch_idx (int).                           Epoch of training when you want to collapse
- linear_thr, cnn_thr, 
 lstm_thr, critic_thr (floats 0-2)          Threshold for the fibration calculations
                                            (default = 0.7, 0.5, 1.0, 0.8)

The output will be saved as a Metta experiment with the name 'ablation_name':

    exp_name_epoch_epoch_idx_ablationsrandomlinear_thr_cnn_thr_lstm_thr_critic_thr

More information about this script can be found inside the functions/scripts.

STEP 4: TEST 2 - RUN SMALL RANDOM MODEL FROM SCRATCH
------------------------------

This script compares the fibration collapse to a small random null model trained from scratch.
This script runs the ablation model from scratch by randomly initializing all the weights and retraining. It differs from STEP 3 in that the size remains the same, but all weights are now randomized, and training proceeds from scratch.

You have to define a new agent in the  './agent/src/metta/agent/component_policies/'; e.g, the file 'fast_small.py' contains the dimensions of the layers of the small random model. The dimensions should be obtained from the ablation model in STEP 3.

Then, you can run the training using the commands of Metta:

    ./tools/run.py experiments.recipes.arena_basic_easy_shaped.train --args run='ablation_scratch_name' --overrides policy_architecture.name='fast_small' trainer.evaluation.evaluate_interval=0 trainer.checkpoint.checkpoint_interval=10 trainer.checkpoint.wandb_checkpoint_interval=10 trainer.evaluation.replay_dir=‘exp_name’ trainer.evaluation.skip_git_check=true

This code reproduces the results explained in STEP 5 and shown in the file './validation.pdf' [pdf file](validation.pdf)

STEP 5: VALIDATION RESULTS (VISUALIZE AND COMPARE RESULTS BETWEEN ORIGINAL MODEL, COLLAPSED MODEL, ABLATION, AND SMALL MODELS)
------------------------------

Steps 1 to 4  should reproduce the results in the file './validation.pdf'.

Step 1. The green curve represents the original model (default) trained up to 15G agent steps (50k training time).

Step 2. The red curve represents the collapsed model via fibration, resulting in a 19% reduction (81% reduction in size) from the original model. 

We then perform the two randomization protocols. 

3. Orange curve: We ablate 81% nodes at random, and keep the weights of the original 
model trained at 15G. We keep training. 

4. Purple curve: We train from scratch the small model, 19% of the original size, by initially randomizing all weights.


Use the interface in WANDB to compare metrics and results. 


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
    │
    ├── ablation_scracth_name/      # Ablation model from scratch (4th step)
    │   ├── config.json
    │   └── ablation_scracth_name/
    │       └── checkpoints/
    │           └── ...
    └── ...

DISCLAIMER
----------
- Choose meaningful thresholds.
- Always verify WANDB and fiber output to validate results.
