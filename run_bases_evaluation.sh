#!/bin/bash

# # Original
# command="./tools/run.py experiments.recipes.arena_basic_easy_shaped.evaluate --args policy_uri=/home/ali/Codes/metta/train_dir/osva_sept_8_final/osva_sept_8_final/checkpoints/osva_sept_8_final__e3160__s3094272000__t9825__sc0.pt --overrides replay_dir=/home/ali/Codes/metta/train_dir/replay_testt/"

# eval $command
# sleep 1


command="./tools/run.py experiments.recipes.arena_basic_easy_shaped.evaluate --args policy_uri=/home/ali/Codes/metta/train_dir/osva_sept_8_final_base_linear_0.60_cnn_0.00_lstm_0.00/osva_sept_8_final_base_linear_0.60_cnn_0.00_lstm_0.00/checkpoints/osva_sept_8_final_base_linear_0.60_cnn_0.00_lstm_0.00__e3160__s3094272000__t9825__sc0.pt --overrides replay_dir=/home/ali/Codes/metta/train_dir/replay_testt_/"

eval $command


# for thr_lstm in 0.0; do
#     for thr_cnn in 0.5 ; do
#         for thr_l in 1.0; do
#             thr_l=$(printf "%.2f" $thr_l)
#             thr_cnn=$(printf "%.2f" $thr_cnn)
#             thr_lstm=$(printf "%.2f" $thr_lstm)

#             name_base="arena_1_base_linear_${thr_l}_cnn_${thr_cnn}_lstm_${thr_lstm}"
#             command="./tools/run.py experiments.recipes.arena_basic_easy_shaped.evaluate --args policy_uri=/home/ali/Codes/metta/train_dir/osva_sept_8_final_base_linear_1.00_cnn_0.50_lstm_0.00/osva_sept_8_final_base_linear_1.00_cnn_0.50_lstm_0.00/checkpoints/osva_sept_8_final_base_linear_1.00_cnn_0.50_lstm_0.00__e3160__s3094272000__t9825__sc0.pt --overrides replay_dir=/home/ali/Codes/metta/train_dir/replay_testt_/"

#             eval $command
#             sleep 1
#         done
#     done
# done



#         SELECT e.replay_url
#         FROM episodes e
#         JOIN simulations s ON e.simulation_id = s.id
#         WHERE e.replay_url IS NOT NULL
#          AND s.policy_key = ? AND s.policy_version = ? AND s.env = ?
# ['osva_sept_8_final', 0, 3160]


#    SELECT e.replay_url
#         FROM episodes e
#         JOIN simulations s ON e.simulation_id = s.id
#         WHERE e.replay_url IS NOT NULL
#          AND s.policy_key = ? AND s.policy_version = ? AND s.env = ?
# ['model_2400', 0, 0]
