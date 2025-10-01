#!/usr/bin/env -S uv run

import duckdb
import pandas as pd
import os
import torch
import re
# -------------------------------------------------------------------------------

PATH = './train_dir/'
folder = 'bases_stats/'
datapath = PATH + folder
resPATH = './results_kcore/'
duckdbfiles = [file for file in os.listdir(datapath) if file.endswith(".duckdb")]
# metric_name = 'reward'
metric_name = 'battery_red.get'
# metric_name = 'ore_red.put'
results = []
# -------------------------------------------------------------------------------
for ff in duckdbfiles:
  conn = duckdb.connect(datapath + ff)

  tables = conn.execute("SHOW TABLES").fetchall()

  dfs = {} 

  for tab in tables:
    name_tab = tab[0]
    data = conn.execute("SELECT * FROM " + name_tab).df()
    dfs[name_tab] = data


  metrics = dfs['agent_metrics'].query(f"metric == '{metric_name}'").drop(columns='metric').reset_index(drop=True)
  av_metrics = metrics['value'].mean()

  base_name = dfs['agent_policies']['policy_key'][0]
  pattern = r'linear_(\d+\.?\d*)_cnn_(\d+\.?\d*)_lstm_(\d+\.?\d*)'
  match = re.search(pattern, base_name)
    
  if match:
    thr_l = float(match.group(1))
    thr_cnn = float(match.group(2))
    thr_lstm = float(match.group(3))

  # print(thr_l,thr_cnn,thr_lstm, av_metrics)

  results.append((thr_l,thr_cnn,thr_lstm, av_metrics))

results = sorted(results)

for xx in results:
    print(xx)

# metric_name = 'reward' #'action.attack_nearest.failed.rate' #'action.move.success.activity_rate'
# # name_sim = 'arena/basic'
# time_model = 37500


# exit()
# results = {'thr':[], metric_name:[]}



    # stats_dir = os.path.join(base_dir, 'base_1_' + f"linear_{thr_l.item():.2f}_cnn_{thr_cnn.item():.2f}_lstm_{thr_lstm.item():.2f}", "stats")

#             for ff in os.listdir(stats_dir):
#                 if ff.startswith("all") and ff.endswith(".duckdb"): filename = os.path.join(stats_dir, ff)

#             conn = duckdb.connect(filename)
#             tables = conn.execute("SHOW TABLES").fetchall()

#             dfs = {} 

#             for tab in tables:
#               name_tab = tab[0]
#               data = conn.execute("SELECT * FROM " + name_tab).df()
#               dfs[name_tab] = data


#             # print(dfs['agent_metrics']['metric'].unique())

#             # exit()

#             reward_sim = dfs['agent_metrics'].query(f"metric == '{metric_name}'") # check metric
#             reward_sim = reward_sim.drop(columns=['metric','agent_id']).reset_index(drop=True)

#             eps_sim = dfs['episodes'].drop(columns=['created_at', 'completed_at','replay_url'])
#             sim = dfs['simulations'].drop(columns=['suite', 'env', 'policy_key', 'policy_version','created_at', 'finished_at'])

#             partial_eps = pd.merge(reward_sim, eps_sim,
#                 left_on='episode_id', right_on='id', how='left').drop(columns=['id']).reset_index(drop=True)

#             res = pd.merge(partial_eps, sim,
#                 left_on='simulation_id', right_on='id', how='left').drop(columns=['id']).reset_index(drop=True)

#             avg_by_name = res.groupby('name')['value'].mean().reset_index()

#             print(thr_l,thr_cnn,thr_lstm)
#             print(avg_by_name)

#             # print('==============')

#             value = avg_by_name.loc[avg_by_name['name'] == name_sim, 'value'].values[0]

#             results['thr'].append((thr_l.item(), thr_cnn.item(), thr_lstm.item()))
#             results[metric_name].append(value)

#             print(thr_l,thr_cnn,thr_lstm, value)

# torch.save(results, '/home/user/metta/results_coloring/original_1_full_collapse_performance_' + f'{time_model:04d}' + '.pth')

