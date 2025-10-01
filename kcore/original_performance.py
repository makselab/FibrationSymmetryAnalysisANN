#!/usr/bin/env -S uv run

import duckdb
import pandas as pd
import os
import torch
import re
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------

PATH = './train_dir/'
folder = 'original_stats/'
datapath = PATH + folder

# datapath = './train_dir/stats_1.0/'
resPATH = './results_kcore/'
duckdbfiles = [file for file in os.listdir(datapath) if file.endswith(".duckdb")]
metric_name = 'reward' #'battery_red.get' # metric_name = 'ore_red.put'
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

  # ===============================

  metrics = dfs['agent_metrics'].query(f"metric == '{metric_name}'").drop(columns='metric').reset_index(drop=True)
  metrics = metrics['value'].mean()

  # print(metrics)

  # exit()
  # metrics = metrics.groupby('agent_id')['value'].mean().reset_index(drop=True)

  results.append(metrics)

  # results.append(dfs['agent_metrics'])

# results = pd.concat(results, ignore_index=True)

# exit()
fig, axs = plt.subplots(1,1)
axs.plot(results, '*', alpha=0.7)
axs.set_xlabel('Idx')
axs.set_ylabel('Metric')
# plt.gsrid(True, alpha=0.3)
fig.savefig('org_v1.svg',format='svg')

# stats = results.groupby('metric')['value'].agg([
#     ('MIN', 'min'),
#     ('MAX', 'max'), 
#     ('MEAN', 'mean'),
#     ('STD', 'std')
# ]).reset_index()

# stats['CV'] = stats['STD'] / stats['MEAN']
# stats = stats.sort_values('CV', ascending=False).reset_index(drop=True)

# print(stats)






# ==========================================


  # metrics = dfs['agent_metrics'].query(f"metric == '{metric_name}'").drop(columns='metric').reset_index(drop=True)
  # av_metrics = metrics['value'].mean()

  # base_name = dfs['agent_policies']['policy_key'][0]

  # # print(thr_l,thr_cnn,thr_lstm, av_metrics)

  # results.append((thr_l,thr_cnn,thr_lstm, av_metrics))

# results = sorted(results)

# -------------------------------------------------------------------------------


# conn = duckdb.connect('/home/user/metta/evaluation_model_35700.duckdb')

# tables = conn.execute("SHOW TABLES").fetchall()

# dfs = {} 

# for tab in tables:
#   name_tab = tab[0]
#   data = conn.execute("SELECT * FROM " + name_tab).df()
#   dfs[name_tab] = data


# metrics = dfs['agent_metrics'].query(f"metric == '{metric_name}'").drop(columns='metric').reset_index(drop=True)
# av_metrics = metrics['value'].mean()

# print(av_metrics)



  # results.append((thr_l,thr_cnn,thr_lstm, metrics))

# metric_name = 'reward' #'action.attack_nearest.failed.rate' #'action.move.success.activity_rate'
# # name_sim = 'arena/basic'
# time_model = 37500


# exit()
# results = {'thr':[], metric_name:[]}




#!/usr/bin/env -S uv run



# -------------------------------------------------------------------------------

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

