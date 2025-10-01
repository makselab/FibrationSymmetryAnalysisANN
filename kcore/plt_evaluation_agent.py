#!/usr/bin/env -S uv run
import duckdb
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# -------------------------------------------------------------------------------

PATH = './train_dir/'
resPATH = './results_kcore/'
exp_name = 'arena_1/'
folder = 'training_eval/'

pathdata = os.path.join(PATH, exp_name, folder)
resdata = os.path.join(resPATH, exp_name)

if not os.path.exists(resdata): os.makedirs(resdata)

# -------------------------------------------------------------------------------
duckdbfiles = [file for file in os.listdir(pathdata) if file.endswith(".duckdb")]
results = []

metric_name = 'reward' #'action.attack_nearest.failed.rate' #'action.move.success.activity_rate'
sim_name = 'train_task_0'

for ff in duckdbfiles:
  conn = duckdb.connect(pathdata + ff)

  tables = conn.execute("SHOW TABLES").fetchall()

  dfs = {} 

  for tab in tables:
    name_tab = tab[0]
    data = conn.execute("SELECT * FROM " + name_tab).df()
    dfs[name_tab] = data

  # ===============================
  metrics = dfs['agent_metrics'].query(f"metric == '{metric_name}'").drop(columns='metric').reset_index(drop=True)
  metrics = metrics['value'].mean()

  print(metrics)

  exit()

  epoch = dfs['agent_policies']['policy_version'][0]


  results.append((epoch, metrics))

  conn.close()


results = sorted(results)

for xx in results:
  print(xx)

exit()
# temporal_av_metric = sum(y for x, y in results) / len(results)
# print('Time Avg', temporal_av_metric)
# ------------------------------------------------------

fig, axs = plt.subplots(1,1)
time = [a for a, _ in results]
curve = [b for _, b in results]

# y_smooth = savgol_filter(curve, window_length=11, polyorder=3)
y_smooth = curve

axs.plot(time, y_smooth, '*')

axs.set_xlabel('Epochs')
axs.set_ylabel('Reward')
fig.savefig(resdata + metric_name + '_training_.svg',format='svg')


# ===

''' List of metrics:
# 'reward' 'action.attack.failed' 'action.failure_penalty'
#  'action.get_items.failed' 'action.move.failed' 'action.move.success'
#  'action.noop.success' 'action.put_items.failed' 'action.rotate.success'
#  'movement.direction.down' 'movement.direction.left'
#  'movement.direction.right' 'movement.direction.up'
#  'movement.rotation.to_down' 'movement.rotation.to_left'
#  'movement.rotation.to_right' 'movement.rotation.to_up'
#  'movement.sequential_rotations' 'status.max_steps_without_motion'
#  'action.get_items.success' 'ore_red.gained' 'ore_red.get'
#  'action.put_items.success' 'armor.gained' 'armor.get' 'ore_red.lost'
#  'ore_red.put' 'action.attack.agent.friendly_fire' 'action.attack.success'
#  'laser.gained' 'laser.get' 'laser.lost' 'battery_red.gained'
#  'battery_red.get' 'battery_red.lost' 'battery_red.put'
#  'status.frozen.ticks' 'status.frozen.ticks.agent'
#  'action.attack.agent.steals.laser.from.agent' 'heart.gained' 'heart.get'
'''