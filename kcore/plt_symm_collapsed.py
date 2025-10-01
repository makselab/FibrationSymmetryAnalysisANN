import matplotlib.pyplot as plt
import torch

data_1 = torch.load('/home/ali/Codes/metta/symm_vs_time_steps_final.pth')
data_2 = torch.load('/home/ali/Codes/metta/symm_vs_time_steps_final_collapsed.pth')

fig, axs = plt.subplots()
axs.plot(data_1['steps'], data_1['reduction'])
axs.plot(data_2['steps'], [data_1['reduction'][-1]*p for p in data_2['reduction']])

fig.savefig('symm_together.svg',format='svg')