import matplotlib.pyplot as plt
import torch

# data = torch.load('/home/ali/Codes/metta/symm_vs_time_steps_final.pth')
data = torch.load('/home/ali/Codes/metta/symm_vs_time_steps_final_collapsed.pth')

fig, axs = plt.subplots()
axs.plot(data['steps'], data['reduction'])

fig.savefig('fig_1_v2_coll.svg',format='svg')
# plt.show()