import matplotlib.pyplot as plt
import torch

import numpy as np
from collections import defaultdict
from itertools import product

reduction = torch.load('/home/user/metta/results_kcore/arena_1/_reduction_35700.pth')
# performance = torch.load('/home/user/metta/results_coloring/original_1_full_collapse_performance_51540.pth')
# metric_name = 'ore_red.gained.avg'

x = []
y = []
col = []

colors = {0.0:'red', 0.5:'blue', 1.0:'green', 2.0:'black'}

# =================================

# for ii, thr in enumerate(reduction['thr']):
#     # print(thr, reduction['red'][ii], performance['ore_red.gained.avg'][ii])

#     thr_linear = reduction['thr'][ii][0]
#     thr_cnn = reduction['thr'][ii][1]
#     thr_lstm = reduction['thr'][ii][2]

#     # if thr_linear != thr_cnn or thr_linear != thr_lstm or thr_cnn != thr_lstm: continue

#     x.append(reduction['red'][ii])
#     y.append(performance['ore_red.gained.avg'][ii])
#     col.append(colors[thr_cnn])


# fig_5, axs_5 = plt.subplots(1,1)
# for i in range(len(x)):
#     axs_5.plot(x[i],y[i], '*', c=col[i])
# fig_5.savefig('perf_vs_red.svg',format='svg')
# =================================

fig, axs = plt.subplots(1,2)

puntos_3d = reduction['thr']
reduc = reduction['red']
# metric = performance['ore_red.gained.avg']

puntos_array = np.array(puntos_3d)
x = puntos_array[:, 0]  # Valores x
y = puntos_array[:, 1]  # Valores y  
z = puntos_array[:, 2]  # Valores z

reduc = np.array(reduc)  # Valores B
# metric = np.array(metric)  # Valores B

y_values = np.unique(y)
z_values = np.unique(z)

# Colormap para distinguir las curvas
colors = plt.cm.jet(np.linspace(0, 1, len(y_values)))
line_style = ['solid', 'dotted', 'dashed', 'dashdot']

color_index = 0
legend_handles_1 = []
legend_handles_2 = []

# Para cada combinación de y y z
for y_val in y_values:
    line_index = 0
    for z_val in z_values:
        # Filtrar datos para esta combinación
        mask = (y == y_val) & (z == z_val)
        
        x_filtered = x[mask]
        reduc_filtered = reduc[mask]
        # perf_filtered = metric[mask]
        
        # Ordenar por x para tener una curva suave
        sort_idx = np.argsort(x_filtered)
        x_sorted = x_filtered[sort_idx]
        reduc_sorted = reduc_filtered[sort_idx]
        # metric_sorted = perf_filtered[sort_idx]

        
        # Graficar la curva
        line, = axs[0].plot(x_sorted, reduc_sorted, 
                        color=colors[color_index],
                        ls =line_style[line_index], 
                        markersize=6,
                        linewidth=2,
                        label=f'y={y_val}, z={z_val}')

        legend_handles_1.append(line)

        # line, = axs[1].plot(x_sorted, metric_sorted,ls =line_style[line_index],
        #                 color=colors[color_index],
        #                 markersize=6,
        #                 linewidth=2,
        #                 label=f'y={y_val}, z={z_val}')
        
        
        # legend_handles_2.append(line)

        line_index += 1
    
    color_index += 1

# Personalizar el gráfico
axs[0].set_xlabel('Thr Linear', fontsize=12)
axs[0].set_ylabel('Reduction', fontsize=12)
axs[1].set_xlabel('Thr Linear', fontsize=12)
axs[1].set_ylabel('Metric', fontsize=12)

# plt.grid(True, alpha=0.3)

# Leyenda fuera del gráfico para mejor visualización
axs[0].legend(handles=legend_handles_1, 
           title='CNN,LSTM',
           bbox_to_anchor=(1.05, 1), 
           loc='upper left',
           fontsize=10)
axs[1].legend(handles=legend_handles_1, 
           title='CNN,LSTM',
           bbox_to_anchor=(1.05, 1), 
           loc='upper left',
           fontsize=10)

# plt.tight_layout()
# plt.show()
fig.savefig('prueba2.svg',format='svg')

# print(reduction)
# print(performance)

# # color = 'tab:red'
# # ax1.set_xlabel('Threshold Linear')
# # ax1.set_ylabel('Metric', color=color)
# # ax1.plot(performance['thr'], performance[metric_name], color=color)
# # ax1.tick_params(axis='y', labelcolor=color)

# # ax2 = ax1.twinx()
# # color = 'tab:blue'
# # ax2.set_ylabel('Reduction', color=color)  # we already handled the x-label with ax1
# # ax2.plot(reduction['thr'], reduction['red'], color=color)
# # ax2.tick_params(axis='y', labelcolor=color)

# # fig.tight_layout()  # otherwise the right y-label is slightly clipped



# # Crear el gráfico principal
# plt.figure(figsize=(14, 9))

# # Colormap para distinguir las curvas
# colors = plt.cm.tab20(np.linspace(0, 1, len(y_unicos) * len(z_unicos)))

# # Organizar datos por combinación (y,z)
# combinaciones = defaultdict(list)

# for i in range(len(puntos_3d)):
#     punto = puntos_3d[i]
#     b_val = valores_B[i]
#     clave = (punto[1], punto[2])  # clave = (y, z)
#     combinaciones[clave].append((punto[0], b_val))  # (x, B)

# # Graficar cada combinación
# legend_handles = []
# color_idx = 0

# for (y_val, z_val), datos in combinaciones.items():
#     # Separar x y B para esta combinación
#     x_vals = [d[0] for d in datos]
#     b_vals = [d[1] for d in datos]
    
#     # Ordenar por x para curva suave
#     sort_idx = np.argsort(x_vals)
#     x_sorted = np.array(x_vals)[sort_idx]
#     b_sorted = np.array(b_vals)[sort_idx]
    
#     # Graficar la curva
#     line, = plt.plot(x_sorted, b_sorted, 
#                     'o-', 
#                     color=colors[color_idx],
#                     markersize=6,
#                     linewidth=2,
#                     markeredgecolor='black',
#                     markeredgewidth=0.5,
#                     label=f'y={y_val}, z={z_val}')
    
#     legend_handles.append(line)
#     color_idx += 1

# # Personalizar el gráfico
# plt.xlabel('Valor X', fontsize=12, fontweight='bold')
# plt.ylabel('Valor B', fontsize=12, fontweight='bold')
# plt.title('Valores B vs X para diferentes combinaciones de Y y Z', 
#           fontsize=14, fontweight='bold')
# plt.grid(True, alpha=0.3)

# # Leyenda organizada
# plt.legend(handles=legend_handles, 
#            title='Combinaciones (Y, Z)',
#            bbox_to_anchor=(1.05, 1), 
#            loc='upper left',
#            fontsize=9,
#            ncol=2 if len(legend_handles) > 10 else 1)

# plt.tight_layout()
# plt.show()
