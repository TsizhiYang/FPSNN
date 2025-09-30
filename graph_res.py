import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from scipy.interpolate import griddata

config = {
    "mathtext.fontset": 'stix',
}

plt.rcParams.update(config)

plt.rcParams['font.sans-serif'] = "Times New Roman"
# plt.rcParams['figure.figsize']=(5, 3.5)

color_dict = {
    "reference": "dimgrey",
    "FPSNN": "red",
    "FPANN": "blue",
    "APSNN": "green",
    "DDSNN": "orange",
}
alpha = 0.8

label_dict_dict = {
    "ann": {
        "fdm": "FPANN",
        "std": "APANN",
        "dl": "DDANN",
    },
    "snn": {
        "fdm": "FPSNN",
        "std": "APSNN",
        "dl": "DDSNN",
    },
}

equation_type = "wave"          # "heat", "wave"

network_type = "snn"            # "snn", "ann"
task_type = "fdm"               # "fdm", "std", "dl"

label = label_dict_dict[network_type][task_type]

initialization = "xavier"       # "undef", "xavier"
if equation_type == "heat":
    trials = 10
    epoches = 200000
    record_epoch = 1000
    show_epoch = 10000
    graph_epoch = 50000
    learning_rate = 1e-4
    phys_weight = 1
    cond_weight = 500000
elif equation_type == "wave":
    trials = 10
    epoches = 250000
    record_epoch = 1000
    show_epoch = 10000
    graph_epoch = 50000
    learning_rate = 1e-3
    phys_weight = 10
    cond_weight = 10000

# show_zero_epoch_graph = True
t_axis_lines = 40

if network_type == "snn" and task_type == "fdm":
    # label = "SNN FDM-PINN"
    name = "SNN_FDM"
if network_type == "snn" and task_type == "std":
    # label = "SNN PINN"
    name = "SNN_STD"
if network_type == "snn" and task_type == "dl":
    # label = "SNN DD"
    name = "SNN_DL"
if network_type == "ann" and task_type == "fdm":
    # label = "ANN FDM-PINN"
    name = "ANN_FDM"
if network_type == "ann" and task_type == "std":
    # label = "ANN PINN"
    name = "ANN_STD"
if network_type == "ann" and task_type == "dl":
    # label = "ANN DD"
    name = "ANN_DL"


if network_type == "snn":
    from packs.networks.spiking_network import *
elif network_type == "ann":
    from packs.networks.artificial_network import *

if equation_type == "heat":
    from packs.reference_generation.heat_ref import *
elif equation_type == "wave":
    from packs.reference_generation.wave_ref import *


elev = 20
azim = 30



x = torch.linspace(0, 1, x_step + 1)
t = torch.linspace(0, time, t_step + 1)
x_ = torch.stack([x for _ in range(t_step + 1)]).unsqueeze(-1)
t_ = torch.stack([t for _ in range(x_step + 1)]).unsqueeze(-1)
input_mesh = torch.cat((t_.permute(1, 0, 2), x_), dim=2)  # format: input_mesh[i, j] = (t_{i+1}, x_{j+1})
shape = input_mesh.shape
input = input_mesh.reshape(-1, 2).to(device)

x_ticks = [0.0, 0.5, 1.0]
t_ticks = [0.0 * time, 0.5 * time, 1.0 * time]
if equation_type == "heat":
    z_ticks = [0, 0.4, 0.8]
elif equation_type == "wave":
    z_ticks = [-1.0, 0, 1.0]

if network_type == "snn":
    model = SNN(time_window=x_step,
                in_scale=torch.tensor([[0., time], [0., 1.]]),
                in_shape_as=input[0],
                init_method=initialization).to(device)
elif network_type == "ann":
    model = ANN(init_method=initialization).to(device)

model.to(device)
model.load_state_dict(torch.load("model_params/" + equation_type + "/" + name + "_0.pth", map_location=device))

pred = model(input)
pred_mesh = pred.reshape(shape[:2]).detach().cpu()
pred_mesh_flatten = pred_mesh.flatten()

fig = plt.figure(1, figsize=(10, 3))

X, T =  np.meshgrid(x, t)
x_flatten = X.flatten()
t_flatten = T.flatten()
output_mesh_flatten = output_mesh.flatten()
xi = torch.linspace(0, 1, 100)
ti = torch.linspace(0, time, 40)
Xi, Ti = np.meshgrid(xi, ti)

t_show = int(t_step / t_axis_lines)

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.set_title("(a) Reference solution", pad=0)
Zi = griddata((t_flatten, x_flatten), output_mesh_flatten, (Ti, Xi), method='linear')
if equation_type == "heat":
    pass
elif equation_type == "wave":
    ax1.plot3D(np.zeros_like(ti), ti, Zi[:, 0], color='blue', linewidth=2, linestyle='-')
    ax1.plot3D(np.ones_like(ti), ti, Zi[:, -1], color='blue', linewidth=2, linestyle='-')
ax1.plot3D(xi, np.zeros_like(xi), Zi[0, :], color='red', linewidth=2)
ax1.plot_surface(Xi, Ti, Zi, rstride=1, cstride=1, cmap='coolwarm', alpha=0.4)
ax1.set_xlabel("$x$", labelpad=0)
ax1.set_ylabel("$t$", labelpad=0)
ax1.set_zlabel("$u$", labelpad=0)

    
# t_show = int(t_step / t_axis_lines)
# for i in range(output_mesh.shape[0]):
#     if i % t_show == 0:
#         ax1.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], output_mesh[i], c=color_dict["reference"], alpha=alpha)
#         plt.xlabel("t")
#         plt.ylabel("x")
#         ax1.set_zlabel("$u$")
ax1.set_xlim([0, 1])
ax1.set_ylim([0, time])
if equation_type == "heat":
    ax1.set_zlim([0, 0.8])
elif equation_type == "wave":
    ax1.set_zlim([-1.2, 1.2])
ax1.tick_params(axis='x', which='major', pad=0)
ax1.tick_params(axis='y', which='major', pad=0)
ax1.tick_params(axis='z', which='major', pad=0)

ax1.view_init(elev=elev, azim=azim)

ax1.xaxis.set_ticks(x_ticks)
ax1.yaxis.set_ticks(t_ticks)
ax1.zaxis.set_ticks(z_ticks)

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.set_title("(b) Model solution", pad=0)
Zi = griddata((t_flatten, x_flatten), pred_mesh_flatten, (Ti, Xi), method='linear')
ax2.plot_surface(Xi, Ti, Zi, rstride=1, cstride=1, cmap='coolwarm', alpha=0.4)
# ax2.plot3D(xi, np.zeros_like(xi), Zi[0, :], color='red', linewidth=2, zorder=5)
ax2.set_xlabel("$x$", labelpad=0)
ax2.set_ylabel("$t$", labelpad=0)
ax2.set_zlabel("$\phi$", labelpad=0)
# if equation_type == "heat":
#     pass
# elif equation_type == "wave":
#     ax2.plot3D(np.zeros_like(ti), ti, Zi[:, 0], color='blue', linewidth=2, linestyle='-', zorder=4)
#     ax2.plot3D(np.ones_like(ti), ti, Zi[:, -1], color='blue', linewidth=2, linestyle='-', zorder=4)
# for i in range(output_mesh.shape[0]):
#     if i % t_show == 0:
#         ax2.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], pred_mesh[i], c=color_dict[label], alpha=alpha)
#         plt.xlabel("t")
#         plt.ylabel("x")
#         ax2.set_zlabel("$\phi$")
# if equation_type == "heat":
#     ax2.set_xlim([0, 0.01])
#     ax2.set_ylim([0, 1])
#     ax2.set_zlim([0, 0.6])
# elif equation_type == "wave":
#     ax2.set_xlim([0, 1])
#     ax2.set_ylim([0, 1])
#     ax2.set_zlim([-1, 1])
ax2.set_xlim([0, 1])
ax2.set_ylim([0, time])
if equation_type == "heat":
    ax2.set_zlim([0, 0.8])
elif equation_type == "wave":
    ax2.set_zlim([-1.2, 1.2])
ax2.tick_params(axis='x', which='major', pad=0)
ax2.tick_params(axis='y', which='major', pad=0)
ax2.tick_params(axis='z', which='major', pad=0)

ax2.view_init(elev=elev, azim=azim)

ax2.xaxis.set_ticks(x_ticks)
ax2.yaxis.set_ticks(t_ticks)
ax2.zaxis.set_ticks(z_ticks)

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.set_title("(c) Error (log scale)", pad=0)
# ax3.scatter(X, T, torch.log(abs(pred_mesh - output_mesh)), s=.5, alpha=alpha)
for i in range(output_mesh.shape[0]):
    if i % t_show == 0:
        ax3.scatter(X[i], T[i], torch.log(abs(pred_mesh[i] - output_mesh[i])), s=.4, c=color_dict[label], alpha=alpha)
ax3.set_xlabel("$x$", labelpad=0)
ax3.set_ylabel("$t$", labelpad=0)
ax3.set_zlabel("$\epsilon ~ (\log_{10})$", labelpad=0)
# if equation_type == "heat":
#     ax3.set_xlim([0, 0.01])
#     ax3.set_ylim([0, 1])
#     # ax3.set_zlim([-0.1, 0.1])
# elif equation_type == "wave":
#     ax3.set_xlim([0, 1])
#     ax3.set_ylim([0, 1])
#     # ax3.set_zlim([-0.1, 0.1])
ax3.set_xlim([0, 1])
ax3.set_ylim([0, time])
ax3.tick_params(axis='x', which='major', pad=0)
ax3.tick_params(axis='y', which='major', pad=0)
ax3.tick_params(axis='z', which='major', pad=0)

ax3.view_init(elev=elev, azim=azim)

ax3.xaxis.set_ticks(x_ticks)
ax3.yaxis.set_ticks(t_ticks)
# ax1.zaxis.set_ticks(z_ticks)

# ax4 = fig.add_subplot(4, 1, 4, projection='3d')
# for i in range(output_mesh.shape[0]):
#     if i % t_show == 0:
#         ax4.scatter(input_mesh[i, :, 0], input_mesh[i, :, 1], pred_mesh[i] - output_mesh[i], s=.6, c=color_dict[label], alpha=alpha)
#         plt.xlabel("t")
#         plt.ylabel("x")
#         ax4.set_zlabel("$\epsilon$")
# if equation_type == "heat":
#     ax4.set_xlim([0, 0.01])
#     ax4.set_ylim([0, 1])
#     # ax4.set_zlim([-0.1, 0.1])
# elif equation_type == "wave":
#     ax4.set_xlim([0, 1])
#     ax4.set_ylim([0, 1])
#     # ax4.set_zlim([-0.1, 0.1])

# ax4.xaxis.set_ticks(t_ticks)
# ax4.yaxis.set_ticks(x_ticks)
# ax4.zaxis.set_ticks(z_ticks)

# ax = fig.add_subplot(1, 3, 2, projection='3d')
# for i in range(output_mesh.shape[0]):
#     if i % t_show == 0:
#         ax.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], output_mesh[i], c=color["reference"], alpha=alpha)
#         ax.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], pred_mesh[i], c=color[label], alpha=alpha)
#         plt.xlabel("t")
#         plt.ylabel("x")
# if equation_type == "heat":
#     ax.set_xlim([0, 0.01])
#     ax.set_ylim([0, 1])
#     ax.set_zlim([0, 0.6])
# elif equation_type == "wave":
#     ax.set_xlim([0, 1])
#     ax.set_ylim([0, 1])
#     ax.set_zlim([-1, 1])

# plt.suptitle(label + f", epoch {epoch}")

folder = os.getcwd()
figname = folder + '/data/' + equation_type + '/' + name + "_res" + ".svg"

plt.tight_layout(pad=1.5, w_pad=2, h_pad=0)
plt.savefig(figname, dpi=300, format="svg")
print("image saved as", figname)

plt.show()
