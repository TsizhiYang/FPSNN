import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

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
t_axis_lines = 15

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

x = torch.linspace(0, 1, x_step + 1)
t = torch.linspace(0, time, t_step + 1)
x = torch.stack([x for _ in range(t_step + 1)]).unsqueeze(-1)
t = torch.stack([t for _ in range(x_step + 1)]).unsqueeze(-1)
input_mesh = torch.cat((t.permute(1, 0, 2), x), dim=2)  # format: input_mesh[i, j] = (t_{i+1}, x_{j+1})
shape = input_mesh.shape
input = input_mesh.reshape(-1, 2).to(device)

x_ticks = [0.0, 0.5, 1.0]
t_ticks = [0.0 * time, 0.5 * time, 1.0 * time]
if equation_type == "heat":
    z_ticks = [0, 0.5, 1]
elif equation_type == "wave":
    z_ticks = [-1, 0, 1]

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

fig = plt.figure(1, figsize=(6, 14))

ax1 = fig.add_subplot(4, 1, 1, projection='3d')
t_show = int(t_step / t_axis_lines)
for i in range(output_mesh.shape[0]):
    if i % t_show == 0:
        ax1.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], output_mesh[i], c=color_dict["reference"], alpha=alpha)
        plt.xlabel("t")
        plt.ylabel("x")
        ax1.set_zlabel("$u$")
if equation_type == "heat":
    ax1.set_xlim([0, 0.01])
    ax1.set_ylim([0, 1])
    ax1.set_zlim([0, 0.6])
elif equation_type == "wave":
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_zlim([-1, 1])

ax1.xaxis.set_ticks(t_ticks)
ax1.yaxis.set_ticks(x_ticks)
ax1.zaxis.set_ticks(z_ticks)

ax2 = fig.add_subplot(4, 1, 2, projection='3d')
for i in range(output_mesh.shape[0]):
    if i % t_show == 0:
        ax2.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], pred_mesh[i], c=color_dict[label], alpha=alpha)
        plt.xlabel("t")
        plt.ylabel("x")
        ax2.set_zlabel("$\phi$")
if equation_type == "heat":
    ax2.set_xlim([0, 0.01])
    ax2.set_ylim([0, 1])
    ax2.set_zlim([0, 0.6])
elif equation_type == "wave":
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_zlim([-1, 1])

ax2.xaxis.set_ticks(t_ticks)
ax2.yaxis.set_ticks(x_ticks)
ax2.zaxis.set_ticks(z_ticks)

ax3 = fig.add_subplot(4, 1, 3, projection='3d')
for i in range(output_mesh.shape[0]):
    if i % t_show == 0:
        ax3.scatter(input_mesh[i, :, 0], input_mesh[i, :, 1], torch.log(abs(pred_mesh[i] - output_mesh[i])), s=.6, c=color_dict[label], alpha=alpha)
        plt.xlabel("t")
        plt.ylabel("x")
        ax3.set_zlabel("$\epsilon ~ (\log_{10})$")
if equation_type == "heat":
    ax3.set_xlim([0, 0.01])
    ax3.set_ylim([0, 1])
    # ax3.set_zlim([-0.1, 0.1])
elif equation_type == "wave":
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    # ax3.set_zlim([-0.1, 0.1])

ax3.xaxis.set_ticks(t_ticks)
ax3.yaxis.set_ticks(x_ticks)
# ax1.zaxis.set_ticks(z_ticks)

ax4 = fig.add_subplot(4, 1, 4, projection='3d')
for i in range(output_mesh.shape[0]):
    if i % t_show == 0:
        ax4.scatter(input_mesh[i, :, 0], input_mesh[i, :, 1], pred_mesh[i] - output_mesh[i], s=.6, c=color_dict[label], alpha=alpha)
        plt.xlabel("t")
        plt.ylabel("x")
        ax4.set_zlabel("$\epsilon$")
if equation_type == "heat":
    ax4.set_xlim([0, 0.01])
    ax4.set_ylim([0, 1])
    # ax4.set_zlim([-0.1, 0.1])
elif equation_type == "wave":
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    # ax4.set_zlim([-0.1, 0.1])

ax4.xaxis.set_ticks(t_ticks)
ax4.yaxis.set_ticks(x_ticks)
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
plt.savefig(figname, dpi=300, format="svg")
print("image saved as", figname)

plt.show()
