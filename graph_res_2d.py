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

equation_type = "burgers2d"

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
elif equation_type == "burgers2d":
    trials = 10
    epoches = 250000
    record_epoch = 1000
    show_epoch = 10000
    graph_epoch = 10000
    learning_rate = 2e-3
    phys_weight = 100
    cond_weight = 5e7

# show_zero_epoch_graph = True
t_axis_lines = 40

if network_type == "snn" and task_type == "fdm":
    # label = "SNN FDM-PINN"
    name = "SNN_FDM"


from packs.networks.spiking_network_2d import *
from packs.reference_generation.burgers2d_ref import *


elev = 20
azim = -55



t = torch.linspace(0, time, t_step + 1)
x = torch.linspace(0, 1, x_step + 1)
y = torch.linspace(0, 1, y_step + 1)
in_scale = torch.tensor([[0., time], [0., 1.], [0., 1.]])

t, x, y = torch.meshgrid(t, x, y)
input_mesh = torch.stack((t, x, y), dim=-1)

shape = input_mesh.shape
input = input_mesh.reshape(-1, 3).to(device)

x_ticks = [0.0, 0.5, 1.0]
y_ticks = [0.0, 0.5, 1.0]
z_ticks = [-1.0, 0, 1.0]

model = SNN(time_window=x_step,
            in_scale=torch.tensor([[0., time], [0., 1.], [0., 1.]]),
            in_shape_as=input[0],
            init_method=initialization).to(device)

model.to(device)
model.load_state_dict(torch.load("model_params/" + equation_type + "/" + name + "_0.pth", map_location=device))

pred = model(input)
pred_mesh = pred.reshape(shape[:3]).detach().cpu()

fig = plt.figure(1, figsize=(10, 3))


x = torch.linspace(0, 1, x_step + 1)
y = torch.linspace(0, 1, y_step + 1)
X, Y =  np.meshgrid(x, y)
x_flatten = X.flatten()
y_flatten = Y.flatten()
xi = torch.linspace(0, 1, 100)
yi = torch.linspace(0, 1, 100)
Xi, Yi = np.meshgrid(xi, yi)

t_show = int(t_step / t_axis_lines)

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.set_title("(a) Initial condition", pad=0)
Zi = griddata((x_flatten, y_flatten), output_mesh[0, :, :].flatten(), (Xi, Yi), method='linear')
ax1.plot_surface(Xi, Yi, Zi, rstride=1, cstride=1, cmap='coolwarm', alpha=0.4)
ax1.set_xlabel("$x$", labelpad=0)
ax1.set_ylabel("$y$", labelpad=0)
ax1.set_zlabel("$u|_{t=0}$", labelpad=0)

ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_zlim([-1.2, 1.2])
ax1.tick_params(axis='x', which='major', pad=0)
ax1.tick_params(axis='y', which='major', pad=0)
ax1.tick_params(axis='z', which='major', pad=0)

ax1.view_init(elev=elev, azim=azim)

ax1.xaxis.set_ticks(x_ticks)
ax1.yaxis.set_ticks(y_ticks)
ax1.zaxis.set_ticks(z_ticks)

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.set_title("(b) Reference temporal difference", pad=0)
Zi = griddata((x_flatten, y_flatten), (output_mesh[-1, :, :] - output_mesh[0, :, :]).flatten(), (Xi, Yi), method='linear')
ax2.plot_surface(Xi, Yi, Zi, rstride=1, cstride=1, cmap='coolwarm', alpha=0.4)
ax2.set_xlabel("$x$", labelpad=0)
ax2.set_ylabel("$y$", labelpad=0)
ax2.set_zlabel("$u|_{t=T} - u|_{t=0}$", labelpad=2)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_zlim([-0.01, 0.01])
ax2.tick_params(axis='x', which='major', pad=0)
ax2.tick_params(axis='y', which='major', pad=0)
ax2.tick_params(axis='z', which='major', pad=2)

ax2.view_init(elev=elev, azim=azim)

ax2.xaxis.set_ticks(x_ticks)
ax2.yaxis.set_ticks(y_ticks)
# ax2.zaxis.set_ticks(z_ticks)

print((output_mesh[-1, :, :] - output_mesh[0, :, :]).max())     # 0.0075

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.set_title("(c) Model temporal difference", pad=0)
# ax3.scatter(X, T, torch.log(abs(pred_mesh - output_mesh)), s=.5, alpha=alpha)
Zi = griddata((x_flatten, y_flatten), (pred_mesh[-1, :, :] - pred_mesh[0, :, :]).flatten(), (Xi, Yi), method='linear')
ax3.plot_surface(Xi, Yi, Zi, rstride=1, cstride=1, cmap='coolwarm', alpha=0.4)
ax3.set_xlabel("$x$", labelpad=0)
ax3.set_ylabel("$y$", labelpad=0)
ax3.set_zlabel("$\phi|_{t=T} - \phi|_{t=0}$", labelpad=2)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])
ax3.set_zlim([-0.01, 0.01])
ax3.tick_params(axis='x', which='major', pad=0)
ax3.tick_params(axis='y', which='major', pad=0)
ax3.tick_params(axis='z', which='major', pad=2)

ax3.view_init(elev=elev, azim=azim)

ax3.xaxis.set_ticks(x_ticks)
ax3.yaxis.set_ticks(y_ticks)
# ax3.zaxis.set_ticks(z_ticks)

folder = os.getcwd()
figname = folder + '/data/' + equation_type + '/' + name + "_res" + ".svg"

plt.tight_layout(pad=1.5, w_pad=2, h_pad=0)
plt.savefig(figname, dpi=300, format="svg")
print("image saved as", figname)

plt.show()
