import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import os

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

equation_type = "heat"          # "heat", "wave"

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

spike_rate = model.spike_rate().cpu()

print("Equation type", equation_type)
print("Spike rate of hidden 16-layer", float(spike_rate))

E_mac = 3.1
E_ac = 0.1

E_snn = (
    0.5 * (2 * 16) * E_ac * 40 +
    spike_rate * (16 * 8) * E_ac * 40 +
    (8 * 1) * E_mac
)

E_ann = (
    (2 * 16) * E_mac +
    (16 * 8) * E_mac +
    (8 * 1) * E_mac
)

print("E_snn =", float(E_snn))
print("E_ann =", float(E_ann))

labels = ['ANN', 'SNN']
values = [E_ann, E_snn]
colors = [color_dict['FPANN'], color_dict['FPSNN']]
colors = [to_rgba(color, alpha=.75) for color in colors]

plt.figure(figsize=(3.5, 4))
plt.bar(labels, values, color=colors, width=.8)
plt.ylabel('Energy (pJ)')

figname = equation_type + "_energy" + ".svg"
plt.savefig(figname, dpi=300, format="svg")
plt.show()

