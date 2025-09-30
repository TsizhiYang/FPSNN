from execute_params import *

import numpy as np
import matplotlib.pyplot as plt

import os


def running_mean_5(arr):
    data_1 = np.concatenate((arr[:, 2:], np.repeat(arr[:, -1:], 2, axis=1)), axis=1)
    data_2 = np.concatenate((arr[:, 1:], np.repeat(arr[:, -1:], 1, axis=1)), axis=1)
    data_4 = np.concatenate((np.repeat(arr[:, :1], 1, axis=1), arr[:, :-1]), axis=1)
    data_5 = np.concatenate((np.repeat(arr[:, :1], 2, axis=1), arr[:, :-2]), axis=1)
    return (data_1 + data_2 + arr + data_4 + data_5) / 5

config = {
    "mathtext.fontset": 'stix',
}

plt.rcParams.update(config)

plt.rcParams['font.sans-serif'] = "Times New Roman"

color_dict = {
    "FPSNN": "red",
    "FPANN": "blue",
    "APSNN 1": "green",
    "APSNN 2": "mediumseagreen",
    "DDSNN": "orange",
}

process_equation_type = "wave"              # "heat", "wave", "burgers2d"
process_SNN_FDM = True
process_SNN_DL  = False
process_ANN_FDM = True
process_SNN_STD = False
process_SNN_STD_sigmoid = False

plot_step = 1

if process_equation_type == "heat":
    process_epoches = 200000
    x_vals = [0, 50, 100, 150, 200]
    x_labs = ['0', '50', '100', '150', '200k']
if process_equation_type == "wave":
    process_epoches = 250000
    x_vals = [0, 50, 100, 150, 200, 250]
    x_labs = ['0', '50', '100', '150', '200', '250k']
if process_equation_type == "burgers2d":
    process_epoches = 250000
    x_vals = [0, 50, 100, 150, 200, 250]
    x_labs = ['0', '50', '100', '150', '200', '250k']

graphs = range(trials)

folder = os.getcwd() + '/data/' + process_equation_type + '/' 

x_ax = np.array([i * record_epoch / 1000 for i in range(int(process_epoches / record_epoch) + 1)])
limit = int(process_epoches / record_epoch)

fig = plt.figure(1, figsize=(9, 3.5))

ax = fig.add_subplot(1, 2, 1)

if process_ANN_FDM:
    ANN_FDM_data = []
    for i in graphs:
        ANN_FDM_data.append(np.loadtxt(folder + "ANN_FDM_" + str(i) + '.csv', delimiter=','))
    ANN_FDM_data = np.stack([ANN_FDM_data[i][1] for i in range(len(graphs))])

    ANN_FDM_data = running_mean_5(ANN_FDM_data)

    ANN_FDM_res = np.stack([np.average(ANN_FDM_data, axis=0) - np.std(ANN_FDM_data, axis=0), np.average(ANN_FDM_data, axis=0) + np.std(ANN_FDM_data, axis=0), np.average(ANN_FDM_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], ANN_FDM_res[0][:limit:plot_step], ANN_FDM_res[1][:limit:plot_step], facecolor=color_dict["FPANN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], ANN_FDM_res[2][:limit:plot_step], c=color_dict["FPANN"], label='FPANN')
if process_SNN_STD:
    SNN_STD_data = []
    for i in graphs:
        SNN_STD_data.append(np.loadtxt(folder + "SNN_STD_" + str(i) + '.csv', delimiter=','))
    SNN_STD_data = np.stack([SNN_STD_data[i][1] for i in range(len(graphs))])

    SNN_STD_data = running_mean_5(SNN_STD_data)

    # SNN_STD_res = np.stack([np.average(SNN_STD_data, axis=0) - np.std(SNN_STD_data, axis=0), np.average(SNN_STD_data, axis=0) + np.std(SNN_STD_data, axis=0), np.average(SNN_STD_data, axis=0)]) * (1.5 if process_equation_type == "wave" else 1)
    SNN_STD_res = np.stack([np.average(SNN_STD_data, axis=0) - np.std(SNN_STD_data, axis=0), np.average(SNN_STD_data, axis=0) + np.std(SNN_STD_data, axis=0), np.average(SNN_STD_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_STD_res[0][:limit:plot_step], SNN_STD_res[1][:limit:plot_step], facecolor=color_dict["APSNN 1"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_STD_res[2][:limit:plot_step], c=color_dict["APSNN 1"], label='APSNN 1')
if process_SNN_STD_sigmoid:
    SNN_STD_Sigmoid_data = []
    for i in graphs:
        SNN_STD_Sigmoid_data.append(np.loadtxt(folder + "SNN_STD_Sigmoid_" + str(i) + '.csv', delimiter=','))
    SNN_STD_Sigmoid_data = np.stack([SNN_STD_Sigmoid_data[i][1] for i in range(len(graphs))])

    SNN_STD_Sigmoid_data = running_mean_5(SNN_STD_Sigmoid_data)

    SNN_STD_Sigmoid_res = np.stack([np.average(SNN_STD_Sigmoid_data, axis=0) - np.std(SNN_STD_Sigmoid_data, axis=0), np.average(SNN_STD_Sigmoid_data, axis=0) + np.std(SNN_STD_Sigmoid_data, axis=0), np.average(SNN_STD_Sigmoid_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_STD_Sigmoid_res[0][:limit:plot_step], SNN_STD_Sigmoid_res[1][:limit:plot_step], facecolor=color_dict["APSNN 2"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_STD_Sigmoid_res[2][:limit:plot_step], c=color_dict["APSNN 2"], label='APSNN 2')
if process_SNN_DL:
    SNN_DL_data = []
    for i in graphs:
        SNN_DL_data.append(np.loadtxt(folder + "SNN_DL_" + str(i) + '.csv', delimiter=','))
    SNN_DL_data = np.stack([SNN_DL_data[i][1] for i in range(len(graphs))])

    SNN_DL_data = running_mean_5(SNN_DL_data)

    SNN_DL_res = np.stack([np.average(SNN_DL_data, axis=0) - np.std(SNN_DL_data, axis=0), np.average(SNN_DL_data, axis=0) + np.std(SNN_DL_data, axis=0), np.average(SNN_DL_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_DL_res[0][:limit:plot_step], SNN_DL_res[1][:limit:plot_step], facecolor=color_dict["DDSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_DL_res[2][:limit:plot_step], c=color_dict["DDSNN"], label='DDSNN')
if process_SNN_FDM:
    SNN_FDM_data = []
    for i in graphs:
        SNN_FDM_data.append(np.loadtxt(folder + "SNN_FDM_" + str(i) + '.csv', delimiter=','))
    SNN_FDM_data = np.stack([SNN_FDM_data[i][1] for i in range(len(graphs))])

    SNN_FDM_data = running_mean_5(SNN_FDM_data)
    print(SNN_FDM_data.shape)
    SNN_FDM_res = np.stack([np.average(SNN_FDM_data, axis=0) - np.std(SNN_FDM_data, axis=0), np.average(SNN_FDM_data, axis=0) + np.std(SNN_FDM_data, axis=0), np.average(SNN_FDM_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_FDM_res[0][:limit:plot_step], SNN_FDM_res[1][:limit:plot_step], facecolor=color_dict["FPSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_FDM_res[2][:limit:plot_step], c=color_dict["FPSNN"], label='FPSNN')
    print("FPSNN error:", 10 ** np.average(SNN_FDM_data, axis=0).min())

plt.xticks(x_vals, x_labs)
plt.xlabel("Epochs")
plt.ylabel("Error ($\log_{10}$)")
ax.set_title("(a) Error evolution", pad=10)
plt.legend(loc="upper right")


ax = fig.add_subplot(1, 2, 2)

if process_ANN_FDM:
    ANN_FDM_data = []
    for i in graphs:
        ANN_FDM_data.append(np.loadtxt(folder + "ANN_FDM_" + str(i) + '.csv', delimiter=','))
    ANN_FDM_data = np.stack([ANN_FDM_data[i][2] for i in range(len(graphs))])

    ANN_FDM_data = running_mean_5(ANN_FDM_data)

    ANN_FDM_res = np.stack([np.average(ANN_FDM_data, axis=0) - np.std(ANN_FDM_data, axis=0), np.average(ANN_FDM_data, axis=0) + np.std(ANN_FDM_data, axis=0), np.average(ANN_FDM_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], ANN_FDM_res[0][:limit:plot_step], ANN_FDM_res[1][:limit:plot_step], facecolor=color_dict["FPANN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], ANN_FDM_res[2][:limit:plot_step], c=color_dict["FPANN"], label='FPANN')
if process_SNN_STD:
    SNN_STD_data = []
    for i in graphs:
        SNN_STD_data.append(np.loadtxt(folder + "SNN_STD_" + str(i) + '.csv', delimiter=','))
    SNN_STD_data = np.stack([SNN_STD_data[i][2] for i in range(len(graphs))])

    SNN_STD_data = running_mean_5(SNN_STD_data)

    SNN_STD_res = np.stack(
        [np.average(SNN_STD_data, axis=0) - np.std(SNN_STD_data, axis=0), np.average(SNN_STD_data, axis=0) + np.std(SNN_STD_data, axis=0), np.average(SNN_STD_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_STD_res[0][:limit:plot_step], SNN_STD_res[1][:limit:plot_step], facecolor=color_dict["APSNN 1"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_STD_res[2][:limit:plot_step], c=color_dict["APSNN 1"], label='APSNN 1')
if process_SNN_STD_sigmoid:
    SNN_STD_Sigmoid_data = []
    for i in graphs:
        SNN_STD_Sigmoid_data.append(np.loadtxt(folder + "SNN_STD_Sigmoid_" + str(i) + '.csv', delimiter=','))
    SNN_STD_Sigmoid_data = np.stack([SNN_STD_Sigmoid_data[i][2] for i in range(len(graphs))])

    SNN_STD_Sigmoid_data = running_mean_5(SNN_STD_Sigmoid_data)

    SNN_STD_Sigmoid_res = np.stack([np.average(SNN_STD_Sigmoid_data, axis=0) - np.std(SNN_STD_Sigmoid_data, axis=0), np.average(SNN_STD_Sigmoid_data, axis=0) + np.std(SNN_STD_Sigmoid_data, axis=0), np.average(SNN_STD_Sigmoid_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_STD_Sigmoid_res[0][:limit:plot_step], SNN_STD_Sigmoid_res[1][:limit:plot_step], facecolor=color_dict["APSNN 2"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_STD_Sigmoid_res[2][:limit:plot_step], c=color_dict["APSNN 2"], label='APSNN 2')
if process_SNN_DL:
    SNN_DL_data = []
    for i in graphs:
        SNN_DL_data.append(np.loadtxt(folder + "SNN_DL_" + str(i) + '.csv', delimiter=','))
    SNN_DL_data = np.stack([SNN_DL_data[i][2] for i in range(len(graphs))])
    SNN_DL_res = np.stack(
        [np.average(SNN_DL_data, axis=0) - np.std(SNN_DL_data, axis=0), np.average(SNN_DL_data, axis=0) + np.std(SNN_DL_data, axis=0), np.average(SNN_DL_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_DL_res[0][:limit:plot_step], SNN_DL_res[1][:limit:plot_step], facecolor=color_dict["DDSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_DL_res[2][:limit:plot_step], c=color_dict["DDSNN"], label='DDSNN')
if process_SNN_FDM:
    SNN_FDM_data = []
    for i in graphs:
        SNN_FDM_data.append(np.loadtxt(folder + "SNN_FDM_" + str(i) + '.csv', delimiter=','))
    SNN_FDM_data = np.stack([SNN_FDM_data[i][2] for i in range(len(graphs))])

    SNN_FDM_data = running_mean_5(SNN_FDM_data)

    SNN_FDM_res = np.stack([np.average(SNN_FDM_data, axis=0) - np.std(SNN_FDM_data, axis=0), np.average(SNN_FDM_data, axis=0) + np.std(SNN_FDM_data, axis=0), np.average(SNN_FDM_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_FDM_res[0][:limit:plot_step], SNN_FDM_res[1][:limit:plot_step], facecolor=color_dict["FPSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_FDM_res[2][:limit:plot_step], c=color_dict["FPSNN"], label='FPSNN')

plt.xticks(x_vals, x_labs)
plt.xlabel("Epochs")
plt.ylabel("Loss ($\log_{10}$)")
ax.set_title("(b) Loss evolution", pad=10)
plt.legend(loc="upper right")


plt.tight_layout(pad=1, w_pad=3, h_pad=2)

figname = "fig-" + process_equation_type + "_error_loss" + ".svg"
plt.savefig(figname, dpi=300, format="svg")
plt.show()
print("image saved as", figname)

