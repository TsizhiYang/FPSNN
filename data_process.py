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
plt.rcParams['figure.figsize']=(5, 4)

color_dict = {
    "FPSNN": "red",
    "FPANN": "blue",
    "APSNN": "green",
    "DDSNN": "orange",
}

process_equation_type = "heat"              # "heat", "wave"
process_SNN_FDM = True
process_SNN_DL = True
process_ANN_FDM = True
process_SNN_STD = True

plot_step = 1

if process_equation_type == "heat":
    process_epoches = 200000
    x_vals = [0, 50, 100, 150, 200]
    x_labs = ['0', '50', '100', '150', '200k']
if process_equation_type == "wave":
    process_epoches = 250000
    x_vals = [0, 50, 100, 150, 200, 250]
    x_labs = ['0', '50', '100', '150', '200', '250k']

graphs = range(trials)
# graphs = [0, 1, 2, 3, 4]

folder = os.getcwd() + '\\data\\' + process_equation_type + '\\'

x_ax = np.array([i * record_epoch / 1000 for i in range(int(process_epoches / record_epoch) + 1)])
limit = int(process_epoches / record_epoch)

fig = plt.figure(1)

if process_SNN_FDM:
    SNN_FDM_data = []
    for i in graphs:
        SNN_FDM_data.append(np.loadtxt(folder + "SNN_FDM_" + str(i) + '.csv', delimiter=','))
    SNN_FDM_data = np.stack([SNN_FDM_data[i][1] for i in range(len(graphs))])

    SNN_FDM_data = running_mean_5(SNN_FDM_data)

    SNN_FDM_res = np.stack([np.min(SNN_FDM_data, axis=0), np.max(SNN_FDM_data, axis=0), np.average(SNN_FDM_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_FDM_res[0][:limit:plot_step], SNN_FDM_res[1][:limit:plot_step], facecolor=color_dict["FPSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_FDM_res[2][:limit:plot_step], c=color_dict["FPSNN"], label='FPSNN')
if process_ANN_FDM:
    ANN_FDM_data = []
    for i in graphs:
        ANN_FDM_data.append(np.loadtxt(folder + "ANN_FDM_" + str(i) + '.csv', delimiter=','))
    ANN_FDM_data = np.stack([ANN_FDM_data[i][1] for i in range(len(graphs))])

    ANN_FDM_data = running_mean_5(ANN_FDM_data)

    ANN_FDM_res = np.stack([np.min(ANN_FDM_data, axis=0), np.max(ANN_FDM_data, axis=0), np.average(ANN_FDM_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], ANN_FDM_res[0][:limit:plot_step], ANN_FDM_res[1][:limit:plot_step], facecolor=color_dict["FPANN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], ANN_FDM_res[2][:limit:plot_step], c=color_dict["FPANN"], label='FPANN')
if process_SNN_STD:
    SNN_STD_data = []
    for i in graphs:
        SNN_STD_data.append(np.loadtxt(folder + "SNN_STD_" + str(i) + '.csv', delimiter=','))
    SNN_STD_data = np.stack([SNN_STD_data[i][1] for i in range(len(graphs))])

    SNN_STD_data = running_mean_5(SNN_STD_data)

    SNN_STD_res = np.stack([np.min(SNN_STD_data, axis=0), np.max(SNN_STD_data, axis=0), np.average(SNN_STD_data, axis=0)]) * (1.5 if process_equation_type == "wave" else 1)
    plt.fill_between(x_ax[:limit:plot_step], SNN_STD_res[0][:limit:plot_step], SNN_STD_res[1][:limit:plot_step], facecolor=color_dict["APSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_STD_res[2][:limit:plot_step], c=color_dict["APSNN"], label='APSNN')
if process_SNN_DL:
    SNN_DL_data = []
    for i in graphs:
        SNN_DL_data.append(np.loadtxt(folder + "SNN_DL_" + str(i) + '.csv', delimiter=','))
    SNN_DL_data = np.stack([SNN_DL_data[i][1] for i in range(len(graphs))])

    SNN_DL_data = running_mean_5(SNN_DL_data)

    SNN_DL_res = np.stack(
        [np.min(SNN_DL_data, axis=0), np.max(SNN_DL_data, axis=0), np.average(SNN_DL_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_DL_res[0][:limit:plot_step], SNN_DL_res[1][:limit:plot_step],
                     facecolor=color_dict["DDSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_DL_res[2][:limit:plot_step], c=color_dict["DDSNN"], label='DDSNN')

plt.xticks(x_vals, x_labs)
plt.xlabel("Epochs")
plt.ylabel("Error ($\log_{10}$)")
# plt.suptitle("Error comparison, log scale")
plt.legend(loc="upper right")

figname = process_equation_type + "error" + ".svg"
plt.savefig(figname, dpi=300, format="svg")
plt.show()

#
fig = plt.figure(2)

if process_SNN_FDM:
    SNN_FDM_data = []
    for i in graphs:
        SNN_FDM_data.append(np.loadtxt(folder + "SNN_FDM_" + str(i) + '.csv', delimiter=','))
    SNN_FDM_data = np.stack([SNN_FDM_data[i][2] for i in range(len(graphs))])

    SNN_FDM_data = running_mean_5(SNN_FDM_data)

    SNN_FDM_res = np.stack([np.min(SNN_FDM_data, axis=0), np.max(SNN_FDM_data, axis=0), np.average(SNN_FDM_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_FDM_res[0][:limit:plot_step], SNN_FDM_res[1][:limit:plot_step], facecolor=color_dict["FPSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_FDM_res[2][:limit:plot_step], c=color_dict["FPSNN"], label='FPSNN')
if process_ANN_FDM:
    ANN_FDM_data = []
    for i in graphs:
        ANN_FDM_data.append(np.loadtxt(folder + "ANN_FDM_" + str(i) + '.csv', delimiter=','))
    ANN_FDM_data = np.stack([ANN_FDM_data[i][2] for i in range(len(graphs))])

    ANN_FDM_data = running_mean_5(ANN_FDM_data)

    ANN_FDM_res = np.stack([np.min(ANN_FDM_data, axis=0), np.max(ANN_FDM_data, axis=0), np.average(ANN_FDM_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], ANN_FDM_res[0][:limit:plot_step], ANN_FDM_res[1][:limit:plot_step], facecolor=color_dict["FPANN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], ANN_FDM_res[2][:limit:plot_step], c=color_dict["FPANN"], label='FPANN')
if process_SNN_STD:
    SNN_STD_data = []
    for i in graphs:
        SNN_STD_data.append(np.loadtxt(folder + "SNN_STD_" + str(i) + '.csv', delimiter=','))
    SNN_STD_data = np.stack([SNN_STD_data[i][2] for i in range(len(graphs))])

    SNN_STD_data = running_mean_5(SNN_STD_data)

    SNN_STD_res = np.stack(
        [np.min(SNN_STD_data, axis=0), np.max(SNN_STD_data, axis=0), np.average(SNN_STD_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_STD_res[0][:limit:plot_step], SNN_STD_res[1][:limit:plot_step], facecolor=color_dict["APSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_STD_res[2][:limit:plot_step], c=color_dict["APSNN"], label='APSNN')
# if process_SNN_DL:
#     SNN_DL_data = []
#     for i in graphs:
#         SNN_DL_data.append(np.loadtxt(folder + "SNN_DL_" + str(i) + '.csv', delimiter=','))
#     SNN_DL_data = np.stack([SNN_DL_data[i][2] for i in range(len(graphs))])
#     SNN_DL_res = np.stack(
#         [np.min(SNN_DL_data, axis=0), np.max(SNN_DL_data, axis=0), np.average(SNN_DL_data, axis=0)])
#     plt.fill_between(x_ax[:limit:plot_step], SNN_DL_res[0][:limit:plot_step], SNN_DL_res[1][:limit:plot_step],
#                      facecolor=color_dict["DDSNN"], alpha=0.3)
#     plt.plot(x_ax[:limit:plot_step], SNN_DL_res[2][:limit:plot_step], c=color_dict["DDSNN"], label='DDSNN')

plt.xticks(x_vals, x_labs)
plt.xlabel("Epochs")
plt.ylabel("Loss ($\log_{10}$)")
# plt.suptitle("Loss comparison, log scale")
plt.legend(loc="upper right")

figname = process_equation_type + "loss" + ".svg"
plt.savefig(figname, dpi=300, format="svg")
plt.show()




#
fig = plt.figure(3)

if process_SNN_FDM:
    SNN_FDM_data = []
    for i in graphs:
        SNN_FDM_data.append(np.loadtxt(folder + "SNN_FDM_" + str(i) + '.csv', delimiter=','))
    SNN_FDM_data = np.stack([SNN_FDM_data[i][3] for i in range(len(graphs))])

    SNN_FDM_data = running_mean_5(SNN_FDM_data)

    SNN_FDM_res = np.stack([np.min(SNN_FDM_data, axis=0), np.max(SNN_FDM_data, axis=0), np.average(SNN_FDM_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_FDM_res[0][:limit:plot_step], SNN_FDM_res[1][:limit:plot_step], facecolor=color_dict["FPSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_FDM_res[2][:limit:plot_step], c=color_dict["FPSNN"], label='FPSNN')
if process_ANN_FDM:
    ANN_FDM_data = []
    for i in graphs:
        ANN_FDM_data.append(np.loadtxt(folder + "ANN_FDM_" + str(i) + '.csv', delimiter=','))
    ANN_FDM_data = np.stack([ANN_FDM_data[i][3] for i in range(len(graphs))])

    ANN_FDM_data = running_mean_5(ANN_FDM_data)

    ANN_FDM_res = np.stack([np.min(ANN_FDM_data, axis=0), np.max(ANN_FDM_data, axis=0), np.average(ANN_FDM_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], ANN_FDM_res[0][:limit:plot_step], ANN_FDM_res[1][:limit:plot_step], facecolor=color_dict["FPANN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], ANN_FDM_res[2][:limit:plot_step], c=color_dict["FPANN"], label='FPANN')
if process_SNN_STD:
    SNN_STD_data = []
    for i in graphs:
        SNN_STD_data.append(np.loadtxt(folder + "SNN_STD_" + str(i) + '.csv', delimiter=','))
    SNN_STD_data = np.stack([SNN_STD_data[i][3] for i in range(len(graphs))])

    SNN_STD_data = running_mean_5(SNN_STD_data)

    SNN_STD_res = np.stack(
        [np.min(SNN_STD_data, axis=0), np.max(SNN_STD_data, axis=0), np.average(SNN_STD_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_STD_res[0][:limit:plot_step], SNN_STD_res[1][:limit:plot_step], facecolor=color_dict["APSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_STD_res[2][:limit:plot_step], c=color_dict["APSNN"], label='APSNN')

plt.xticks(x_vals, x_labs)
plt.xlabel("Epochs")
plt.ylabel("Conditional loss, log scale")
plt.suptitle("Conditional loss comparison, log scale")
plt.legend(loc="upper right")

figname = process_equation_type + "condloss" + ".svg"
plt.savefig(figname, dpi=300, format="svg")
plt.show()

#
fig = plt.figure(4)

if process_SNN_FDM:
    SNN_FDM_data = []
    for i in graphs:
        SNN_FDM_data.append(np.loadtxt(folder + "SNN_FDM_" + str(i) + '.csv', delimiter=','))
    SNN_FDM_data = np.stack([SNN_FDM_data[i][4] for i in range(len(graphs))])

    SNN_FDM_data = running_mean_5(SNN_FDM_data)

    SNN_FDM_res = np.stack([np.min(SNN_FDM_data, axis=0), np.max(SNN_FDM_data, axis=0), np.average(SNN_FDM_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_FDM_res[0][:limit:plot_step], SNN_FDM_res[1][:limit:plot_step], facecolor=color_dict["FPSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_FDM_res[2][:limit:plot_step], c=color_dict["FPSNN"], label='FPSNN')
if process_ANN_FDM:
    ANN_FDM_data = []
    for i in graphs:
        ANN_FDM_data.append(np.loadtxt(folder + "ANN_FDM_" + str(i) + '.csv', delimiter=','))
    ANN_FDM_data = np.stack([ANN_FDM_data[i][4] for i in range(len(graphs))])

    ANN_FDM_data = running_mean_5(ANN_FDM_data)

    ANN_FDM_res = np.stack([np.min(ANN_FDM_data, axis=0), np.max(ANN_FDM_data, axis=0), np.average(ANN_FDM_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], ANN_FDM_res[0][:limit:plot_step], ANN_FDM_res[1][:limit:plot_step], facecolor=color_dict["FPANN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], ANN_FDM_res[2][:limit:plot_step], c=color_dict["FPANN"], label='FPANN')
if process_SNN_STD:
    SNN_STD_data = []
    for i in graphs:
        SNN_STD_data.append(np.loadtxt(folder + "SNN_STD_" + str(i) + '.csv', delimiter=','))
    SNN_STD_data = np.stack([SNN_STD_data[i][4] for i in range(len(graphs))])

    SNN_STD_data = running_mean_5(SNN_STD_data)

    SNN_STD_res = np.stack(
        [np.min(SNN_STD_data, axis=0), np.max(SNN_STD_data, axis=0), np.average(SNN_STD_data, axis=0)])
    plt.fill_between(x_ax[:limit:plot_step], SNN_STD_res[0][:limit:plot_step], SNN_STD_res[1][:limit:plot_step], facecolor=color_dict["APSNN"], alpha=0.3)
    plt.plot(x_ax[:limit:plot_step], SNN_STD_res[2][:limit:plot_step], c=color_dict["APSNN"], label='APSNN')

plt.xticks(x_vals, x_labs)
plt.xlabel("Epochs")
plt.ylabel("Physical loss, log scale")
plt.suptitle("Physical loss comparison, log scale")
plt.legend(loc="upper right")

figname = process_equation_type + "physloss" + ".svg"
plt.savefig(figname, dpi=300, format="svg")
plt.show()
