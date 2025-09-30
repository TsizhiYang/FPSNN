from packs.SaveSystemMessage import save_system_message
save_system_message(True)               # save the system stdout and stderr to files

from execute_params import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

if network_type == "snn":
    from packs.networks.spiking_network_2d import *
elif network_type == "ann":
    from packs.networks.artificial_network_2d import *


equation_type = "burgers2d"

from packs.initial_function.burgers2d_init import *
from packs.reference_generation.burgers2d_ref import *

t = torch.linspace(0, time, t_step + 1)
x = torch.linspace(0, 1, x_step + 1)
y = torch.linspace(0, 1, y_step + 1)
in_scale = torch.tensor([[0., time], [0., 1.], [0., 1.]])

t, x, y = torch.meshgrid(t, x, y)
input_mesh = torch.stack((t, x, y), dim=-1)

shape = input_mesh.shape
input = input_mesh.reshape(-1, 3).to(device)

error_crit = nn.MSELoss(reduction='mean')
criterion = nn.MSELoss(reduction="mean")

for trial in range(trials):

    if network_type == "snn":
        model = SNN(time_window=x_step,
                    in_scale=in_scale,
                    in_shape_as=input[0],
                    init_method=initialization).to(device)
    elif network_type == "ann":
        model = ANN(init_method=initialization).to(device)

    if task_type == "std":
        input.requires_grad_()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    rec__loss = []
    rec__phys_loss = []
    rec__cond_loss = []
    rec__error = []

    for epoch in tqdm(range(epoches + 1)):

        pred = model(input)
        pred_mesh = pred.reshape(shape[:3])

        if task_type != "dl":
            if task_type == "fdm":
                pred_x = (pred_mesh[:, 2:, :] - pred_mesh[:, :-2, :]) / (2 * dx)
                pred_xx = (pred_mesh[:, 2:, :] + pred_mesh[:, :-2, :] - 2 * pred_mesh[:, 1:-1, :]) / (dx ** 2)
                pred_y = (pred_mesh[:, :, 2:] - pred_mesh[:, :, :-2]) / (2 * dy)
                pred_yy = (pred_mesh[:, :, 2:] + pred_mesh[:, :, :-2] - 2 * pred_mesh[:, :, 1:-1]) / (dy ** 2)
                pred_t = (pred_mesh[2:, :, :] - pred_mesh[:-2, :, :]) / (2 * dt)
                phys_loss = criterion(pred_xx[1:-1, :, 1:-1] + pred_yy[1:-1, 1:-1, :], (pred_x[1:-1, :, 1:-1] + pred_y[1:-1, 1:-1, :]) * pred_mesh[1:-1, 1:-1, 1:-1] + pred_t[:, 1:-1, 1:-1])
                cond_loss = criterion(pred_mesh[0, :, :], output_mesh[0, :, :].to(device)) + \
                            criterion(pred_mesh[:, [0, -1], :], output_mesh[:, [0, -1], :].to(device)) + \
                            criterion(pred_mesh[:, :, [0, -1]], output_mesh[:, :, [0, -1]].to(device))
            elif task_type == "std":
                pred_d = torch.autograd.grad(outputs=pred,
                                        inputs=input,
                                        grad_outputs=torch.ones_like(pred),
                                        create_graph=True, retain_graph=True)[0]
                pred_t = pred_d[:, 0].reshape(shape[:3])
                pred_x = pred_d[:, 1].reshape(shape[:3])
                pred_xd = torch.autograd.grad(outputs=pred_x,
                                                inputs=input,
                                                grad_outputs=torch.ones_like(pred_x),
                                                create_graph=True, retain_graph=True)[0]
                pred_xx = pred_xd[:, 1].reshape(shape[:3])
                pred_y = pred_d[:, 2].reshape(shape[:3])
                pred_yd = torch.autograd.grad(outputs=pred_y,
                                                inputs=input,
                                                grad_outputs=torch.ones_like(pred_y),
                                                create_graph=True, retain_graph=True)[0]
                pred_yy = pred_yd[:, 2].reshape(shape[:3])
                phys_loss = criterion(pred_t, pred_mesh * pred_x + pred_xx)

                cond_loss = criterion(pred_mesh[0, :, :], output_mesh[0, :, :].to(device)) + \
                            criterion(pred_mesh[:, 0, :], output_mesh[:, 0, :].to(device)) + \
                            criterion(pred_mesh[:, -1, :], output_mesh[:, -1, :].to(device)) + \
                            criterion(pred_mesh[:, :, 0], output_mesh[:, :, 0].to(device)) + \
                            criterion(pred_mesh[:, :, -1], output_mesh[:, :, -1].to(device))

            loss = phys_weight * phys_loss + cond_weight * cond_loss
        elif task_type == "dl":
            loss = error_crit(output, pred)
        error = error_crit(output, pred)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % record_epoch == 0:
            rec__loss.append(loss.item())
            if task_type != "dl":
                rec__phys_loss.append(phys_loss.item())
                rec__cond_loss.append(cond_loss.item())
            rec__error.append(error.item())
        if epoch % show_epoch == 0:
            if task_type != "dl":
                print(f'\n'
                      f'trail {trial + 1}, epoch {epoch}:\n'
                      f'    loss = {loss.item()},\n'
                      f'        physics = {phys_loss.item()}, conditional = {cond_loss.item()},\n'
                      f'    error = {error.item()}.')
            elif task_type == "dl":
                print(f'\n'
                      f'trail {trial + 1}, epoch {epoch}:\n'
                      f'    loss = {loss.item()},\n'
                      f'    error = {error.item()}.')
        if epoch % graph_epoch == 0 and (show_zero_epoch_graph if (epoch == 0) else True):
            pred_mesh = pred.reshape(shape[:3]).detach().cpu()

            fig = plt.figure(1)
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            ax1.plot_surface(input_mesh[0, :, :, 1], input_mesh[0, :, :, 2], output_mesh[0, :, :], rstride=1, cstride=1, cmap='coolwarm', alpha=0.6, linewidth=0, edgecolor='none')
            plt.xlabel("x")
            plt.ylabel("y")
            ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            ax2.plot_surface(input_mesh[0, :, :, 1], input_mesh[0, :, :, 2], pred_mesh[0, :, :], rstride=1, cstride=1, cmap='coolwarm', alpha=0.6, linewidth=0, edgecolor='none')
            plt.xlabel("x")
            plt.ylabel("y")
            ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            ax3.plot_surface(input_mesh[-1, :, :, 1], input_mesh[-1, :, :, 2], output_mesh[-1, :, :], rstride=1, cstride=1, cmap='coolwarm', alpha=0.6, linewidth=0, edgecolor='none')
            plt.xlabel("x")
            plt.ylabel("y")
            ax4 = fig.add_subplot(2, 2, 4, projection='3d')
            ax4.plot_surface(input_mesh[-1, :, :, 1], input_mesh[-1, :, :, 2], pred_mesh[-1, :, :], rstride=1, cstride=1, cmap='coolwarm', alpha=0.6, linewidth=0, edgecolor='none')
            plt.xlabel("x")
            plt.ylabel("y")

            plt.suptitle(label + f", epoch {epoch}")
            # plt.show()
            plt.close()

    torch.save(model.state_dict(), "model_params/" + equation_type + '/' + name + "_" + str(trial) + ".pth")

    pred_mesh = pred.reshape(shape[:3]).detach().cpu()

    fig = plt.figure(1)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot_surface(input_mesh[0, :, :, 1], input_mesh[0, :, :, 2], output_mesh[0, :, :], rstride=1, cstride=1, cmap='coolwarm', alpha=0.6, linewidth=0, edgecolor='none')
    plt.xlabel("x")
    plt.ylabel("y")
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot_surface(input_mesh[0, :, :, 1], input_mesh[0, :, :, 2], pred_mesh[0, :, :], rstride=1, cstride=1, cmap='coolwarm', alpha=0.6, linewidth=0, edgecolor='none')
    plt.xlabel("x")
    plt.ylabel("y")
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.plot_surface(input_mesh[-1, :, :, 1], input_mesh[-1, :, :, 2], output_mesh[-1, :, :], rstride=1, cstride=1, cmap='coolwarm', alpha=0.6, linewidth=0, edgecolor='none')
    plt.xlabel("x")
    plt.ylabel("y")
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot_surface(input_mesh[-1, :, :, 1], input_mesh[-1, :, :, 2], pred_mesh[-1, :, :], rstride=1, cstride=1, cmap='coolwarm', alpha=0.6, linewidth=0, edgecolor='none')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.suptitle(label + f", final epoch {epoches}")
    folder = os.getcwd()
    figname = folder + '/data/' + equation_type + '/_res_' + name + "_" + str(trial) + ".svg"
    plt.savefig(figname, dpi=300, format="svg")
    # plt.show()
    plt.close()

    rec__error = np.log10(np.array(rec__error))
    rec__loss = np.log10(np.array(rec__loss))
    if task_type != "dl":
        rec__phys_loss = np.log10(np.array(rec__phys_loss))
        rec__cond_loss = np.log10(np.array(rec__cond_loss))

    x_ax = np.array([i * record_epoch for i in range(int(epoches / record_epoch) + 1)])
    fig = plt.figure(1)
    plt.plot(x_ax, rec__error, c='red', label=label)
    plt.suptitle("Error, log scale")
    plt.legend()
    folder = os.getcwd()
    figname = folder + '/data/' + equation_type + '/' + name + "_" + str(trial) + ".svg"
    plt.savefig(figname, dpi=300, format="svg")
    # plt.show()
    plt.close()

    if task_type != "dl":
        sv = np.stack([x_ax, rec__error, rec__loss, rec__cond_loss, rec__phys_loss]).transpose([0, 1])
    elif task_type == "dl":
        sv = np.stack([x_ax, rec__error, rec__loss]).transpose([0, 1])
    folder = os.getcwd()
    rd_name = folder + '/data/' + equation_type + '/' + name + "_" + str(trial) + '.csv'
    with open(rd_name, 'w+'):
        np.savetxt(rd_name, sv, delimiter=',', fmt='%.6f')
