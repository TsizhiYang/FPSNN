from execute_params import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

if network_type == "snn":
    from packs.networks.spiking_network import *
elif network_type == "ann":
    from packs.networks.artificial_network import *

if equation_type == "heat":
    from packs.initial_function.heat_init import *
    from packs.reference_generation.heat_ref import *
elif equation_type == "wave":
    from packs.initial_function.wave_init import *
    from packs.reference_generation.wave_ref import *

x = torch.linspace(0, 1, x_step + 1)
t = torch.linspace(0, time, t_step + 1)
x = torch.stack([x for _ in range(t_step + 1)]).unsqueeze(-1)
t = torch.stack([t for _ in range(x_step + 1)]).unsqueeze(-1)
input_mesh = torch.cat((t.permute(1, 0, 2), x), dim=2)  # format: input_mesh[i, j] = (t_{i+1}, x_{j+1})
shape = input_mesh.shape
input = input_mesh.reshape(-1, 2).to(device)

error_crit = nn.MSELoss(reduction='mean')
criterion = nn.MSELoss(reduction="mean")

for trial in range(trials):

    if network_type == "snn":
        model = SNN(time_window=x_step,
                    in_scale=torch.tensor([[0., time], [0., 1.]]),
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
        pred_mesh = pred.reshape(shape[:2])

        if task_type != "dl":
            if task_type == "fdm":
                if equation_type == "heat":
                    pred_xx = (pred_mesh[:, 2:] + pred_mesh[:, :-2] - 2 * pred_mesh[:, 1:-1]) / (dx ** 2)
                    pred_t = (pred_mesh[2:, :] - pred_mesh[:-2, :]) / (2 * dt)
                    phys_loss = criterion(pred_xx[1:-1, :], pred_t[:, 1:-1])
                    cond_loss = criterion(pred_mesh[0, :], output_mesh[0, :].to(device))
                elif equation_type == "wave":
                    pred_augmesh = torch.cat((pred_mesh[0, :].unsqueeze(0), pred_mesh))
                    pred_xx = (pred_augmesh[:, 2:] + pred_augmesh[:, :-2] - 2 * pred_augmesh[:, 1:-1]) / (dx ** 2)
                    pred_tt = (pred_augmesh[2:, :] + pred_augmesh[:-2, :] - 2 * pred_augmesh[1:-1, :]) / (dt ** 2)
                    phys_loss = criterion(pred_xx[1:-1, :], pred_tt[:, 1:-1])
                    cond_loss = criterion(pred_mesh[0, :], output_mesh[0, :].to(device)) + \
                                criterion(pred_mesh[:, 0], torch.zeros_like(pred_mesh[:, 0])) + \
                                criterion(pred_mesh[:, -1], torch.zeros_like(pred_mesh[:, -1]))
                elif equation_type == "laplace":
                    pred_x = (pred_mesh[:, 2:] - pred_mesh[:, :-2]) / (2 * dx)
                    pred_xx = (pred_mesh[:, 2:] + pred_mesh[:, :-2] - 2 * pred_mesh[:, 1:-1]) / (dx ** 2)
                    pred_t = (pred_mesh[2:, :] - pred_mesh[:-2, :]) / (2 * dt)
                    phys_loss = criterion(pred_xx[1:-1, :] - pred_x[1:-1, :] * pred_mesh[1:-1, 1:-1] / c, pred_t[:, 1:-1] / c)
                    cond_loss = criterion(pred_mesh[0, :], output_mesh[0, :].to(device)) + \
                                criterion(pred_mesh[:, 0], output_mesh[:, 0].to(device)) + \
                                criterion(pred_mesh[:, -1], output_mesh[:, -1].to(device))
            elif task_type == "std":
                if equation_type == "heat":
                    pred_d = torch.autograd.grad(outputs=pred, inputs=input,
                                                 grad_outputs=torch.ones_like(pred),
                                                 create_graph=True, retain_graph=True)[0]
                    pred_t = pred_d[:, 0].reshape(shape[:2])
                    pred_x = pred_d[:, 1]
                    pred_xd = torch.autograd.grad(outputs=pred_x, inputs=input,
                                                  grad_outputs=torch.ones_like(pred_x),
                                                  create_graph=True, retain_graph=True)[0]
                    pred_xx = pred_xd[:, 1].reshape(shape[:2])
                    phys_loss = criterion(pred_xx, pred_t)
                    cond_loss = criterion(pred_mesh[0, :], output_mesh[0, :].to(device))
                elif equation_type == "wave":
                    pred_d = torch.autograd.grad(outputs=pred, inputs=input,
                                                 grad_outputs=torch.ones_like(pred),
                                                 create_graph=True, retain_graph=True)[0]
                    pred_t = pred_d[:, 0].reshape(shape[:2])
                    pred_td = torch.autograd.grad(outputs=pred_t, inputs=input,
                                                  grad_outputs=torch.ones_like(pred_t),
                                                  create_graph=True, retain_graph=True)[0]
                    pred_tt = pred_td[:, 0].reshape(shape[:2])
                    pred_x = pred_d[:, 1]
                    pred_xd = torch.autograd.grad(outputs=pred_x, inputs=input,
                                                  grad_outputs=torch.ones_like(pred_x),
                                                  create_graph=True, retain_graph=True)[0]
                    pred_xx = pred_xd[:, 1].reshape(shape[:2])
                    phys_loss = criterion(pred_xx, pred_tt)
                    cond_loss = criterion(pred_mesh[0, :], output_mesh[0, :].to(device)) + \
                                criterion(pred_t[0, :], torch.zeros_like(pred_t[0, :])) + \
                                criterion(pred_mesh[:, 0], torch.zeros_like(pred_mesh[:, 0])) + \
                                criterion(pred_mesh[:, -1], torch.zeros_like(pred_mesh[:, -1]))
                elif equation_type == "laplace":
                    pred_d = torch.autograd.grad(outputs=pred,
                                         inputs=input,
                                         grad_outputs=torch.ones_like(pred),
                                         create_graph=True, retain_graph=True)[0]
                    pred_t = pred_d[:, 1].reshape(shape[:2])
                    pred_td = torch.autograd.grad(outputs=pred_t,
                                                  inputs=input,
                                                  grad_outputs=torch.ones_like(pred_t),
                                                  create_graph=True, retain_graph=True)[0]
                    pred_tt = pred_td[:, 1].reshape(shape[:2])
                    pred_x = pred_d[:, 0]
                    pred_xd = torch.autograd.grad(outputs=pred_x,
                                                 inputs=input,
                                                 grad_outputs=torch.ones_like(pred_x),
                                                 create_graph=True, retain_graph=True)[0]
                    pred_xx = pred_xd[:, 0].reshape(shape[:2])
                    phys_loss = criterion(pred_xx, pred_tt)
                    cond_loss = criterion(pred_mesh[0, :], output_mesh[0, :].to(device)) + \
                                criterion(pred_t[0, :], torch.zeros_like(pred_t[0, :])) + \
                                criterion(pred_mesh[:, 0], torch.zeros_like(pred_mesh[:, 0])) + \
                                criterion(pred_mesh[:, -1], torch.zeros_like(pred_mesh[:, -1]))

            loss = phys_weight * phys_loss + cond_weight * cond_loss
        elif task_type == "dl":
            loss = error_crit(output, pred)
            # loss = error_crit(output_mesh[:30, :].to(device), pred_mesh[:30, :])
        error = error_crit(output, pred)

        if epoch % record_epoch == 0:
            rec__loss.append(loss.item())
            if task_type != "dl":
                rec__phys_loss.append(phys_loss.item())
                rec__cond_loss.append(cond_loss.item())
            rec__error.append(error.item())
            if error.item() <= min(rec__error):
                torch.save(model.state_dict(), "model_params/" + equation_type + '/' + name + "_" + str(trial) + ".pth")
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
            pred_mesh = pred.reshape(shape[:2]).detach().cpu()

            fig = plt.figure(1)
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            t_show = int(t_step / t_axis_lines)
            for i in range(output_mesh.shape[0]):
                if i % t_show == 0:
                    ax1.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], output_mesh[i], c='red', alpha=0.2)
                    plt.xlabel("t")
                    plt.ylabel("x")
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            for i in range(output_mesh.shape[0]):
                if i % t_show == 0:
                    ax2.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], pred_mesh[i], c='blue', alpha=0.2)
                    plt.xlabel("t")
                    plt.ylabel("x")
            ax = fig.add_subplot(1, 3, 3, projection='3d')
            for i in range(output_mesh.shape[0]):
                if i % t_show == 0:
                    ax.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], output_mesh[i], c='red', alpha=0.2)
                    ax.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], pred_mesh[i], c='blue', alpha=0.2)
                    plt.xlabel("t")
                    plt.ylabel("x")

            plt.suptitle(label + f", epoch {epoch}")

            plt.show()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    pred_mesh = pred.reshape(shape[:2]).detach().cpu()

    fig = plt.figure(1)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    t_show = int(t_step / t_axis_lines)
    for i in range(output_mesh.shape[0]):
        if i % t_show == 0:
            ax1.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], output_mesh[i], c='red', alpha=0.2)
            plt.xlabel("t")
            plt.ylabel("x")
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    for i in range(output_mesh.shape[0]):
        if i % t_show == 0:
            ax2.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], pred_mesh[i], c='blue', alpha=0.2)
            plt.xlabel("t")
            plt.ylabel("x")
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    for i in range(output_mesh.shape[0]):
        if i % t_show == 0:
            ax.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], output_mesh[i], c='red', alpha=0.2)
            ax.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], pred_mesh[i], c='blue', alpha=0.2)
            plt.xlabel("t")
            plt.ylabel("x")

    plt.suptitle(label + f", final epoch {epoches}")
    folder = os.getcwd()
    figname = folder + '\\data\\' + equation_type + '\\' + name + "_" + str(trial) + ".svg"
    plt.savefig(figname, dpi=300, format="svg")
    plt.show()

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
    figname = folder + '\\data\\' + equation_type + '\\' + name + "_" + str(trial) + ".svg"
    plt.savefig(figname, dpi=300, format="svg")

    plt.show()

    if task_type != "dl":
        sv = np.stack([x_ax, rec__error, rec__loss, rec__cond_loss, rec__phys_loss]).transpose([0, 1])
    elif task_type == "dl":
        sv = np.stack([x_ax, rec__error, rec__loss]).transpose([0, 1])
    folder = os.getcwd()
    rd_name = folder + '\\data\\' + equation_type + '\\' + name + "_" + str(trial) + '.csv'
    with open(rd_name, 'w+'):
        np.savetxt(rd_name, sv, delimiter=',', fmt='%.6f')
