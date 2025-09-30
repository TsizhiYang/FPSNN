import torch


x_step = 40
y_step = 40


def init_func(input):
    x, y = input
    return torch.sin(x * torch.pi) * torch.cos(y * torch.pi)