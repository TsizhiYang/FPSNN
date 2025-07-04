import torch


x_step = 40


def init_func(x):
    return torch.sin(x * torch.pi)
