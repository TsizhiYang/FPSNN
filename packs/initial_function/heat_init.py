import torch


x_step = 40


def init_func(x):
    return torch.max(20 * torch.min(x - 0.45, 0.55 - x), torch.tensor([0 for _ in range(x_step + 1)]))
