from ..initial_function.heat_init import *
import matplotlib.pyplot as plt

device = torch.device("cuda")

alpha = 1
dx = 1 / x_step
dt = 0.4 * dx ** 2
time = 0.01
t_step = int(time / dt)


def elapse(u_cur):
    u_xx = (u_cur[2:] + u_cur[:-2] - 2 * u_cur[1:-1]) / (dx ** 2)
    u_xx = torch.cat((u_xx[0].unsqueeze(-1), u_xx, u_xx[-1].unsqueeze(-1)))
    u_t = alpha * u_xx
    u_next = u_cur + u_t * dt
    return u_next


x0 = torch.tensor([(i / x_step) for i in range(x_step + 1)])
u = init_func(x0)
output = []
for i in range(int(time / dt) + 1):
    u = elapse(u)
    output.append(u.unsqueeze(0))
output_mesh = torch.cat(output)
output = output_mesh.reshape(-1, 1).to(device)
