from ..initial_function.wave_init import *

device = torch.device("cuda")

c = 1
t_step = 40
dx = 1 / x_step
dt = 1 / t_step
time = t_step * dt


def elapse(u_pre, u_cur):
    u_xx = (u_cur[2:] + u_cur[:-2] - 2 * u_cur[1:-1]) / (dx ** 2)
    u_xx = torch.cat((torch.zeros(1), u_xx, torch.zeros(1)))
    u_tt = c * u_xx
    u_next = u_tt * (dt ** 2) + 2 * u_cur - u_pre
    return u_cur, u_next


x0 = torch.tensor([(i / x_step) for i in range(x_step + 1)])
u = init_func(x0)
pu = u = torch.cat((torch.zeros(1), u[1:-1], torch.zeros(1)))
output = [u.unsqueeze(0)]
for i in range(t_step):
    pu, u = elapse(pu, u)
    output.append(u.unsqueeze(0))
output_mesh = torch.cat(output)
output = output_mesh.reshape(-1, 1).to(device)
