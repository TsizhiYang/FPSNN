from ..initial_function.burgers2d_init import *



device = torch.device("cuda")

dx = 1 / x_step
dy = 1 / y_step
t_step = 40
time = 0.0004
dt = time / t_step


def elapse(u_cur):
    u_x = (u_cur[2:, :] - u_cur[:-2, :]) / (2 * dx)
    u_x = torch.cat((torch.zeros([1, y_step + 1]), u_x, torch.zeros([1, y_step + 1])), dim=0)
    u_xx = (u_cur[2:, :] + u_cur[:-2, :] - 2 * u_cur[1:-1, :]) / (dx ** 2)
    u_xx = torch.cat((torch.zeros([1, y_step + 1]), u_xx, torch.zeros([1, y_step + 1])), dim=0)
    u_y = (u_cur[:, 2:] - u_cur[:, :-2]) / (2 * dy)
    u_y = torch.cat((torch.zeros([x_step + 1, 1]), u_y, torch.zeros([x_step + 1, 1])), dim=1)
    u_yy = (u_cur[:, 2:] + u_cur[:, :-2] - 2 * u_cur[:, 1:-1]) / (dy ** 2)
    u_yy = torch.cat((torch.zeros([x_step + 1, 1]), u_yy, torch.zeros([x_step + 1, 1])), dim=1)
    u_t = u_xx + u_yy - u * u_x - u * u_y
    u_next = u_t * dt + u_cur
    return u_next


x0 = torch.linspace(0, 1, x_step + 1)
y0 = torch.linspace(0, 1, y_step + 1)
input0 = torch.meshgrid(x0, y0, indexing='ij')

u = init_func(input0)
output = [u.unsqueeze(0)]
for i in range(t_step):
    u = elapse(u)
    output.append(u.unsqueeze(0))
output_mesh = torch.cat(output)
output = output_mesh.reshape(-1, 1).to(device)
