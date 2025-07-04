import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['figure.figsize']=(5, 3.5)

graph_type = "heat"

if graph_type == "heat":
    from packs.initial_function.heat_init import *
    from packs.reference_generation.heat_ref import *
elif graph_type == "wave":
    from packs.initial_function.wave_init import *
    from packs.reference_generation.wave_ref import *


x = torch.linspace(0, 1, x_step + 1)
t = torch.linspace(0, time, t_step + 1)
x = torch.stack([x for _ in range(t_step + 1)]).unsqueeze(-1)
t = torch.stack([t for _ in range(x_step + 1)]).unsqueeze(-1)
input_mesh = torch.cat((t.permute(1, 0, 2), x), dim=2)

fig = plt.figure(1)
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
# t_show = int(t_step / t_axis_lines)
t_show = 1
for i in range(output_mesh.shape[0]):
    if i % t_show == 0:
        ax1.plot3D(input_mesh[i, :, 0], input_mesh[i, :, 1], output_mesh[i], c='red', alpha=0.2)
        plt.xlabel("t")
        plt.ylabel("x")

figname = graph_type + "ref" + ".svg"
plt.savefig(figname, dpi=300, format="svg")
plt.show()
