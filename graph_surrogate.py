
import torch
import numpy as np
import torch.nn as nn


device = torch.device("cuda")

LIF_thresh = 0.5
decay = 0.4
lens = 1

class ActFun_rectangle(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.gt(input, 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (1 / lens) * (ActFun_rectangle.apply(input + lens / 2) - ActFun_rectangle.apply(input - lens / 2))
    

class ActFun_sigmoid(torch.autograd.Function):
    alpha = 5.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.gt(input, 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * ActFun_sigmoid.alpha * torch.sigmoid(input * ActFun_sigmoid.alpha) * (1 - torch.sigmoid(input * ActFun_sigmoid.alpha))
    

import matplotlib.pyplot as plt
config = {
    "mathtext.fontset": 'stix',
}
plt.rcParams.update(config)
plt.rcParams['font.sans-serif'] = "Times New Roman"


plt.figure(figsize=(6, 4))

# Prepare input range
x = torch.linspace(-1.4, 1.4, 300, device=device, requires_grad=True)
x_np = x.detach().cpu().numpy()

# Rectangle surrogate gradient
y_rect = ActFun_rectangle.apply(x)
y_rect.sum().backward(retain_graph=True)
grad_rect = x.grad.detach().cpu().numpy()
x.grad.zero_()
plt.plot(x_np, grad_rect, color='green',label='APSNN 1')
plt.fill_between(x_np, grad_rect, 0, color='green', alpha=0.1)

# Sigmoid surrogate gradient
y_sigmoid = ActFun_sigmoid.apply(x)
y_sigmoid.sum().backward()
grad_sigmoid = x.grad.detach().cpu().numpy()
plt.plot(x_np, grad_sigmoid, color='mediumseagreen', label='APSNN 2')
plt.fill_between(x_np, grad_sigmoid, 0, color='mediumseagreen', alpha=0.1)

# Delta gradient
plt.plot(x_np, torch.zeros_like(x).cpu().numpy(), 'r-', label='$\dot{s}(u)$')
plt.plot([0, 0], [0, 1.8], 'r-', linewidth=1)
plt.text(0.05, 1.5, r'$+\infty$', fontsize=14, color='k')

# Plot the activation functions with dashed lines
y_rect_act = ActFun_rectangle.apply(x).detach().cpu().numpy()
plt.plot(x_np, y_rect_act, 'b--', label='$s(u)$')

# Plot
plt.xlabel('$u$')
plt.xlim(-1.5, 1.5)
plt.ylim(-0.3, 1.7)
plt.xticks([-1.0, -0.5, 0, 0.5, 1.0])
plt.yticks([0, 0.5, 1.0, 1.5])
# plt.title('Surrogate Gradients')
plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()
order = [3, 2, 0, 1]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
# plt.grid(True)

figname = "fig-surrogates" + ".svg"
plt.savefig(figname, dpi=300, format="svg")
plt.show()
print("image saved as", figname)

