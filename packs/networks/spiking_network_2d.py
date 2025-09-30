import torch
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
        return grad_output * ActFun_sigmoid.alpha * torch.sigmoid(input * ActFun_sigmoid.alpha) * (1 - torch.sigmoid(input * ActFun_sigmoid.alpha)) * (1 / lens)


class SNN(nn.Module):

    def __init__(self, time_window=40,
                 in_scale=torch.tensor([[0., 1.]]),
                 in_shape_as=torch.tensor([0.]),
                 out_dim=1,
                 init_method="undef",
                 actfun="rectangle"):

        super(SNN, self).__init__()

        in_scale = in_scale.to(device)
        self.time_window = time_window
        self.in_sample = in_shape_as.to(device)                 # one sample input to read input length

        encode_len = in_shape_as.reshape(-1).shape[0]
        self.linear = nn.Linear(encode_len, 16)                 # LIF fc layer, input_len -> 16
        if actfun == "rectangle":
            self.ActFun = ActFun_rectangle
        elif actfun == "sigmoid":
            self.ActFun = ActFun_sigmoid

        def normalize(x):
            return (x - in_scale[:, 0]) / (in_scale[:, 1] - in_scale[:, 0])
        self.scaler = normalize                                 # affine normalization

        def encode(x, time_step):
            x = x.reshape(x.shape[0], -1)  # flatten
            thresh = (time_step + 1) / (time_window + 1)
            return self.ActFun.apply(x - thresh)
        self.encoder = encode                                   # lower triangular encoder

        self.decoder1 = nn.Linear(16, 8)                 # decode fc layer 1, 16 -> 8
        self.decoder2 = nn.Linear(8, out_dim)            # decode fc layer 2, 8 -> out_dim

        if init_method == "undef":
            pass
        elif init_method == "xavier":
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.xavier_uniform_(self.decoder1.weight)
            nn.init.xavier_uniform_(self.decoder2.weight)

        self.spike_recorder = None

    def forward(self, raw_input):

        input = self.scaler(raw_input)

        mem = 0
        spike = 0
        out_spike = 0

        spike_recorder = []

        for time_step in range(self.time_window):

            encoded = self.encoder(input, time_step)
            mem, spike = self.mem_update(self.linear, encoded, mem, spike)
            spike_recorder.append(spike)
            out_spike = out_spike + spike

        self.spike_recorder = torch.stack(spike_recorder)

        out_spike = out_spike / self.time_window
        res = self.decoder1(out_spike)
        res = torch.sigmoid(res)
        res = self.decoder2(res)

        return res

    def mem_update(self, operation, pre_spike, memory, spiked, decay=decay):

        memory = memory * decay * (1. - spiked) + operation(pre_spike)
        spike = self.ActFun.apply(memory - LIF_thresh)
        return memory, spike                                   # LIF node membrane potential and spike update

    def spike_rate(self):

        print(self.spike_recorder.shape)
        num_total = self.spike_recorder.numel()
        num_spike = torch.count_nonzero(self.spike_recorder)

        return num_spike / num_total

