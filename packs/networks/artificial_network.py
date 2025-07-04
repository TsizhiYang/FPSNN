import torch
import torch.nn as nn


device = torch.device("cuda")


class ANN(nn.Module):

    def __init__(self, init_method="undef"):
        super(ANN, self).__init__()
        self.linear0 = nn.Linear(2, 16)
        self.linear1 = nn.Linear(16, 8)
        self.linear2 = nn.Linear(8, 1)

        if init_method == "undef":
            pass
        elif init_method == "xavier":
            nn.init.xavier_uniform_(self.linear0.weight)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, input):
        output = input
        output = self.linear0(output)
        output = torch.sigmoid(output)
        output = self.linear1(output)
        output = torch.sigmoid(output)
        output = self.linear2(output)
        return output
