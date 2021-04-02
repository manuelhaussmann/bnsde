import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# A mean-field fully connected layer using the local reparameterization trick
class ProbLinear(nn.Module):
    def __init__(self, n_in, n_out):
        super(ProbLinear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.bias = nn.Parameter(th.Tensor(n_out))
        self.mu_w = nn.Parameter(th.Tensor(n_out, n_in))
        self.logsig2_w = nn.Parameter(th.Tensor(n_out, n_in))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.normal_(-9, 0.001)
        self.bias.data.zero_()

    def forward(self, input):
        mu_out = F.linear(input, self.mu_w, self.bias)
        s2_w = self.logsig2_w.clamp(-11, 11).exp()
        var_out = F.linear(input.pow(2), s2_w) + 1e-8
        return mu_out + var_out.sqrt() * th.randn_like(mu_out)

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.n_in} -> {self.n_out})"


class BNN(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=100):
        super(BNN, self).__init__()

        self.dense1 = ProbLinear(n_in, n_hidden)
        self.dense2 = ProbLinear(n_hidden, n_hidden)
        self.dense3 = ProbLinear(n_hidden, n_hidden)
        self.dense4 = ProbLinear(n_hidden, n_out)

    def forward(self, input):
        out = F.softplus(self.dense1(input))
        out = F.softplus(self.dense2(out))
        out = F.softplus(self.dense3(out))
        out = self.dense4(out)
        return out
