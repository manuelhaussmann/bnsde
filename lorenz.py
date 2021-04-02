import os
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader


class LorenzPrior(nn.Module):
    def __init__(self, param_known=None, prior_std=1., diffusion=1., learn_params=False):
        super(LorenzPrior, self).__init__()

        if param_known is None:
            param_known = [False, False, False]
        self.param_known = param_known
        self.prior_std = prior_std
        self.diffusion = diffusion
        self.zeta_mu = Parameter(10 + self.prior_std * th.randn(1), requires_grad=learn_params)  # TRUE: 10
        self.kappa_mu = Parameter(28 + self.prior_std * th.randn(1), requires_grad=learn_params)  # TRUE: 28
        self.rho_mu = Parameter(8 / 3 + self.prior_std * th.randn(1), requires_grad=learn_params)  # TRUE: 8/3

    def drift_func(self, data):
        if data.dim() == 1:
            data = data.unsqueeze(0)

        zeta_k = self.param_known[0] * (self.zeta_mu + self.diffusion * th.randn_like(self.zeta_mu))
        kappa_k = self.param_known[1] * (self.kappa_mu + self.diffusion * th.randn_like(self.kappa_mu))
        rho_k = self.param_known[2] * (self.rho_mu + self.diffusion * th.randn_like(self.rho_mu))

        x, y, z = data[:, 0, None], data[:, 1, None], data[:, 2, None]
        x_dot = self.param_known[0] * (zeta_k * (y - x))
        y_dot = self.param_known[1] * (x * (kappa_k - z) - y)
        z_dot = self.param_known[2] * (x * y - z * rho_k)

        out = th.cat([x_dot, y_dot, z_dot], 1)
        return out

    def forward(self, data):
        return self.drift_func(data)


# The Lorenz Attractor data set
class LorenzDataSet:
    def __init__(self, train_length, test_length, N_train, N_test, gen_data=False):
        super(LorenzDataSet, self).__init__()

        if gen_data or not os.path.isfile("lorenz.npy"):
            print("Generate Lorenz Attractor data")
            generate_stochastic_lorenz()
        self.data = np.load("lorenz.npy")

        self.name = "Lorenz"
        self.n_dim = self.data.shape[1] - 1

        self.data_train = th.from_numpy(np.reshape(self.data[:N_train * train_length], [N_train, train_length, 4])).type(th.float32)
        self.train_loader = DataLoader(TensorDataset(self.data_train), batch_size=4, shuffle=True)
        self.data_test = th.from_numpy(
            np.reshape(self.data[N_train * train_length:N_train * train_length + N_test * test_length],
                       [N_test, test_length, 4])).type(th.float32)
        self.test_loader = DataLoader(TensorDataset(self.data_test), batch_size=1, shuffle=False)
        self.N_train = N_train


# Generate the Lorenz data
def generate_stochastic_lorenz():
    def x_dot_lorenz_true(x, t):
        zeta = 10
        kappa = 28
        rho = 8 / 3

        x, y, z = x[0], x[1], x[2]
        x_dot = zeta * (y - x)
        y_dot = x * (kappa - z) - y
        z_dot = x * y - z * rho
        return np.array([x_dot, y_dot, z_dot])

    # Diffusion matrix
    G = np.eye(3)

    dt = .0001
    x = np.array([1, 1, 28])
    t = np.arange(0, 100, dt)

    sol = np.zeros((len(t), 3))

    for i, tstep in enumerate(t):
        sol[i] = x
        x = x + x_dot_lorenz_true(x, tstep) * dt + G.dot(np.random.randn(3, 1)).flatten() * np.sqrt(dt)

    observations = np.concatenate((sol, t.reshape(-1, 1)), 1)

    observations = observations[0::100]

    print(observations.shape)

    np.save("lorenz", observations)

    return observations
