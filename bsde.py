import math
import numpy as np
import torch as th
import torch.nn as nn

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


class BSDE(nn.Module):
    def __init__(self, drift_func, prior_process=None, lrate=1e-3, is_pac=False):
        # BSDE assuming the Lorenz attractor experiment of 3 spatial dimensions and a constant diffusion
        super(BSDE, self).__init__()

        self.drift_func = drift_func
        self.prior_process = prior_process
        self.is_pac = is_pac

        self.prior_prec = 0.1
        self.delta = 0.05
        self.dim = 3
        self.opt = th.optim.Adam(self.parameters(), lr=lrate)

    def KL(self, prior_prec, W_logsigma, W_mu):
        W_logsigma = W_logsigma.clamp(-11, 11)
        kl = 0.5 * (prior_prec * (W_mu.pow(2) + W_logsigma.exp()) - W_logsigma - 1 - np.log(prior_prec)).sum()
        return kl

    def euler_maruyama_step(self, x_k, dt, compute_kl=False):
        # x_k := [h_k, t_k]
        h_old = x_k[:self.dim]
        dW = th.randn_like(h_old)

        f_k = self.drift_func(x_k)
        # r_k includes the gamma parameter
        r_k = self.prior_process.drift_func(x_k)[0] if self.prior_process is not None else 0

        h_new = h_old +  (f_k + r_k) * dt + th.sqrt(dt) * dW

        if compute_kl:
            kl = 0.5 * th.sum(f_k*f_k)*dt
        else:
            kl = 0

        return th.cat((h_new, (x_k[-1] + dt)[None]), 0), kl

    def predict(self, trajectory, n_samples=1):
        # Compute marginal ll (up to additive constant) and kl for one trajectory of the batch
        y = th.zeros(n_samples, trajectory.size(0), self.dim).to(device)
        for s in range(n_samples):
            y[s,0] = trajectory[0, :self.dim]
            x = trajectory[0]
            for k in range(1, trajectory.size(0)):
                dt = trajectory[k,-1] - trajectory[k-1,-1]
                x, _ = self.euler_maruyama_step(x, dt, compute_kl=False)
                y[s, k] = x[:3]
        return y.mean(0)

    def integrate(self, trajectory, n_samples=1, compute_kl=False):
        # Compute marginal ll (up to additive constant) and kl for one trajectory of the batch
        kl = 0
        y = th.zeros(n_samples, trajectory.size(0), self.dim).to(device)
        for s in range(n_samples):
            y[s,0] = trajectory[0, :self.dim]
            x = trajectory[0]
            for k in range(1, trajectory.size(0)):
                dt = trajectory[k,-1] - trajectory[k-1,-1]
                x, klterm = self.euler_maruyama_step(x, dt, compute_kl)
                kl += klterm
                y[s, k] = x[:3]

        mll = -0.5*(y - trajectory[None, :, :self.dim]).pow(2).sum((2,1)).mean()
        return mll, kl/n_samples

    def step(self, batch, N, n_samples=1):
        # Implementation of Algorithm 1 (see https://arxiv.org/abs/2006.09914)
        total_mll = 0
        total_kl = 0
        # Compute and collect loss over the batch
        # Note that for demonstrative purposes trajectories are handled individually
        # this loop could (and should) be vectorized for real world applications for increased efficiency
        for trajectory in batch:
            mll, kl = self.integrate(trajectory, n_samples=n_samples, compute_kl=self.is_pac)
            total_mll += mll
            total_kl += kl

        # Add PAC part
        if self.is_pac:
            # KL over the BNN in the drift
            total_kl += sum([self.KL(self.prior_prec, l.W_logsigma, l.W_mu) for l in self.drift_func.modules() if hasattr(l, "W_mu")])
            loss = -total_mll + th.sqrt((total_kl + math.log(2*math.sqrt(N)/self.delta))/(2*N))
        else:
            loss = -total_mll

        # Update parameters
        loss.backward()
        self.opt.step()
        return loss.item()
