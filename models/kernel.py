from abc import ABC
from utilities.util import *
import torch.nn as nn

class CovFunction(nn.Module, ABC):
    def __init__(self, n_dim, c = None, scales = None, sig_opt = False, scale_opt = True):
        super(CovFunction, self).__init__()
        self.n_dim, self.device = n_dim, device
        if scales is None:  # all parameters are assumed to be stored in log form
            self.weights = nn.Parameter(torch.FloatTensor(np.zeros((self.n_dim, 1))).to(self.device),
                                        requires_grad = scale_opt)
        else:
            self.weights = nn.Parameter(scales, requires_grad = scale_opt)
        if c is None:
            self.sn = nn.Parameter(torch.FloatTensor([0.0]).to(self.device), requires_grad = sig_opt)
        else:
            self.sn = nn.Parameter(c, requires_grad = sig_opt)

    def forward(self, U, V = None):  # U is of size _ by n_dim, V is of size _ by n_dim
        if V is None:
            V = U
        scales = torch.exp(-1.0 * self.weights).float().view(1, -1)
        a = torch.sum((U * scales) ** 2, 1).reshape(-1, 1)
        b = torch.sum((V * scales) ** 2, 1) - 2 * torch.mm((U * scales), (V * scales).t())
        res = torch.exp(2.0 * self.sn) * torch.exp(-0.5 * (a.float() + b.float()))
        return res

class MeanFunction(nn.Module, ABC):  # simply a constant mean function
    def __init__(self, c = None, mean_opt = True):
        super(MeanFunction, self).__init__()
        self.device = device
        if c is None:
            self.mean = nn.Parameter(torch.FloatTensor([0.0]).to(self.device), requires_grad = mean_opt)
        else:
            self.mean = nn.Parameter(c, requires_grad = mean_opt)

    def forward(self, U):  # simply the constant mean function; input form: (_, x_dim)
        return torch.ones(U.size(0), 1).to(self.device) * self.mean

class LikFunction(nn.Module, ABC):  # return log likelihood of a Gaussian N(z | x, noise^2 * I)
    def __init__(self, c = None, noise_opt = True):
        super(LikFunction, self).__init__()
        self.device = device
        if c is None:
            self.noise = nn.Parameter(torch.FloatTensor([0.0]).to(self.device), requires_grad = noise_opt)
        else:
            self.noise = nn.Parameter(c, requires_grad = noise_opt)

    def forward(self, o, x):
        diff = o - x
        n, d = o.size(0), o.size(1)
        res = -0.5 * n * d * torch.log(torch.tensor(2 * np.pi).to(self.device)) - 0.5 * n * self.noise - \
              0.5 * torch.exp(-2.0 * self.noise) * torch.trace(torch.mm(diff.t(), diff))
        return res

