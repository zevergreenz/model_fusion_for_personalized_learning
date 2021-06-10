from abc import ABC
from utilities.util import *

class InferenceNet(nn.Module, ABC):
    def __init__(self, input_dim, output_dim, hidden_dim = 32, norm = False):
        super(InferenceNet, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.f = nn.Linear(input_dim, hidden_dim)
        self.g_mean = nn.Linear(hidden_dim, output_dim)
        self.g_var = nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.norm = norm

    def forward(self, x, n_sample = 100):  # x is batch_size by input_dim
        x = self.f(x)  # x is now batch_size by hidden_dim
        if (x.shape[0] > 1) and (self.norm is True):
            x = F.relu(self.bn(x))  # x remains batch_size by hidden_dim
        else:
            x = F.relu(x)  # x remains batch_size by hidden_dim
        mu, log_var = self.g_mean(x), self.g_var(x)  # mu and log_var are batch_size by output_dim
        std = torch.exp(0.5 * log_var)  # std is batch_size by output_dim
        if n_sample is None:
            return mu, log_var  # generative model is assumed to factorize across input components (hence, log_var is a diag matrix)
        eps = torch.empty(n_sample, x.shape[0], self.output_dim).normal_(0, 1).to(device)
        return mu + eps * std, mu, log_var  # sample size is n_sample by batch_size by output_dim
