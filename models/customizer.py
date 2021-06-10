from models.pic import *
from sklearn.cluster import KMeans
import torch.nn.functional as F

class Personalizer(nn.Module, ABC):
    def __init__(self, w_dim, a_dim, z_dim, h_dim = 32, dev = device):
        super(Personalizer, self).__init__()
        self.w_dim, self.a_dim, self.z_dim, self.h_dim = w_dim, a_dim, z_dim, h_dim
        self.dev = dev
        self.fw = nn.Linear(self.w_dim, self.a_dim).to(self.dev)
        self.fe = nn.Linear(self.a_dim, self.h_dim).to(self.dev)
        self.zm = nn.Linear(self.h_dim, self.z_dim).to(self.dev)
        self.zv = nn.Linear(self.h_dim, self.z_dim).to(self.dev)

    def sample(self, W, A, n_sample = 100):  # w is n_batch by w_dim; a is n_batch by a_dim; w is specific while a is base
        latent = F.relu(self.fe(F.relu(self.fw(W) + A)))  # latent is batch size by h_dim
        mu, log_var = self.zm(latent), self.zv(latent)  # mu and log_var are batch size by z_dim
        std = torch.exp(0.5 * log_var)  # std is batch_size by z_dim
        if n_sample is None:
            return mu, log_var  # generative model is assumed to factorize across input components (hence, log_var is a diag matrix)
        eps = torch.empty(n_sample, W.shape[0], self.z_dim).normal_(0, 1).to(self.dev)
        return mu + eps * std, mu, log_var  # sample size is n_sample by batch_size by output_dim

    def forward(self, Z, W, A, B = None):  # w is n_batch by w_dim; a is 1 by a_dim; z is n_batch by z_dim
        latent = F.relu(self.fe(F.relu(self.fw(W) + A)))  # latent is batch size by h_dim
        mu, log_var = self.zm(latent), F.tanh(self.zv(latent))  # mu and log_var are batch size by z_dim
        res = -0.5 * Z.shape[0] * Z.shape[1] * np.log(2 * np.pi) - torch.sum(log_var) \
              -0.5 * torch.sum(((Z - mu) * torch.exp(-1.0 * log_var)) ** 2)
        res /= (Z.shape[0] * Z.shape[1])
        return res, None  # compute log p(z | w, a)

class NewPersonalizer(nn.Module, ABC):
    def __init__(self, w_dim, a_dim, z_dim, h_dim = 32, dev=device):
        super(NewPersonalizer, self).__init__()
        self.w_dim, self.a_dim, self.z_dim, self.h_dim = w_dim, a_dim, z_dim, h_dim
        self.dev = dev
        self.fw = nn.Linear(self.w_dim + self.a_dim, 2 * self.h_dim).to(self.dev)
        self.fe = nn.Linear(2 * self.h_dim, self.h_dim).to(self.dev)
        self.zm = nn.Linear(self.h_dim, self.z_dim).to(self.dev)
        self.zv = nn.Linear(self.h_dim, self.z_dim).to(self.dev)

    def sample(self, W, A, n_sample = 100):  # w is n_batch by w_dim; a is n_batch by a_dim; w is specific while a is base
        latent = F.relu(self.fe(F.relu(self.fw(torch.cat([W, A], dim = -1)))))  # latent is batch size by h_dim
        mu, log_var = self.zm(latent).to(self.dev), F.tanh(self.zv(latent)).to(self.dev)  # mu and log_var are batch size by z_dim
        std = torch.exp(0.5 * log_var)  # std is batch_size by z_dim
        if n_sample is None:
            return mu, log_var  # generative model is assumed to factorize across input components (hence, log_var is a diag matrix)
        eps = torch.empty(n_sample, W.shape[0], self.z_dim).normal_(0, 1).to(self.dev)
        return mu + eps * std, mu, log_var  # sample size is n_sample by batch_size by output_dim

    # Idea: ELBO(B) = sum_t E_q [ log p(zt | wt, wo)] - KL(.||.)
    # where B[t][i][k] = 1 iff z[t][i] is matched to mu[t][k] and log_var[t][k] -- B is n_model by z_dim by z_dim
    # where mu[t] = mu_net(wt, wo) and log_var[t] = var_net(wt, wo) -- both is z_dim in dimension
    # "Matched" means z[t][i] ~ N(mu[t][k], diag[exp(2 * log_var[t][k])])
    #                         = N(sum_k B[t][i][k] * mu[t][k], diag(sum_k B[t][i][k] * exp(2 * log_var[t][k])))
    # Here, B[t][i][k] in {0, 1} and sum_k B[t][i][k] = 1

    # Alternating optimization: (a) Fix B, optimize q; (b) Fix q, optimize B
    # For (b), I need to represent ELBO(B) = sum_t sum_i sum_k B[t][i][k] * C[t][i][k] - KL(.||.)
    # Since p(zt | wt, wo) = prod_i p(zti | wt, wo) where
    # p(zti | wt, wo) = N(zti; sum_k B[t][i][k] * mu[t][k], diag[sum_k B[t][i][k] * exp(2 * log_var[t][k])])
    # This implies log p(zt | wt, wo) = const
    #              + sum_i (-0.5 * (zti - sum_k B[t][i][k] * mu[t][k])^2 / exp(2 * sum_k B[t][i][k] * log_var[t][k])
    #              - sum_k B[t][i][k] * log_var[t][k])
    # This means sum_t log p(zt | wt, wo) = const + sum_t sum_i sum_k B[t][i][k] * D[t][i][k]
    # where D[t][i][k] = -0.5 * (zti - mu[t][k])^2 / exp(2 * log_var[t][k])
    #                    - log_var[t][k]
    # Plugging these all in ELBO(B): ELBO(B) = const + sum_t sum_i sum_k B[t][i][k] * C[t][i][k] - KL(. || .)
    # where C[t][i][k] = E_q [ D[t][i][k] ]

    # this function computes sum_t log p(zt | wt, wo) where in the code: W means bunch of wt ~ q, A means wo ~ q
    def forward(self, Z, W, A, B = None):  # W is n_batch by w_dim; A is 1 by a_dim; Z is n_batch by z_dim
        # B is n_batch by z_dim by z_dim -- B[t][i][j] = 1 iff z[t][i] is matched to mu[t][j] and log_var[t][j]
        latent = F.relu(self.fe(F.relu(self.fw(torch.cat([W, A.repeat(W.shape[0], 1)], dim=-1)))))  # latent is batch size by h_dim
        mu, log_var = self.zm(latent), F.tanh(self.zv(latent))  # mu and log_var are batch size by z_dim
        if B is None:  # this means no alignment is needed -- by default B[t][i][j] = 1 iff i == j
            res = -0.5 * Z.shape[0] * Z.shape[1] * np.log(2 * np.pi) - torch.sum(log_var) \
                  - 0.5 * torch.sum(((Z - mu) * torch.exp(-1.0 * log_var)) ** 2)
            return res / (Z.shape[0] * Z.shape[1]), None  # compute log p(z | w, a)
        # if B is specified, we need to perform an alignment
        mu, log_var = mu.unsqueeze(2), log_var.unsqueeze(1)
        D = -0.5 * ((Z.unsqueeze(1) - mu) ** 2) * torch.exp(-2.0 * log_var) - log_var  # n_batch by z_dim by z_dim
        return torch.sum(B * D) / (Z.shape[0] * Z.shape[-1]), D.detach().cpu().numpy()

class WeightNet(nn.Module, ABC):  # handle p(w|u) and p(u)
    def __init__(self, S, w_dim, n_cluster = 10, n_pic = 5, dev = device):
        super(WeightNet, self).__init__()
        self.s_dim, self.w_dim, self.n_pic, self.device = S.shape[1], w_dim, n_pic, dev
        self.partition = KMeans(n_clusters = n_cluster, random_state = 0).fit(S)
        self.membership = self.partition.labels_
        self.Z = torch.tensor(self.partition.cluster_centers_, dtype = torch.float, device = self.device)
        self.S_cluster = torch.tensor(self.partition.predict(S), dtype = torch.float, device = self.device)
        self.pic = [PIC(self.s_dim, n_cluster, Z = self.Z, device = self.device) for _ in range(n_pic)]
        for pic in self.pic:
            pic.center = [self.Z[i].clone().view(1, -1) for i in range(self.Z.shape[0])]
        self.warper = nn.Sequential(nn.Linear(n_pic, 256), nn.Tanh(), nn.Linear(256, w_dim)).to(device)

    def sample(self, u, S, n_sample = 100):  # S is n_batch by s_dim; and u is n_inducing by 1
        core = torch.stack([pic.sample(u, S, n_sample = n_sample) for pic in self.pic])  # core is n_pic by n_sample by n_batch by 1
        core_permute = core.permute(1, 2, 3, 0)  # n_sample by n_batch by 1 by n_pic
        return self.warper(core_permute).view(n_sample, S.shape[0], self.w_dim)  # n_sample by n_batch by 1 by w_dim

    def sample_core(self, u, S, n_sample = 100):
        return torch.cat([pic.sample(u, S, n_sample = n_sample) for pic in self.pic], dim = 2)

    def prior(self, n_sample = 100):
        return self.pic[0].prior(n_sample = n_sample)  # return u ~ N(0, Kuu)  -- n_sample by n_inducing by 1

    def forward(self, S_ast, S_group, core_group, use_cache = False):  # S_ast is 1 by s_dim
        # core_group has n_group elements. Each is group_size[i] by n_pic
        core_ast = []
        for t in range(self.n_pic):
            Y_group = [core_group[i][:, t].view(-1, 1) for i in range(len(S_group))]
            core_ast.append(self.pic[t].predict(S_ast, S_group, Y_group, use_cache = use_cache))
        core = torch.stack(core_ast).view(1, -1).to(device)
        return self.warper(core)

def compute_mean(Z, B = None):
    if type(Z) is list:
        z_mean, n_data = torch.zeros(1, Z[0].shape[1]).to(device), 0.
        for i in range(len(Z)):
            if B is None:
                z_mean += torch.sum(Z[i], dim = 0, keepdim = True)
            else:
                z_mean += torch.sum(Z[i].unsqueeze(1) @ B[i])
            n_data += Z[i].shape[0]
        return z_mean / n_data
    if B is None:
        return torch.mean(Z, dim = 0, keepdim = True)
    return torch.mean(Z.unsqueeze(1).to(device) @ B.to(device), dim = 0)

class BaseNet(nn.Module, ABC):  # q(theta | z1, z2, ..., zn)
    def __init__(self, a_dim, z_dim, dev = device):
        super(BaseNet, self).__init__()
        self.a_dim, self.z_dim = a_dim, z_dim
        self.device = dev
        self.fm = nn.Linear(z_dim, a_dim).to(self.device)
        self.fv = nn.Linear(z_dim, a_dim).to(self.device)
        self.bn = nn.BatchNorm1d(a_dim).to(self.device)

    # for each sample, KL(q(wo | Z) || N(0, I)) where wo is 1 by a_dim or a_dim by 1
    # q(wo | Z) = N(wo; mu, diag(exp(log_var)))
    # KL = 0.5 * (sum(log_var) + sum(exp(log_var))] + mu^Tmu - a_dim)

    def forward(self, Z, n_sample = 100, B = None):  # Z is n_data by z_dim
        z_mean = compute_mean(Z, B = B)
        mu = self.fm(z_mean).to(self.device)  # 1 by a_dim
        log_var = self.bn(F.tanh(self.fv(z_mean))).to(self.device)  # 1 by a_dim
        if n_sample is None:
            return mu, log_var
        std = torch.exp(0.5 * log_var).to(self.device)  # std is 1 by a_dim
        eps = torch.empty(n_sample, 1, self.a_dim).normal_(0, 1).to(self.device)  # n_sample by 1 by a_dim
        return mu + eps * std, mu, log_var  # sample and compute KL(q(wo | Z) || N(0, I))








