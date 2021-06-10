from abc import ABC
from models.kernel import *

class PIC(nn.Module, ABC):  # PIC approximation of exact GP
    def __init__(self, n_dim, n_inducing, Z = None, scales = None, device = device):
        super(PIC, self).__init__()
        self.device, self.n_dim, self.n_inducing = device, n_dim, n_inducing
        self.cov = CovFunction(self.n_dim, c = torch.FloatTensor([np.log(1.0)]).to(self.device), scales = scales).to(self.device)
        self.lik = LikFunction().to(self.device)
        if Z is not None:
            self.n_inducing = Z.shape[0]
            self.Xu = nn.Parameter(Z, requires_grad = True)
        else:
            self.Xu = nn.Parameter(torch.FloatTensor(np.random.randn(self.n_inducing, self.n_dim)),
                                  requires_grad = True).to(self.device)
        self.mean, self.var = None, None  # caching inducing mean and variance
        self.center = None  # caching block center

    def sample(self, u, S, n_sample = 100):  # S is n_batch by n_dim; and u is n_inducing by n_dim
        Ksu, Kss, Kuu_inv = self.cov(S, self.Xu), self.cov(S), torch.inverse(self.cov(self.Xu))
        mu, var = torch.mm(Ksu, torch.mm(Kuu_inv, u)), Kss - torch.mm(Ksu, torch.mm(Kuu_inv, Ksu.T)) + 0.01 * torch.eye(Kss.shape[0]).to(self.device)
        L = torch.cholesky(var, upper = False)  # L is n_batch by n_batch; mu is n_batch by 1
        z = torch.empty(n_sample, S.shape[0], 1).normal_(0, 1).to(self.device)  # z is n_sample by n_batch by 1
        return mu + L @ z  # output is n_sample by n_batch by 1 -- sampling result from p(w | u)

    def ELBO(self, X, Y):
        Kuu, noise_inv = self.cov(self.Xu), torch.exp(-2.0 * self.lik.noise)
        Guu, Kuu_inv = Kuu, torch.inverse(Kuu)
        res, wing, core = torch.FloatTensor([[0.0]]).to(self.device), torch.zeros(1, self.n_inducing).to(self.device), \
                          torch.FloatTensor([[0.0]]).to(self.device)
        for i in range(len(X)):
            Ksu, Kss = self.cov(X[i], self.Xu), self.cov(X[i], X[i])
            Guu += noise_inv * torch.mm(Ksu.T, Ksu)
            res += noise_inv * torch.trace(Kss - torch.mm(Ksu, torch.mm(Kuu_inv, Ksu.T)))  # adding trace term
            wing += noise_inv * torch.mm(Y[i].T, Ksu)
            core += noise_inv * torch.mm(Y[i].T, Y[i])
            res += 2 * X[i].shape[0] * self.lik.noise
        res += torch.logdet(Guu) - torch.logdet(Kuu)  # compute log det term
        res += (core - torch.mm(wing, torch.mm(torch.inverse(Guu), wing.T)))  # compute quad term
        return 0.5 * res  # this is the negative ELBO -- minimizing it is the same as maximizing ELBO

    def init(self, X):
        self.center = [torch.mean(Xi, dim=0).view(1, -1) for Xi in X]

    def prior(self, n_sample = None):
        if n_sample is None:
            return torch.zeros(self.n_inducing, 1).to(self.device), self.cov(self.Xu)
        L = torch.cholesky(self.cov(self.Xu), upper = False).to(self.device)
        eps = torch.empty(n_sample, self.n_inducing, 1).normal_(0, 1).to(self.device)
        return L @ eps  # n_sample by n_inducing by 1 -- sampling result u ~ p(u) = N(u; 0, K_uu)

    def cache(self, X, Y):
        self.init(X)
        Kuu = self.cov(self.Xu)
        Kuu_inv = torch.inverse(Kuu)
        phi = torch.zeros(self.n_inducing, self.n_inducing).to(self.device)
        self.mean = torch.zeros(self.n_inducing, 1).to(self.device)
        for i in range(len(X)):
            Ksu, Kss = self.cov(X[i], self.Xu), self.cov(X[i])
            Gss_inv = torch.inverse(Kss - torch.mm(torch.mm(Ksu, Kuu_inv), Ksu.T) +
                                    torch.eye(Kss.shape[0]).to(self.device) * torch.exp(2 * self.lik.noise))
            phi += torch.mm(Ksu.T, torch.mm(Gss_inv, Ksu))
            self.mean += torch.mm(Ksu.T, torch.mm(Gss_inv, Y[i]))
        phi = torch.inverse(phi + Kuu)
        self.mean, self.var = torch.mm(Kuu, torch.mm(phi, self.mean)), torch.mm(Kuu, torch.mm(phi, Kuu))  # q(u)

    def map(self, x_ast):
        dist = [torch.sum((x_ast - self.center[i]) ** 2).item() for i in range(len(self.center))]
        return np.argmin(dist)  # this was previously argmax !!! my bad

    def predict(self, x_ast, X, Y, compute_var = False, use_cache = True, idx = None):  # expect X and Y are lists of tensors (n_i, n_dim)
        with torch.no_grad():
            assert x_ast.shape[0] == 1, 'Expecting input as row vector'
            if use_cache is False or self.mean is None:
                self.cache(X, Y)
            if idx is None:
                i = self.map(x_ast)
            else:
                i = idx

            P = self.cov(torch.cat([self.Xu, X[i]], dim = 0))
            P[self.n_inducing:, self.n_inducing:] += torch.eye(X[i].shape[0]).to(self.device) * torch.exp(2 * self.lik.noise)
            P = torch.inverse(P)

            Ksu, Ksi = self.cov(x_ast, self.Xu), self.cov(x_ast, X[i])
            B = torch.mm(Ksu, P[:self.n_inducing, :self.n_inducing]) + torch.mm(Ksi, P[self.n_inducing:, :self.n_inducing])
            L = torch.mm(Ksu, P[:self.n_inducing, self.n_inducing:]) + torch.mm(Ksi, P[self.n_inducing:, self.n_inducing:])
            mean = torch.mm(B, self.mean) + torch.mm(L, Y[i])
            if compute_var is False:
                return mean
            V = torch.mm(Ksu, torch.mm(P[:self.n_inducing, :self.n_inducing], Ksu.T) +
                         torch.mm(P[:self.n_inducing, self.n_inducing:], Ksi.T)) + \
                torch.mm(Ksi, torch.mm(P[self.n_inducing:, :self.n_inducing], Ksu.T) +
                         torch.mm(P[self.n_inducing:, self.n_inducing:], Ksi.T))
            var = V + torch.mm(B, torch.mm(self.var, B.T))
            return mean, var

    def forward(self, x_ast, X, Y, grad = False, compute_var = False, use_cache = True, idx = None):
        if grad is False:
            return self.predict(x_ast, X, Y, compute_var = compute_var, use_cache = use_cache, idx = idx)

        assert x_ast.shape[0] == 1, 'Expecting input as row vector'
        if use_cache is False or self.mean is None:
            self.efficient_cache(X, Y)
        if idx is None:
            i = self.map(x_ast)
        else:
            i = idx

        P = self.cov(torch.cat([self.Xu, X[i]], dim = 0))
        P[self.n_inducing:, self.n_inducing:] += torch.eye(X[i].shape[0]).to(self.device) * torch.exp(2 * self.lik.noise)
        P = torch.inverse(P)

        Ksu, Ksi = self.cov(x_ast, self.Xu), self.cov(x_ast, X[i])
        B = torch.mm(Ksu, P[:self.n_inducing, :self.n_inducing]) + torch.mm(Ksi, P[self.n_inducing:, :self.n_inducing])
        L = torch.mm(Ksu, P[:self.n_inducing, self.n_inducing:]) + torch.mm(Ksi, P[self.n_inducing:, self.n_inducing:])
        mean = torch.mm(B, self.mean) + torch.mm(L, Y[i])
        if compute_var is False:
            return mean
        V = torch.mm(Ksu, torch.mm(P[:self.n_inducing, :self.n_inducing], Ksu.T) +
                     torch.mm(P[:self.n_inducing, self.n_inducing:], Ksi.T)) + \
            torch.mm(Ksi, torch.mm(P[self.n_inducing:, :self.n_inducing], Ksu.T) +
                     torch.mm(P[self.n_inducing:, self.n_inducing:], Ksi.T))
        var = V + torch.mm(B, torch.mm(self.var, B.T))
        return mean, var

def PIC_test(predictor, X, Y, X_test, Y_test):
    predictor.cache(X, Y)
    Y_pred = torch.zeros(len(Y_test))
    for i in range(len(X_test)):
        X_ast, Y_ast = torch.FloatTensor(X_test[i, :].reshape(1, -1)).to(device), Y_test[i]
        Y_pred[i] = predictor.predict(X_ast, X, Y, use_cache = True)
    Y_pred = Y_pred.detach().numpy()
    return np.sqrt(np.mean((Y_pred - Y_test) ** 2))



