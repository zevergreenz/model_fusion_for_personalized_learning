from abc import ABC
from models.kernel import *

class FGP(nn.Module, ABC):  # Exact GP
    def __init__(self, n_dim, scales = None):
        super(FGP, self).__init__()
        self.device = device
        self.n_dim = n_dim
        self.cov = CovFunction(self.n_dim, scales = scales).to(self.device)
        self.mean = MeanFunction().to(self.device)
        self.lik = LikFunction().to(self.device)

    def NLL(self, X, Y):
        Yc = Y.float() - self.mean.mean.float()
        Kxx = self.cov(X)
        torch.diagonal(Kxx).fill_(torch.exp(2.0 * self.cov.sn) + torch.exp(2.0 * self.lik.noise))
        L = torch.cholesky(Kxx, upper = False)
        Linv = torch.mm(torch.inverse(L), Yc)
        res = 0.5 * torch.sum(torch.log(L.diag())) + 0.5 * torch.mm(Linv.t(), Linv)
        return res

    def predict(self, Xt, X, Y, var = False):
        with torch.no_grad():
            Yc = Y.float() - self.mean.mean.float()
            Ktx = self.cov(Xt, X)
            Kxx = self.cov(X)
            Q_inv = torch.inverse(Kxx + torch.exp(2.0 * self.lik.noise) * torch.eye(X.shape[0]).to(self.device)).float()
            Yt = torch.mm(Ktx, torch.mm(Q_inv, Yc)) + self.mean.mean.float()
            if var is False:
                return Yt
            Ktt = self.cov(Xt)
            Vt = Ktt - torch.mm(Ktx, torch.mm(Q_inv, Ktx.t()))
            return Yt, Vt

    def forward(self, Xt, X, Y, grad = False, var = False):
        if grad is False:
            return self.predict(Xt, X, Y, var = var)
        Yc = Y.float() - self.mean.mean.float()
        Ktx = self.cov(Xt, X)
        Kxx = self.cov(X)
        Q_inv = torch.inverse(Kxx + torch.exp(2.0 * self.lik.noise) * torch.eye(X.shape[0]).to(self.device)).float()
        Yt = torch.mm(Ktx, torch.mm(Q_inv, Yc)) + self.mean.mean.float()
        if var is False:
            return Yt
        Ktt = self.cov(Xt)
        Vt = Ktt - torch.mm(Ktx, torch.mm(Q_inv, Ktx.t()))
        return Yt, Vt