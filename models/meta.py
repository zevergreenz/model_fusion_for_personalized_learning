from models.customizer import *
from scipy.optimize import linear_sum_assignment
import tqdm

def KL(mu, log_var):  # mu and log_var: batch_size by x_dim -- assume p(z) = N(z; 0, I)
    b_size = mu.shape[0]
    mu = mu.view(-1, 1)
    log_var = log_var.view(-1, 1)
    res = 0.5 * torch.sum((mu ** 2)) - 0.5 * torch.sum(log_var) + 0.5 * torch.sum(torch.exp(log_var))
    return res * (1.0 / b_size)

class LambdaLayer(nn.Module):
    def __init__(self):
        super(LambdaLayer, self).__init__()
    def forward(self, x):
        return (x ** 2) ** 0.5 + 1e-4

class MetaExpert(nn.Module, ABC):
    def __init__(self, S, Z, z_dim, w_dim, a_dim, n_cluster = 10, dev = device):  # S is n_task by s_dim; Z is n_task by z_dim
        super(MetaExpert, self).__init__()
        self.s_dim, self.z_dim, self.w_dim, self.a_dim = S.shape[1], z_dim, w_dim, a_dim
        self.S, self.Z, self.n_cluster, self.device = S, Z, n_cluster, dev  # S: side information; Z: local model

        self.p_wu = WeightNet(S, w_dim, n_cluster = self.n_cluster, n_pic = 15, dev = self.device)
        self.p_z_wa = NewPersonalizer(w_dim, a_dim, z_dim, dev = self.device)
        self.q_theta = BaseNet(a_dim, z_dim, dev = self.device)

        self.Z_cluster = self.p_wu.partition.labels_
        self.S, self.Z = torch.FloatTensor(self.S).to(self.device), torch.FloatTensor(self.Z).to(self.device)
        self.S_group, self.Z_group = [], []

        for i in range(self.n_cluster):  # for each data block
            self.S_group.append(self.S[self.p_wu.S_cluster == i, :])
            self.Z_group.append(self.Z[self.Z_cluster == i, :])

        self.params = nn.ModuleList([self.p_wu, self.p_z_wa, self.q_theta])
        self.optimizer = optim.Adam(self.params.parameters(), weight_decay = 1e-5)
        # originally, we need Bg[t][i][i] = 1 and 0 otherwise
        self.B = [torch.eye(z_dim, z_dim).unsqueeze(0).repeat(len(self.Z_group[i]), 1, 1).to(device)
                  for i in range(self.n_cluster)]  # list of alignment tensors -- one per group
        self.C = None  # where to store the cost tensors -- one per group

    def L(self, Z, A, U, S, alignment = None):  # Z is n_batch by z_dim; U is n_sample by n_inducing by 1; A is n_sample by 1 by a_dim; S is n_batch by s_dim
        res, C = 0.0, 0.0
        if alignment is None:
            C = None
        W = torch.cat([self.p_wu.sample(U[i], S, n_sample = 1) for i in range(U.shape[0])], dim = 0)  # n_sample by n_batch by w_dim
        for i in range(W.shape[0]):  # W.shape[0] is the no. of samples (wt, wo)
            resi, Di = self.p_z_wa(Z, W[i], A[i], B = alignment)  # this needs to return the alignment cost matrix
            res += resi
            if C is not None:
                C += Di / W.shape[0]
        return res / W.shape[0], C

    def ELBO(self, n_sample = 10, do_alignment = False):  # Z is n_batch by z_dim
        self.params.eval()  # TODO: Change this (?)
        U, res = self.p_wu.prior(n_sample = n_sample), 0.0  # U is n_sample by n_inducing by 1
        if do_alignment:
            _, ma, va = self.q_theta(self.Z_group, n_sample = n_sample, B = self.B)  # A is n_sample by 1 by a_dim  # revert this
            self.C = []
        else:
            _, ma, va = self.q_theta(self.Z_group, n_sample = n_sample)
            self.C = None
        for i in range(self.n_cluster):  # for each data block
            if do_alignment:
                Ai, _, _ = self.q_theta(self.Z_group[i], n_sample = n_sample, B = self.B[i])
                resi, Ci = self.L(self.Z_group[i], Ai, U, self.S_group[i], alignment = self.B[i])  # this needs to return C = ave D
                self.C.append(Ci)
            else:
                Ai, _, _ = self.q_theta(self.Z_group[i], n_sample = n_sample)
                resi, _ = self.L(self.Z_group[i], Ai, U, self.S_group[i])
            res += resi
        res -= KL(ma, va)
        return -1.0 * res  # we want to maximize ELBO but the optimizer is a minimizer so we need to negate the ELBO

    def ELBO_scale(self, n_sample = 10, do_alignment = False):  # Z is n_batch by z_dim
        self.params.eval()
        U, res = self.p_wu.prior(n_sample = n_sample), 0.0  # U is n_sample by n_inducing by 1

        Z_group_scale = []
        for i in range(self.n_cluster):
            Z_group_scale.append(self.Z_group[i] * self.scale_net(self.S_group[i]))

        if do_alignment:
            _, ma, va = self.q_theta(Z_group_scale, n_sample = n_sample, B = self.B)  # A is n_sample by 1 by a_dim  # revert this
            self.C = []
        else:
            _, ma, va = self.q_theta(Z_group_scale, n_sample = n_sample)
            self.C = None
        for i in range(self.n_cluster):  # for each data block
            if do_alignment:
                Ai, _, _ = self.q_theta(Z_group_scale[i], n_sample = n_sample, B = self.B[i])
                resi, Ci = self.L(Z_group_scale[i], Ai, U, self.S_group[i], alignment = self.B[i])  # this needs to return C = ave D
                self.C.append(Ci)
            else:
                Ai, _, _ = self.q_theta(Z_group_scale[i], n_sample = n_sample)
                resi, _ = self.L(Z_group_scale[i], Ai, U, self.S_group[i])
            res += resi
        res -= KL(ma, va)
        return -1.0 * res  # we want to maximize ELBO but the optimizer is a minimizer so we need to negate the ELBO

    def model_align(self, pos, t):
        # negate the sign since we want to maximize the ELBO rather than minimize it
        row_ind, col_ind = linear_sum_assignment(-self.C[pos][t])
        self.B[pos][t] *= 0  # zero out
        for (u, v) in zip(row_ind, col_ind):
            self.B[pos][t][u, v] = 1

    def cluster_align(self, pos):  # re-compute B[pos] -- size(cluster[pos]) by z_dim by z_dim
        for t in range(self.Z_group[pos].shape[0]):  # for each model in this group
            self.model_align(pos, t)

    def fit(self, n_iter = 100, checkpoint = 10, do_alignment = False):
        print('Commence Meta Training ...')
        records = []
        for i in range(1, n_iter + 1):
            self.params.train()
            self.optimizer.zero_grad()
            loss = self.ELBO(do_alignment = do_alignment)  # ELBO(B) and the corresponding cost
            records.append(-loss.item())  # recording the ELBO per iteration
            loss.backward()
            self.optimizer.step()
            if do_alignment:
                for t in range(self.n_cluster):
                    self.cluster_align(t)
            if i % checkpoint == 0:
                print('Iteration {}: ELBO = {}'.format(i, records[-1]))
        print('Training Completed.')
        return records

    def forward_with_grad(self, S_ast, n_sample = 100, verbal = False, do_alignment = False):
        U = self.p_wu.prior(n_sample = n_sample)  # U is n_sample by n_inducing by 1
        idx = self.p_wu.pic[0].map(S_ast)
        if do_alignment:
            rotate = self.B[idx][0].t()
            A, _, _ = self.q_theta(self.Z_group[idx], n_sample = n_sample, B = self.B[idx])  # A is n_sample by 1 by a_dim
        else:
            A, _, _ = self.q_theta(self.Z_group[idx], n_sample = n_sample, B = None)
        core = []
        for i in range(self.n_cluster):
            core.append(torch.cat([self.p_wu.sample_core(U[p], self.S_group[i], n_sample = 1)
                                   for p in range(U.shape[0])], dim = 0))  # n_sample by group_size by n_pic
        res = torch.zeros(1, self.z_dim).to(self.device)
        for p in range(n_sample):
            core_group = [core[i][p] for i in range(self.n_cluster)]  # core[i][p] is group_size[i] by n_pic
            W_ast = self.p_wu(S_ast, self.S_group, core_group, use_cache = False)  # S_ast is 1 by s_dim; W_ast is 1 by w_dim
            Z_ast, _ = self.p_z_wa.sample(W_ast, A[p], n_sample = None)

            if do_alignment:
                Z_ast = torch.mm(rotate, Z_ast.t()).t()
            res += Z_ast.to(self.device)
        res /= n_sample
        return res

    def forward(self, S_ast, n_sample = 100, verbal = False, do_alignment = False, grad = False):  # output Z_ast
        if grad:
            return self.forward_with_grad(S_ast, n_sample = n_sample, verbal = False, do_alignment = do_alignment)
        with torch.no_grad():
            U = self.p_wu.prior(n_sample = n_sample)  # U is n_sample by n_inducing by 1
            idx = self.p_wu.pic[0].map(S_ast)
            if do_alignment:
                rotate = self.B[idx][0].t()
                A, _, _ = self.q_theta(self.Z_group[idx], n_sample = n_sample, B = self.B[idx])  # A is n_sample by 1 by a_dim
            else:
                A, _, _ = self.q_theta(self.Z_group[idx], n_sample = n_sample, B = None)
            core = []
            for i in range(self.n_cluster):
                core.append(torch.cat([self.p_wu.sample_core(U[p], self.S_group[i], n_sample = 1)
                                       for p in range(U.shape[0])], dim = 0))  # n_sample by group_size by n_pic
            res = torch.zeros(1, self.z_dim).to(device)
            for p in range(n_sample):
                core_group = [core[i][p] for i in range(self.n_cluster)]  # core[i][p] is group_size[i] by n_pic
                W_ast = self.p_wu(S_ast, self.S_group, core_group, use_cache = False)  # S_ast is 1 by s_dim; W_ast is 1 by w_dim
                Z_ast, _ = self.p_z_wa.sample(W_ast, A[p], n_sample = None)
                if do_alignment:
                    Z_ast = torch.mm(rotate, Z_ast.t()).t()
                res += Z_ast.to(self.device)
            res /= n_sample
            return res

    def refit(self, n_cluster, EXPERIMENT_NAME, do_alignment, distance, n_iter=100):
        checkpoint, size, sample_per_group = 1, 200, 10
        def sample_task(S_group, Z_group, sample_per_group=10):
            S_fit, Z_fit = [], []
            for i in range(len(S_group)):
                choices = np.random.choice(S_group[i].shape[0], min(sample_per_group, S_group[i].shape[0]))
                for u in choices:
                    S_fit.append(S_group[i][u].view(1, -1))
                    Z_fit.append(Z_group[i][u].view(1, -1))
            return torch.cat(S_fit, dim=0).to(device), torch.cat(Z_fit, dim=0).to(device)

        def sample_task_group(S_group, Z_group, sample_per_group=10):
            S_fit, Z_fit = [], []
            choices = np.random.choice(S_group.shape[0], min(sample_per_group, S_group.shape[0]))
            for u in choices:
                S_fit.append(S_group[u].view(1, -1))
                Z_fit.append(Z_group[u].view(1, -1))
            return torch.cat(S_fit, dim=0).to(device), torch.cat(Z_fit, dim=0).to(device)

        for c in range(n_cluster):
            print("Re-loading Legacy Model and Fitting for Cluster {}".format(c))
            meta_c = torch.load(META_MODEl_FOLDER + EXPERIMENT_NAME + "-legacy.pt", map_location=device)
            optimizer = optim.Adam(meta_c.parameters(), weight_decay=1e-5)
            for iter in range(1, n_iter + 1):
                optimizer.zero_grad()
                L, r = 0., 0.
                S_fit, Z_fit = sample_task_group(meta_c.S_group[c], meta_c.Z_group[c],
                                                 sample_per_group=sample_per_group)
                for i in tqdm.tqdm(range(S_fit.shape[0])):  # only sample 10 per cluster
                    Si = S_fit[i, :].reshape(1, -1)
                    Zo = meta_c(Si, n_sample=10, verbal=False, do_alignment=do_alignment, grad=True)
                    Zi = Z_fit[i, :].reshape(1, -1)
                    Li = distance(Zo, Zi)
                    L += Li
                    r += Li.item() ** 0.5
                L /= S_fit.shape[0]
                r /= S_fit.shape[0]
                L.backward()
                optimizer.step()
                print("Iteration {}: Fitting Loss = {}".format(iter, r))
                if iter % checkpoint == 0:
                    print("Saving Meta Model ...")
                    torch.save(meta_c, META_MODEl_FOLDER + EXPERIMENT_NAME + "-branch-" + str(c) + ".pt")