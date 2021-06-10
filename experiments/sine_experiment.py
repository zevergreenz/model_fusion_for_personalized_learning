from collections import OrderedDict
from sklearn.model_selection import train_test_split
from models.generator import ExpertGenerator
from models.meta import MetaExpert
from utilities.util import *


EXPERIMENT_NAME = 'sine-meta-model'
RESULT_FOLDER = BASE_RESULT_FOLDER + EXPERIMENT_NAME + "/"
do_alignment = True  # change to True/False depending on fit/no-fit

check(RESULT_FOLDER)

def state_dict_to_tensor(state_dict: OrderedDict):
    return torch.cat([t.view(-1) for t in state_dict.values()])

def tensor_to_state_dict(tensor: torch.Tensor):
    return OrderedDict([('0.weight', tensor[0:100].view(100, 1)), ('0.bias', tensor[100:200].view(100)),
                        ('2.weight', tensor[200:300].view(1, 100)), ('2.bias', tensor[300].view(1))])

def load_sine_net(weight, dev = None):
    m = nn.Sequential(nn.Linear(1, 100), nn.ReLU(), nn.Linear(100, 1))
    m.load_state_dict(weight)
    if dev is not None:
        m = m.to(dev)
    return m
net_loader = lambda weight, dev: load_sine_net(weight, dev)

def generate_sine_data(task, test_size = 0.1, dev = None, low = -5, high = 5, size = 50):  # generate data for g(x) = a * sin(2 * pi * f * x + p)
    mag, freq, phase = task  # extract sine parameters from task
    X = np.linspace(low, high, size)
    Y = mag * np.sin(2 * np.pi * freq * X + phase)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = 42)
    X_train, X_test, Y_train, Y_test = torch.FloatTensor(X_train), torch.FloatTensor(X_test), \
                                       torch.FloatTensor(Y_train), torch.FloatTensor(Y_test)
    X_train, X_test, Y_train, Y_test = X_train.view(-1, 1), X_test.view(-1, 1), Y_train.view(-1, 1), Y_test.view(-1, 1)
    if dev is not None:
        X_train, X_test, Y_train, Y_test = X_train.to(dev), X_test.to(dev), Y_train.to(dev), Y_test.to(dev)
    return X_train, X_test, Y_train, Y_test

def generate_sine_net(dev = None):  # generate prediction net for the sine function
    net = nn.Sequential(nn.Linear(1, 100), nn.ReLU(), nn.Linear(100, 1))
    if dev is not None:
        net = net.to(dev)
    return net

data_generator = lambda task, size, test_size, dev: generate_sine_data(task, test_size = test_size, dev = dev, low = -10, high = 10, size = size)
net_generator = lambda dev: generate_sine_net(dev = dev)
generator = ExpertGenerator(data_generator, net_generator, state_dict_to_tensor, tensor_to_state_dict, nn.MSELoss())

# Load side information and pre-trained models
S = torch.load('../saves/pre-trained/S-100-100-big-different-init.pt', map_location = torch.device('cpu'))
Z = torch.load('../saves/pre-trained/Z-100-100-big-different-init.pt', map_location = torch.device('cpu'))
S = S.detach().cpu().numpy()
Z = Z.detach().cpu().numpy()  # Z is pre-trained and will be fixed so it has to be detached

# Train/test partition
seed()  # fix random seed for reproducibility
S_train, S_test, Z_train, Z_test = train_test_split(S, Z, test_size = 0.1, random_state = 42)

# Setup the meta learner
n_cluster = 5
z_dim, w_dim, a_dim = Z.shape[1], 100, 100
low, high = -5, 5

print("Training Legacy Model ...")

meta = MetaExpert(S_train, Z_train, z_dim, w_dim, a_dim, n_cluster = n_cluster, dev = device)
records = meta.fit(n_iter = 5, checkpoint = 10, do_alignment = do_alignment)

print("Saving Legacy Model ...")
torch.save(meta, META_MODEl_FOLDER + EXPERIMENT_NAME + "-legacy.pt")

print("Loading Legacy Meta Model ...")
meta = torch.load(META_MODEl_FOLDER + EXPERIMENT_NAME + "-legacy.pt", map_location = device)
meta = meta.to(device)

loss = nn.MSELoss()
def evaluate(tsk, net_zero, net_data, size = 200, low = -5.0, high = 5.0, prefix = RESULT_FOLDER + 'zero-all-'):
    net_zero.eval()
    net_data.eval()
    mag, freq, phase = tsk  # extract sine parameters from task
    X_test = np.linspace(low, high, size)
    Y_test = mag * np.sin(2 * np.pi * freq * X_test + phase)
    X_test = torch.FloatTensor(X_test).view(-1, 1).to(device)
    Y_test = torch.FloatTensor(Y_test).view(-1, 1).to(device)
    Y_zero, Y_data = net_zero(X_test), net_data(X_test)
    return np.sqrt(loss(Y_test, Y_zero).item()), np.sqrt(loss(Y_test, Y_data).item())

def compute(Z, X):   # X is batch_size by 1; Z is 1 by 301
    res = F.relu(torch.mm(X, Z[0, :100].view(1, -1)) + Z[0, 100:200].view(1, -1))  # res is batch_size by 100
    res = torch.mm(res, Z[0, 200:300].view(-1, 1)) + Z[0, 300].view(1, 1)  # res is batch_size by 1
    return res  # res is batch_size by 1

def distance(Zo, Zi, low = -5.0, high = 5.0, size = 200):
    X_test = np.linspace(low, high, size)
    X_test = torch.FloatTensor(X_test).view(-1, 1).to(device)
    Yo = compute(Zo, X_test)  # batch_size by 1
    Yi = compute(Zi, X_test)  # batch_size by 1
    return torch.mean((Yo - Yi) ** 2.0, dim = 0)

print("Re-fitting ...")
meta.refit(n_cluster, EXPERIMENT_NAME, do_alignment, distance, n_iter=2)

print("Evaluate Few-Shot Performance ...")
metas = []
for c in range(n_cluster):
    metas.append(torch.load(META_MODEl_FOLDER + EXPERIMENT_NAME + "-branch-" + str(c) + ".pt", map_location = device))
for meta in metas:
    meta.eval()

results = []
for i in range(S_test.shape[0]):
    S_ast = torch.FloatTensor(S_test[i, :]).reshape(1, -1).to(device)
    idx = metas[0].p_wu.pic[0].map(S_ast)
    task = (S_ast[0, 0].item(), 1.0 / (2.0 * np.pi), S_ast[0, 1].item())
    Z_ast = metas[idx](S_ast, n_sample = 10, verbal = False, do_alignment = do_alignment)
    M_zero = net_loader(tensor_to_state_dict(Z_ast.view(-1).detach()), dev = device)
    M_data = net_loader(tensor_to_state_dict(torch.FloatTensor(Z_test[i].reshape(1, -1)).view(-1)), dev = device)
    p_zero, p_data = evaluate(task, M_zero, M_data, low = low, high = high, prefix = RESULT_FOLDER + 'zero-all-')
    state_dict = tensor_to_state_dict(Z_ast.view(-1).detach())
    M_one = generator.train(task, state_dict, size = 15, test_size = 1.0, n_iter = 5000, verbal = False)
    M_five = generator.train(task, state_dict, size = 30, test_size = 1.0, n_iter = 5000, verbal = False)
    p_one, p_five = evaluate(task, M_one, M_five, low = low, high = high, prefix = RESULT_FOLDER + 'one-five-')
    print("Task {} - {}: DT = {} | 0-shot = {} | 1-shot = {}, 5-shot = {}".format(i + 1, [task[0], task[2]], p_data, p_zero, p_one, p_five))