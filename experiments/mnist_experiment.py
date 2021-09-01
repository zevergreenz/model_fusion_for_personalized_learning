from collections import OrderedDict
from sklearn.model_selection import train_test_split
from models.generator import ExpertGenerator
from models.meta import MetaExpert
from utilities.util import *
from experiments.mnist import MnistNet, test
from torchvision import datasets, transforms
from collections import defaultdict


EXPERIMENT_NAME = 'mnist-meta-model'
RESULT_FOLDER = BASE_RESULT_FOLDER + EXPERIMENT_NAME + "/"
do_alignment = False  # change to True/False depending on fit/no-fit

check(RESULT_FOLDER)

def prepare_data_loader(digits, train=True, shots=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=train, transform=transform)
    dataset_idx = torch.empty((0), dtype=torch.int)
    for digit in digits:
        idx = (dataset.targets == digit).nonzero().flatten()
        if shots is not None:
            idx = idx[torch.randperm(idx.shape[0])]
            idx = idx[:shots]
        dataset_idx = torch.cat([dataset_idx, idx])
    dataset = torch.utils.data.Subset(dataset, dataset_idx)
    return torch.utils.data.DataLoader(dataset, shuffle=True)

# Load side information and pre-trained models
S = np.load('../saves/pre-trained/mnist_S.npy')
Z = np.load('../saves/pre-trained/mnist_Z.npy')

# Train/test partition
seed()  # fix random seed for reproducibility
S_train, S_test, Z_train, Z_test = train_test_split(S, Z, test_size = 0.1, random_state = 42)

# Setup the meta learner
n_cluster = 5
z_dim, w_dim, a_dim = Z.shape[1], 100, 100

print("Training Legacy Model ...")

meta = MetaExpert(S_train, Z_train, z_dim, w_dim, a_dim, n_cluster = n_cluster, dev = device)
records = meta.fit(n_iter = 50, checkpoint = 10, do_alignment = do_alignment)

print("Saving Legacy Model ...")
torch.save(meta, META_MODEl_FOLDER + EXPERIMENT_NAME + "-legacy.pt")

print("Loading Legacy Meta Model ...")
meta = torch.load(META_MODEl_FOLDER + EXPERIMENT_NAME + "-legacy.pt", map_location = device)
meta = meta.to(device)

loss = nn.MSELoss()
def evaluate(digits, net, prefix = RESULT_FOLDER + 'zero-all-'):
    net.eval()
    test_dataloader = prepare_data_loader(digits, train=False)
    return test(model=net, device=device, test_loader=test_dataloader, digits=None)[1]

def distance(Zo, Zi):
    return torch.mean((Zo - Zi) ** 2.0)

print("Re-fitting ...")
meta.refit(n_cluster, EXPERIMENT_NAME, do_alignment, distance, n_iter=20)

print("Evaluate Few-Shot Performance ...")
metas = []
for c in range(n_cluster):
    metas.append(torch.load(META_MODEl_FOLDER + EXPERIMENT_NAME + "-branch-" + str(c) + ".pt", map_location = device))
for meta in metas:
    meta.eval()

results = defaultdict(list)
model = MnistNet().to(device)
init_state_dict = model.state_dict()
for i in range(S_test.shape[0]):
    S_ast = torch.FloatTensor(S_test[i, :]).reshape(1, -1).to(device)
    digits = (S_ast.reshape(-1) == 1.0).nonzero().flatten().cpu()
    idx = metas[0].p_wu.pic[0].map(S_ast)
    Z_ast = metas[idx](S_ast, n_sample = 10, verbal = False, do_alignment = do_alignment).reshape(-1)
    for n_shot in [0, 1, 5, 10]:
        model.tensor_to_state_dict(Z_ast)
        if n_shot > 0:
            dataloader = prepare_data_loader(digits=digits, train=True, shots=n_shot)
            model.fine_tune(dataloader, n_steps=20)
        accuracy = evaluate(digits, model)
        results[n_shot].append(accuracy)
        print("Digits {} | {}-shot: {}".format(digits, n_shot, accuracy))

for n_shot in [0, 1, 5, 10]:
    print("{}-shot: {} +- {}".format(n_shot, np.mean(results[n_shot]), np.std(results[n_shot])))