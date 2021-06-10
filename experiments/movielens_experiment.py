from models.meta import MetaExpert
from utilities.dataloader import read_data_ml100k
from utilities.util import *


EXPERIMENT_NAME = 'movie-len'
RESULT_FOLDER = BASE_RESULT_FOLDER + EXPERIMENT_NAME + "/"

check(RESULT_FOLDER)

# Load side information and pre-trained models
u_embedding = torch.load(LOCAL_MODEL_FOLDER + 'user_embedding.pt').cpu()
i_embedding = torch.load(LOCAL_MODEL_FOLDER + 'item_embedding.pt').cpu()
u_context   = torch.load(LOCAL_MODEL_FOLDER + 'user_context.pt')
i_context   = torch.load(LOCAL_MODEL_FOLDER + 'item_context.pt')

S_train = u_context[:800]
Z_train = u_embedding[:800]
S_test  = u_context[800:]
Z_test  = u_embedding[800:]

# Train/test partition
S_train = S_train.detach().cpu().numpy()
Z_train = Z_train.detach().cpu().numpy()

# Setup the meta learner
n_cluster = 10
do_alignment = True
z_dim, w_dim, a_dim = Z_train.shape[1], 100, 100

meta = MetaExpert(S_train, Z_train, z_dim, w_dim, a_dim, n_cluster = n_cluster, dev = device)
records = meta.fit(n_iter = 3, checkpoint = 10, do_alignment = do_alignment)

print("Saving Meta Model ...")
torch.save(meta, META_MODEl_FOLDER + EXPERIMENT_NAME + "-legacy.pt")

print("Loading Meta Model ...")
meta = torch.load(META_MODEl_FOLDER + EXPERIMENT_NAME + "-legacy.pt", map_location = device)
meta = meta.to(device)

loss = nn.MSELoss()
def evaluate(i, Z_ast):
    Z_ast = Z_ast.view(-1).detach().cpu()
    data, num_users, num_items = read_data_ml100k()
    data = data.loc[(data['user_id'] == 800 + i) & (data['item_id'] >= 300)]
    loss = 0
    count = 0
    for line in data.itertuples():
        user, item, rating, time = line[1], line[2], line[3], line[4]
        item = int(item) - 1
        pred = torch.dot(Z_ast[:-1], i_embedding[item][:-1].cpu())
        pred += Z_ast[-1] + i_embedding[item][-1].cpu()
        pred = torch.clip(pred, 1, 5)
        loss += (rating - pred) ** 2
        count += 1
    if count == 0:
        return None, None, None
    loss /= count
    loss = torch.sqrt(loss)
    return loss, rating, pred

def evaluate_coldstart(i):
    data, num_users, num_items = read_data_ml100k()
    data = data.loc[(data['user_id'] == 800 + i) & (data['item_id'] < 300)]
    A = []
    b = []
    for line in data.itertuples():
        user, item, rating, time = line[1], line[2], line[3], line[4]
        item = int(item) - 1
        A.append(i_embedding[item].view(1, -1))
        b.append(rating)
    if len(A) == 0:
        return 0, 1, 1
    A = torch.cat(A, dim=0)
    b = torch.FloatTensor(b).view(-1)
    Z = torch.pinverse(A) @ b
    data, num_users, num_items = read_data_ml100k()
    data = data.loc[(data['user_id'] == 800 + i) & (data['item_id'] >= 300)]
    loss = 0
    count = 0
    for line in data.itertuples():
        user, item, rating, time = line[1], line[2], line[3], line[4]
        item = int(item) - 1
        pred = torch.dot(Z.view(-1)[:-1], i_embedding[item][:-1])
        pred += Z.view(-1)[-1] + i_embedding[item][-1]
        loss += (rating - pred) ** 2
        count += 1
        print("Rating {} Prediction {}".format(rating, pred))
    if count == 0:
        return None, None, None
    loss /= count
    loss = torch.sqrt(loss)
    return loss, rating, pred

def evaluate_nn(i, S_ast, k=1):
    idx = torch.argsort(torch.mean(torch.square(torch.FloatTensor(S_train) - S_ast.view(1, -1)), dim=1), dim=0)[:k]
    Z = u_embedding[idx]
    data, num_users, num_items = read_data_ml100k()
    data = data.loc[(data['user_id'] == 800 + i) & (data['item_id'] >= 300)]
    loss = 0
    count = 0
    for line in data.itertuples():
        user, item, rating, time = line[1], line[2], line[3], line[4]
        item = int(item) - 1
        pred = torch.mean(torch.matmul(Z.view(k, -1)[:,:-1], i_embedding[item][:-1]))
        pred += torch.mean(Z.view(k, -1)[:,-1]) + i_embedding[item][-1]
        loss += (rating - pred) ** 2
        count += 1
        print("Rating {} Prediction {}".format(rating, pred))
    if count == 0:
        return None, None, None
    loss /= count
    loss = torch.sqrt(loss)
    return loss, rating, pred

def distance(Zo, Zi):
    return torch.sum(torch.square(Zo - Zi))

print("Re-fitting ...")
meta.refit(n_cluster, EXPERIMENT_NAME, do_alignment, distance, n_iter=2)

print("Evaluate Few-Shot Performance ...")
metas = []
for c in range(n_cluster):
    metas.append(torch.load(META_MODEl_FOLDER + EXPERIMENT_NAME + "-branch-" + str(c) + ".pt", map_location = device))
for meta in metas:
    meta.eval()
    meta.to(device)

results = []
for i in range(S_test.shape[0] - 1):
    S_ast = torch.FloatTensor(S_test[i, :]).reshape(1, -1).to(device)
    idx = metas[0].p_wu.pic[0].map(S_ast)
    Z_ast = metas[idx](S_ast, n_sample = 10, verbal = False, do_alignment = do_alignment)
    Z_ast = Z_ast.cpu().view(-1)
    error, rating, pred = evaluate(i, Z_ast)
    if error is not None:
        results.append(error)
    print("User {}: RMSE {}".format(i, error))
    print("Cumulative RMSE: {} ({})".format(torch.mean(torch.FloatTensor(results)), torch.std(torch.FloatTensor(results))))