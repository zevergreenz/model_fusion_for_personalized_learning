from torch.utils.data import DataLoader, TensorDataset

from utilities.util import *
import matplotlib.pyplot as plt


def constant_data_generator(task, size, dev = device):
    X = np.linspace(-10.0, 10.0, size)
    mag, freq, phase = task  # extract sine parameters from task
    Y = mag * np.sin(2 * np.pi * freq * X + phase)
    X = torch.FloatTensor(X).view(-1, 1).to(dev)
    Y = torch.FloatTensor(Y).view(-1, 1).to(dev)
    return X, Y

class ExpertGenerator:
    def __init__(self, data_generator, net_generator, state_dict_to_tensor, tensor_to_state_dict, loss):
        self.data_generator, self.net_generator = data_generator, net_generator
        self.state_dict_to_tensor, self.tensor_to_state_dict = state_dict_to_tensor, tensor_to_state_dict
        self.loss = loss

    def train(self, task, state_dict = None, size = 10000, batch_size = 4096, n_iter = 10000, test_size = 0.1, verbal = True, checkpoint = 10):
        net = self.net_generator(dev = device)
        if state_dict is not None:
            net.load_state_dict(state_dict)
        if size == 0:
            return net
        X_train, Y_train = constant_data_generator(task, int(size * test_size), dev = device)
        data_iter = DataLoader(TensorDataset(X_train, Y_train), batch_size = min(batch_size, X_train.shape[0]), shuffle = True)
        optimizer = optim.Adam(net.parameters(), weight_decay = 1e-5)
        for i in range(1, n_iter + 1):
            for (X, Y) in data_iter:
                net.train()
                optimizer.zero_grad()
                regret = self.loss(net(X), Y)
                regret.backward()
                optimizer.step()
            if verbal is True and i % checkpoint == 0:
                with torch.no_grad():
                    train_regret = np.sqrt(self.loss(Y_train, net(X_train)).item())
                    print("Training epoch {}: loss (train) = {}".format(i, train_regret))
        return net
