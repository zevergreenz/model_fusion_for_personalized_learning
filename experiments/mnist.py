import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utilities.util import device


class MnistNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10):
        super(MnistNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def state_dict_to_tensor(self):
        s = self.state_dict()
        fc1 = torch.cat([s['fc1.weight'], s['fc1.bias'].view(-1, 1)], dim=1)
        fc2 = torch.cat([s['fc2.weight'], s['fc2.bias'].view(-1, 1)], dim=1)
        return torch.cat([fc1.flatten(), fc2.flatten()], dim=0)

    def tensor_to_state_dict(self, weights):
        fc1 = weights[:(self.input_dim + 1) * self.hidden_dim].reshape(self.hidden_dim, self.input_dim + 1)
        fc2 = weights[(self.input_dim + 1) * self.hidden_dim:].reshape(self.output_dim, self.hidden_dim + 1)
        state_dict = OrderedDict([
            ('fc1.weight', fc1[:, :-1]),
            ('fc1.bias', fc1[:, -1]),
            ('fc2.weight', fc2[:, :-1]),
            ('fc2.bias', fc2[:, -1])
        ])
        self.load_state_dict(state_dict)

    def fine_tune(self, dataloader, n_steps=20):
        if len(dataloader.dataset) == 0:
            return
        self.train()
        optimizer = optim.Adadelta(self.parameters(), lr=args.lr)
        for _ in range(n_steps):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                data = data.view(-1, self.input_dim)
                optimizer.zero_grad()
                output = self(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, model.input_dim)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, digits=None):
    model.eval()
    test_loss = 0
    correct = 0
    if digits is not None:
        mask = torch.zeros(1, 10).to(device)
        mask[0, digits.long()] = 1.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, model.input_dim)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            if digits is not None:
                output -= output.min(dim=1, keepdim=True).values
                output *= mask
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = float(correct) / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, accuracy


def train_model(digits, samples_per_digit=500):
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.Resize(20),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_idx = torch.empty((0), dtype=torch.int)
    for digit in digits:
        idx = (dataset1.targets == digit).nonzero().flatten()
        idx = idx[torch.randperm(idx.shape[0])]
        idx = idx[:samples_per_digit]
        train_idx = torch.cat([train_idx, idx])
    dataset1 = torch.utils.data.Subset(dataset1, train_idx)

    test_idx = torch.empty((0), dtype=torch.int)
    for digit in digits:
        idx = (dataset2.targets == digit).nonzero().flatten()
        test_idx = torch.cat([test_idx, idx])
    dataset2 = torch.utils.data.Subset(dataset2, test_idx)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = MnistNet(input_dim=400).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    return model


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

list_of_digits = list(itertools.combinations(range(10), 2))

sideinfo = torch.empty((0, 10), dtype=torch.float).to(device)
weights = None
for digits in list_of_digits:
    m = train_model(digits, samples_per_digit=100000)
    w = m.state_dict_to_tensor().view(1, -1)
    s = torch.zeros((1, 10), dtype=torch.float).to(device)
    s[0, digits] = 1.0
    sideinfo = torch.cat([sideinfo, s], dim=0)
    if weights is None:
        weights = w
    else:
        weights = torch.cat([weights, w], dim=0)

sideinfo = sideinfo.cpu().numpy()
weights = weights.cpu().numpy()
np.save('mnist_S', sideinfo)
np.save('mnist_Z', weights)