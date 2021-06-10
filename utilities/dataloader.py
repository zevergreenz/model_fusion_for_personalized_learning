from d2l import mxnet as d2l
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

d2l.DATA_HUB['ml-100k'] = ('http://files.grouplens.org/datasets/movielens/ml-100k.zip',
                           'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

def read_data_ml100k(dir = None):
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names = names, engine = 'python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    if dir is not None:
        if not os.path.exists(dir):  # check if the specified folder exists
            os.makedirs(dir)  # if not, create it
        data.to_csv(dir + 'MovieLens.csv')
    return data, num_users, num_items

def split_data(data, num_users, num_items, split_mode = 'random', test_ratio = 0.1):
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key = lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data, test_data = pd.DataFrame(train_data), pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]

    return train_data, test_data

def load_data_ml100k(data, num_users, num_items, feedback = 'explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter

def split_and_load_ml100k(split_mode = 'seq-aware', feedback = 'explicit', test_ratio = 0.1, batch_size = 256, num_users=800):
    data, _, num_items = read_data_ml100k()
    data = data.loc[data['user_id'] < num_users + 1]
    train_data, test_data = split_data(data, num_users, num_items, split_mode = split_mode, test_ratio = test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(train_data, num_users, num_items, feedback = feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(test_data, num_users, num_items, feedback =  feedback)
    train_set = TensorDataset(torch.LongTensor(np.array(train_u)), torch.LongTensor(np.array(train_i)),
                              torch.FloatTensor(np.array(train_r)))
    test_set = TensorDataset(torch.LongTensor(np.array(test_u)), torch.LongTensor(np.array(test_i)),
                             torch.FloatTensor(np.array(test_r)))
    train_iter = DataLoader(train_set, shuffle = True, batch_size = batch_size)
    test_iter = DataLoader(test_set, shuffle = True, batch_size = batch_size)
    return num_users, num_items, train_iter, test_iter

def split_and_load_ml100k_context(split_mode = 'seq-aware', feedback = 'explicit', test_ratio = 0.1, batch_size = 256, num_items=20):
    data, num_users, _ = read_data_ml100k()
    data = data.loc[data['item_id'] < num_items + 1]
    train_data, test_data = split_data(data, num_users, num_items, split_mode = split_mode, test_ratio = test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(train_data, num_users, num_items, feedback = feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(test_data, num_users, num_items, feedback =  feedback)
    train_set = TensorDataset(torch.LongTensor(np.array(train_u)), torch.LongTensor(np.array(train_i)),
                              torch.FloatTensor(np.array(train_r)))
    test_set = TensorDataset(torch.LongTensor(np.array(test_u)), torch.LongTensor(np.array(test_i)),
                             torch.FloatTensor(np.array(test_r)))
    train_iter = DataLoader(train_set, shuffle = True, batch_size = batch_size)
    test_iter = DataLoader(test_set, shuffle = True, batch_size = batch_size)
    return num_users, num_items, train_iter, test_iter