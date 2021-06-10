import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

import numpy as np
import pandas as pd
import torch.optim as optim


import warnings
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os

def check(FOLDER):
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

BASE = "/tmp/"
PROJECT_NAME = "personalized-learning"

BASE_RESULT_FOLDER = BASE + PROJECT_NAME + "/saves/results/"
LOCAL_MODEL_FOLDER = BASE + PROJECT_NAME + "/saves/pre-trained/"
META_MODEl_FOLDER = BASE + PROJECT_NAME + "/saves/meta-model/"

check(BASE_RESULT_FOLDER)
check(LOCAL_MODEL_FOLDER)
check(META_MODEl_FOLDER)

def seed(np_seed = 93086455, torch_seed = 75216451):
    np.random.seed(np_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(torch_seed)

def print_cuda_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    print(f'Total:{t}, Cached:{c}, Alloc:{a}, Free:{t - c - a}')

