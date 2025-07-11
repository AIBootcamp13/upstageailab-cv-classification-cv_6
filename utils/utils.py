__all__ = ['set_seed', 'makedirs', 'get_yaml', 'seed_worker']
import os
import random

import numpy as np
import torch
import yaml

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed 고정 완료!')


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print('폴더 확인 완료!!')
    
def get_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    return data

def seed_worker(worker_id):
    """각 워커마다 고유하게 시드를 설정"""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
