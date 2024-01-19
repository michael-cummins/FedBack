import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from admm.agents import FedConsensus
from admm.servers import EventADMM
from admm.models import Cifar10CNN, Model2
from admm.utils import average_params
from admm.data import partition_data, split_dataset
from admm.moon_dataset import get_dataloader, partition_data
from cifar_jobs import FedLearnJob, FedADMMJob

sns.set_theme()
num_gpus = 1
if torch.cuda.is_available(): 
    device = 'cuda'
    # torch.cuda.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    gpu = ''
    for i in range(num_gpus): gpu += f'{torch.cuda.get_device_name(i)}\n'
    print(gpu)
else:
    raise Exception('GPU not available')

if __name__ == '__main__':
    num_clients=10
    batch_size=64
    
    # torch.manual_seed(42)
    # np.random.seed(42)
    (
        _,
        _,
        _,
        _,
        net_dataidx_map,
    ) = partition_data(
        partition='noniid',
        num_clients=num_clients,
        beta=0.5,
    )
    _, test_global_dl, _, _ = get_dataloader(
        datadir='./data/cifar10',
        train_bs=32,
        test_bs=32,
    )
    trainloaders = []
    for idx in range(num_clients):
        train_dl, _, _, _ = get_dataloader(
            './data/cifar10', batch_size, 32, net_dataidx_map[idx]
        )
        trainloaders.append(train_dl)
    trainsets = [loader.dataset for loader in trainloaders]
    
    for i, dataset in enumerate(trainsets):
        labels = np.zeros(10)
        dummy_loader = DataLoader(dataset, batch_size=1)
        for data, target in dummy_loader:
            labels[int(target.item())] += 1
        print(f'Dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    labels = np.zeros(10)
    dummy_loader = DataLoader(test_global_dl.dataset, batch_size=1)
    for data, target in dummy_loader:
        labels[int(target.item())] += 1
    print(f'Validation dataset {i} distribution: {labels} - num_samples = {labels.sum()}')


    """
    Run FedAVG and FedProx Experiments
    """
    t_max = 100
    num_clients = 10

    prox_args = {
        'train_loaders':trainloaders, 'test_loader':test_global_dl, 'val_loader': test_global_dl,
        't_max':t_max, 'lr':0.01, 'device':device, 'prox':True
    }
    avg_args = prox_args.copy()
    avg_args['prox'] = False
    args = (prox_args, avg_args)

    ADMM_args = {
        'train_loaders':trainloaders, 'test_loader':test_global_dl, 'val_loader': test_global_dl,
        't_max':t_max, 'lr':0.01, 'device':device, 'num_agents':num_clients
    }

    # for arg in args:
    job = FedADMMJob(**ADMM_args)
    job.run()