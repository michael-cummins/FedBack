import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from admm.agents import FedLearn, FedConsensus, FedADMM
from admm.servers import FedAgg
from admm.models import FCNet
from admm.utils import average_params
from admm.data import partition_data, split_dataset
from mnist_jobs import FedLearnJob, FedEventJob
import seaborn as sns
sns.set_theme()

num_gpus = 1
if torch.cuda.is_available(): 
    device = 'cuda'
    torch.cuda.manual_seed(78)
    torch.backends.cuda.matmul.allow_tf32 = True
    gpu = ''
    for i in range(num_gpus): gpu += f'{torch.cuda.get_device_name(i)}\n'
    print(gpu)
else:
    raise Exception('GPU not available')

if __name__ == '__main__':
    mnist_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda x: torch.flatten(x))
    ])

    mnist_trainset = datasets.MNIST(
        root='./data/mnist_data', train=True,
        download=True, transform=mnist_transform, 
    )
    mnist_testset = datasets.MNIST(
        root='./data/mnist_data', train=False,
        download=True, transform=mnist_transform
    )

    train_dataset, val_dataset, _ = split_dataset(dataset=mnist_trainset, train_ratio=0.8, val_ratio=0.2)

    trainsets = partition_data(
        num_clients=100,
        iid=False,
        balance=True,
        power_law=False,
        seed=42,
        trainset=train_dataset.dataset,
        labels_per_partition=1
    )

    """
    Print data to evaluate heterogeneity
    """
    for i, dataset in enumerate(trainsets):
        labels = np.zeros(10)
        dummy_loader = DataLoader(dataset, batch_size=1)
        for data, target in dummy_loader:
            labels[target.item()] += 1
        print(f'Dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    labels = np.zeros(10)
    dummy_loader = DataLoader(val_dataset, batch_size=1)
    for data, target in dummy_loader:
        labels[target.item()] += 1
    print(f'Validation dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    labels = np.zeros(10)
    dummy_loader = DataLoader(mnist_testset, batch_size=1)
    for data, target in dummy_loader:
        labels[target.item()] += 1
    print(f'Validation dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    """
    Build DataLoaders
    """

    batch_size = 28
    train_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) for dataset in trainsets
    ]
    test_loader = DataLoader(mnist_testset, batch_size=100, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True, num_workers=0)

    """
    Run FedAVG and FedProx Experiments
    """
    t_max = 100
    num_clients = 100

    event_args = {
        'train_loaders':train_loaders, 'test_loader':test_loader, 'val_loader': val_loader,
        't_max':t_max, 'lr':0.15, 'device':device, 'num_clients':num_clients
    }

    job = FedEventJob(**event_args)
    job.run()
    
    prox_args = {
        'train_loaders':train_loaders, 'test_loader':test_loader, 'val_loader': val_loader,
        't_max':t_max, 'lr':0.01, 'device':device, 'prox':True
    }
    avg_args = prox_args.copy()
    avg_args['prox'] = False
    args = (prox_args, avg_args)

    for arg in args:
        job = FedLearnJob(**arg)
        job.run()

    """
    Run FedEvent Exxperiments
    """