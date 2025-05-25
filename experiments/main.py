import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy
from admm.agents import FedConsensus
from admm.servers import EventADMM
from admm.models import Cifar10CNN, Model2
from admm.utils import average_params
# from admm.data import partition_data, split_dataset
# from admm.moon_dataset import get_dataloader, partition_data
from admm.data import get_cifar_data, get_mnist_data
from jobs import FedLearnJob, FedADMMJob, FedEventJob

sns.set_theme()
num_gpus = 1
if torch.cuda.is_available(): 
    device = 'cuda'
    torch.cuda.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    gpu = ''
    for i in range(num_gpus): gpu += f'{torch.cuda.get_device_name(i)}\n'
    print(gpu)
else:
    raise Exception('GPU not available')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Example script with command line parsing.")
    parser.add_argument("--avg", action="store_true", default=False, help="Enable avg (default: False)")
    parser.add_argument("--cifar", action="store_true", default=False, help="Enable avg (default: False)")
    parser.add_argument("--mnist", action="store_true", default=False, help="Enable avg (default: False)")
    parser.add_argument("--prox", action="store_true", default=False, help="Enable prox (default: False)")
    parser.add_argument("--admm", action="store_true", default=False, help="Enable admm (default: False)")
    parser.add_argument("--back", action="store_true", default=False, help="Enable admm (default: False)")
    parser.add_argument("--rate", type=float, required=True, help="Set the rate")

    args = parser.parse_args()
    print("avg:", args.avg)
    print("prox:", args.prox)
    print("admm:", args.admm)
    print("back:", args.back)
    print("rate:", args.rate)

    # Check for exclusive flags
    if sum([args.avg, args.prox, args.admm, args.back]) != 1:
        parser.error("Exactly one of --avg, --prox, or --admm must be set to True.")
    if args.cifar and args.mnist: parser.error('Can only specify one experiment, eith cifar or mnist')
    if not args.cifar and not args.mnist: parser.error('Must specify eith cifar or mnist')
    
    if args.cifar:
        num_clients = 100
        batch_size = 20
        train_loaders, test_loader = get_cifar_data(num_clients=num_clients, batch_size=batch_size)
        val_loader = copy(test_loader)
    elif args.mnist:
        num_clients = 100
        batch_size = 42
        train_loaders, test_loader, val_loader = get_mnist_data(num_clients=num_clients, batch_size=batch_size)

    """
    Run FedAVG and FedProx Experiments
    """
    rate = args.rate
    if args.cifar:
        t_max = 500
        epochs = 3
        if rate <= 0.2 and rate > 0.1: t_max = int(1.5*t_max)
        elif rate <= 0.05: t_max = int(3*t_max)
        elif rate <= 0.1: t_max = int(2*t_max)
    if args.mnist:
        t_max = 150
        epochs = 2
    
    
    prox_args = {
        'train_loaders': train_loaders, 'test_loader': test_loader, 'val_loader': val_loader,
        't_max': t_max, 'lr': 0.01, 'device': device, 'prox': True, 'epochs': epochs, 'rate': rate,
        'cifar': args.cifar, 'mnist':args.mnist
    }
    avg_args = prox_args.copy()
    avg_args['prox'] = False

    ADMM_args = {
        'train_loaders':train_loaders, 'test_loader':test_loader, 'val_loader': val_loader,
        't_max': t_max, 'lr':0.01, 'device':device, 'num_agents':num_clients, 'epochs': epochs, 'rate': rate,
        'cifar': args.cifar, 'mnist':args.mnist
    }

    back_args = {
        'train_loaders':train_loaders, 'test_loader':test_loader, 'val_loader': val_loader,
        't_max': t_max, 'lr':0.01, 'device':device, 'num_agents':num_clients, 'epochs': epochs, 'rate': rate,
        'cifar': args.cifar, 'mnist':args.mnist
    }

    if args.prox: job = FedLearnJob(**prox_args)
    elif args.avg: job = FedLearnJob(**avg_args)
    elif args.admm: job = FedADMMJob(**ADMM_args)
    elif args.back: job = FedEventJob(**back_args)
    else: raise ValueError('Need to select an algorithm (avg, prox, admm)')
    job.run()