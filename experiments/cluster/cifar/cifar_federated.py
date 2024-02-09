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

sns.set_theme()
num_gpus = 1
# torch.set_num_threads(16)
if torch.cuda.is_available(): 
    device = 'cuda'
    torch.cuda.manual_seed(42)
    # torch.backends.cuda.matmul.allow_tf32 = True
    gpu = ''
    for i in range(num_gpus): gpu += f'{torch.cuda.get_device_name(i)}\n'
    print(gpu)
else:
    raise Exception('GPU not available')

if __name__ == '__main__':
    """
    Data Preperation
    """

    # cifar_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # cifar_trainset = datasets.CIFAR10(
    #     root='./data/cifar10', train=True,
    #     download=True, transform=cifar_transform
    # )
    # cifar_testset = datasets.CIFAR10(
    #     root='./data/cifar10', train=False,
    #     download=True, transform=cifar_transform
    # )

    # train_dataset, val_dataset, _ = split_dataset(dataset=cifar_trainset, train_ratio=0.8, val_ratio=0.2)

    # trainsets = partition_data(
    #     num_clients=16,
    #     iid=True,
    #     balance=False,
    #     power_law=False,
    #     seed=108,
    #     trainset=train_dataset.dataset,
    #     labels_per_partition=10
    # )
    num_clients=10
    batch_size=64
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
        num_labels=10
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

    # labels = np.zeros(10)
    # dummy_loader = DataLoader(val_dataset, batch_size=1)
    # for data, target in dummy_loader:
    #     labels[int(target.item())] += 1
    # print(f'Validation dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    labels = np.zeros(10)
    dummy_loader = DataLoader(test_global_dl.dataset, batch_size=1)
    for data, target in dummy_loader:
        labels[int(target.item())] += 1
    print(f'Validation dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    
    """
    Setting up Consensus Problem
    """

    # deltas = list(range(0,32,4))
    # deltas = [8,10,14]
    # torch.autograd.detect_anomaly(True)
    # deltas = [0, 10, 16, 20, 24, 28, 30, 32]
    
    # rate_refs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1]
    rate_ref=0.1
    Kzs = [5, 3, 1]
    Kzs = [10]
    # Kzs.reverse()
    lr = 0.01
    t_max = 100
    rho = 0.01
    
    acc_per_item = np.zeros((len(Kzs), t_max))
    rate_per_item = np.zeros((len(Kzs), t_max))
    loads = []
    test_accs = []
    
    gamma = 0
    global_weight = rho/(rho*num_clients - 2*gamma)
    total_samples = sum([len(loader.dataset) for loader in trainloaders])

    item_list = Kzs
    item_key = 'K_z'

    for i, item in enumerate(item_list):
        print(f'Testing with K_z = {item} and initialising delta_z = 0 and using PI control')
        
        agents = []
        for j, loader in enumerate(trainloaders):
            data_ratio = len(loader.dataset)/total_samples            
            torch.manual_seed(42)
            model = Cifar10CNN()
            agents.append(
                FedConsensus(
                    N=len(trainloaders),
                    delta=0,
                    rho=rho/(data_ratio*num_clients),
                    model=model,
                    loss=nn.CrossEntropyLoss(),
                    train_loader=loader,
                    epochs=2,
                    data_ratio=data_ratio,
                    device=device,
                    lr=lr,
                    global_weight=global_weight
                ) 
            )

        # Broadcast average to all agents and check if equal
        for agent in agents:
            agent.primal_avg = average_params([agent.model.parameters() for agent in agents])
            # print(f'Agents device = {next(agent.model.parameters()).device}')
        if device == 'cuda': torch.cuda.synchronize()

        """
        Run the consensus algorithm
        """

        torch.manual_seed(42)
        global_model = Cifar10CNN()
        server = EventADMM(clients=agents, t_max=t_max, model=global_model, device=device)
        server.spin(loader=test_global_dl, K_x=5, K_z=item, rate_ref=rate_ref)
        
        # For plotting purposes
        acc_per_item[i,:] = server.val_accs
        rate_per_item[i,:] = server.rates
        load = sum(rate_per_item[i,:])/t_max
        acc = server.validate_global(loader=test_global_dl)
        print(f'Load for {item_key} {item} = {load} | Test accuracy = {acc}')
        loads.append(load)
        test_accs.append(acc.cpu().numpy())

    """
    Plot Results
    """

    T = range(t_max)
    
    for acc_per, item in zip(acc_per_item, item_list):
        plt.plot(T, acc_per, label=f'{item_key}={item}')
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time Step')
    plt.ylabel('Accuracy')
    plt.title('Validation Set Accuracy')
    plt.savefig('./images/FedEvent/fc_val_100.png')
    plt.cla()
    plt.clf()

    for rate, item in zip(rate_per_item, item_list):
        plt.plot(T, rate, label=f'{item_key}={item}')
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time Step')
    plt.ylabel('Rate')
    plt.title('Communication Rate')
    plt.savefig('./images/FedEvent/fc_comm_rate_100.png')
    plt.cla()
    plt.clf()

    for load, acc, item in zip(loads, test_accs, item_list):
        plt.plot(acc, load, label=f'{item_key}={item}', marker='x')
    plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
    plt.xlabel('Test Accuracy')
    plt.ylabel('Communication Load')
    plt.title('Test/Load tradeoff')
    plt.savefig('./images/FedEvent/fc_test_load_100.png')
    plt.cla()
    plt.clf()
    
    # Save plotting data
    np.save(file='figure_data/FedEvent/rates_per_delta_100', arr=rate_per_item)
    np.save(file='figure_data/FedEvent/accs_per_delta_100', arr=acc_per_item)
    np.save(file='figure_data/FedEvent/loads_per_delta_100', arr=loads)
    np.save(file='figure_data/FedEvent/deltas_100', arr=item_list)