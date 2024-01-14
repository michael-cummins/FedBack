import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from admm.agents import FedConsensus
from admm.servers import EventADMM
from admm.models import Cifar10CNN
from admm.utils import average_params
from admm.data import partition_data, split_dataset

sns.set_theme()

if torch.cuda.is_available(): 
    device = 'cuda'
    print('GPU available')
else:
    raise Exception('GPU not available')
    # device = 'cpu'

if __name__ == '__main__':

    """
    Data Preperation
    """

    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar_trainset = datasets.CIFAR10(
        root='./data/cifar10', train=True,
        download=True, transform=cifar_transform
    )
    cifar_testset = datasets.CIFAR10(
        root='./data/cifar10', train=False,
        download=True, transform=cifar_transform
    )

    train_dataset, val_dataset, _ = split_dataset(dataset=cifar_trainset, train_ratio=0.8, val_ratio=0.2)

    trainsets = partition_data(
        num_clients=10,
        iid=False,
        balance=True,
        power_law=False,
        seed=42,
        trainset=train_dataset.dataset,
        labels_per_partition=2
    )
    
    for i, dataset in enumerate(trainsets):
        labels = np.zeros(10)
        dummy_loader = DataLoader(dataset, batch_size=1)
        for data, target in dummy_loader:
            labels[int(target.item())] += 1
        print(f'Dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    labels = np.zeros(10)
    dummy_loader = DataLoader(val_dataset, batch_size=1)
    for data, target in dummy_loader:
        labels[int(target.item())] += 1
    print(f'Validation dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    labels = np.zeros(10)
    dummy_loader = DataLoader(cifar_testset, batch_size=1)
    for data, target in dummy_loader:
        labels[int(target.item())] += 1
    print(f'Validation dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    # 2 agents, batch 256, epochs 1 -> 100 steps per agent = 200 steps - 80% test acc
    # 10 agents, batch 50, epochs 1 -> 100 steps per agent = 1000 steps - 68% test acc
    batch_size = 50
    train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in trainsets]
    test_loader = DataLoader(cifar_testset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    """
    Setting up Consensus Problem
    """

    deltas = [0]
    lr = 0.001
    rho = 0.01
    t_max = 100

    acc_per_delta = np.zeros((len(deltas), t_max))
    rate_per_delta = np.zeros((len(deltas), t_max))
    loads = []
    test_accs = []

    for i, delta in enumerate(deltas):
    
        agents = []
        for loader in train_loaders:
            torch.manual_seed(78)
            model = Cifar10CNN()
            agents.append(
                FedConsensus(
                    N=len(train_loaders),
                    delta=delta,
                    rho=rho,
                    model=model,
                    loss=nn.CrossEntropyLoss(),
                    train_loader=loader,
                    epochs=1,
                    data_ratio=1,
                    device=device,
                    lr=lr
                ) 
            )

        # Broadcast average to all agents and check if equal
        for agent in agents:
            agent.primal_avg = average_params([agent.model.parameters() for agent in agents])
            print(f'Agents device = {next(agent.model.parameters()).device}')
        if device == 'cuda': torch.cuda.synchronize()

        """
        Run the consensus algorithm
        """

        torch.manual_seed(78)
        server = EventADMM(clients=agents, t_max=t_max, model=Cifar10CNN(), device=device)
        server.spin(loader=val_loader)
        final = agents[0].last_communicated
        
        # For plotting purposes
        acc_per_delta[i,:] = server.val_accs
        rate_per_delta[i,:] = server.rates
        load = sum(rate_per_delta[i,:])/t_max
        acc = server.validate_global(loader=test_loader)
        print(f'Load for delta {delta} = {load}, Test accuracy = {acc}')
        loads.append(load)
        test_accs.append(acc.cpu().numpy())

    """
    Plot Results
    """

    T = range(t_max)
    
    for acc, delta in zip(acc_per_delta, deltas):
        plt.plot(T, acc, label=f'delta={delta:.2f}')
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time Step')
    plt.ylabel('Accuracy')
    plt.title('Validation Set Accuracy - Fully Connected - niid')
    plt.savefig('./images/cifar/fc_val.png')

    for rate, delta in zip(rate_per_delta, deltas):
        plt.plot(T, rate, label=f'rate={delta:.2f}')
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time Step')
    plt.ylabel('Rate')
    plt.title('Communication Rate - Fully Connected')
    plt.savefig('./images/cifar/fc_comm_rate.png')

    for load, acc, delta in zip(loads, test_accs, deltas):
        plt.plot(acc, load, label=f'rate={delta:.2f}', marker='x')
    plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
    plt.xlabel('Test Accuracy')
    plt.ylabel('Communication Load')
    plt.title('Fully Connected')
    plt.savefig('./images/cifar/fc_test_load.png')
    
    # Save plotting data
    np.save(file='figure_data/cifar/rates_per_delta', arr=rate_per_delta)
    np.save(file='figure_data/cifar/accs_per_delta', arr=acc_per_delta)
    np.save(file='figure_data/cifar/loads_per_delta', arr=loads)
    np.save(file='figure_data/cifar/deltas', arr=deltas)