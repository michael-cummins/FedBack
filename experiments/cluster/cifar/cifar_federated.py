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
    num_clients=100
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

    # 2 agents, batch 256, epochs 1 -> 100 steps per agent = 200 steps - 80% test acc
    # 10 agents, batch 50, epochs 1 -> 100 steps per agent = 1000 steps - 68% test acc
    # 10 agents (niid), batch 50, epochs 2 -> 200 steps per agent over 100 rounds = 2000 steps - 30% test acc
    # batch_size = 32
    # train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True) for dataset in trainsets]
    # test_loader = DataLoader(cifar_testset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)

    """
    Setting up Consensus Problem
    """

    # deltas = list(range(0,32,4))
    # deltas = [8,10,14]
    # torch.autograd.detect_anomaly(True)
    deltas = [0, 4, 8, 12, 16, 20, 24]
    lr = 0.1
    t_max = 100
    rho = 0.01/num_clients
    acc_per_delta = np.zeros((len(deltas), t_max))
    rate_per_delta = np.zeros((len(deltas), t_max))
    loads = []
    test_accs = []
    gamma = 0
    global_weight = rho/(rho*num_clients - 2*gamma)
    print(f'Testing for Nan')

    total_samples = sum([len(loader.dataset) for loader in trainloaders])
    for i, delta in enumerate(deltas):
        agents = []
        for j, loader in enumerate(trainloaders):
            data_ratio = len(loader.dataset)/total_samples
            # print(f'agent {j} data ratio: {data_ratio}')
            torch.manual_seed(42)
            model = Cifar10CNN()
            agents.append(
                FedConsensus(
                    N=len(trainloaders),
                    delta=delta,
                    rho=rho/data_ratio,
                    model=model,
                    loss=nn.CrossEntropyLoss(),
                    train_loader=loader,
                    epochs=20,
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
        server.spin(loader=test_global_dl)
        
        # For plotting purposes
        acc_per_delta[i,:] = server.val_accs
        rate_per_delta[i,:] = server.rates
        load = sum(rate_per_delta[i,:])/t_max
        acc = server.validate_global(loader=test_global_dl)
        print(f'Load for delta {delta} = {load} | Test accuracy = {acc}')
        loads.append(load)
        test_accs.append(acc.cpu().numpy())

    """
    Plot Results
    """

    T = range(t_max)
    
    for acc_per, delta in zip(acc_per_delta, deltas):
        plt.plot(T, acc_per, label=f'rho={delta}:.2f')
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time Step')
    plt.ylabel('Accuracy')
    plt.title('Validation Set Accuracy - Fully Connected - niid')
    plt.savefig('./images/FedEvent/fc_val_100.png')
    plt.cla()
    plt.clf()

    for rate, delta in zip(rate_per_delta, deltas):
        plt.plot(T, rate, label=f'rate={rho:.2f}')
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time Step')
    plt.ylabel('Rate')
    plt.title('Communication Rate - Fully Connected')
    plt.savefig('./images/FedEvent/fc_comm_rate_100.png')
    plt.cla()
    plt.clf()

    for load, acc, delta in zip(loads, test_accs, deltas):
        plt.plot(acc, load, label=f'rate={delta:.2f}', marker='x')
    plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
    plt.xlabel('Test Accuracy')
    plt.ylabel('Communication Load')
    plt.title('Fully Connected')
    plt.savefig('./images/FedEvent/fc_test_load_100.png')
    plt.cla()
    plt.clf()
    
    # Save plotting data
    np.save(file='figure_data/FedEvent/rates_per_delta_100', arr=rate_per_delta)
    np.save(file='figure_data/FedEvent/accs_per_delta_100', arr=acc_per_delta)
    np.save(file='figure_data/FedEvent/loads_per_delta_100', arr=loads)
    np.save(file='figure_data/FedEvent/deltas_100', arr=deltas)