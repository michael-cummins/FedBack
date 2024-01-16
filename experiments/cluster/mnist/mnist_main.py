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
    # device='cpu'
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
    train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) for dataset in trainsets]
    test_loader = DataLoader(mnist_testset, batch_size=100, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True, num_workers=0)

    """
    Setup Federated Learning Experiment
    """
    lr = 0.001
    t_max = 5
    loaders = train_loaders
    rates = np.arange(start=0.1, stop=0.7, step=0.1)
    rates = [*rates.tolist(), 1]
    print(rates)

    agents = []
    acc_per_rate = np.zeros((len(rates), t_max))
    rate_per_rate = np.zeros((len(rates), t_max))
    loads = []
    test_accs = []

    for i, rate in enumerate(rates):
        for loader in loaders:
            torch.manual_seed(78)
            model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)
            agents.append(
                FedLearn(
                    rho=0,
                    loss=nn.CrossEntropyLoss(),
                    model=model,
                    train_loader=loader,
                    epochs=1,
                    device=device,
                    lr=lr
                )
            )

        torch.manual_seed(78)
        global_model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)
        server = FedAgg(
            clients=agents, 
            t_max=t_max, 
            model=global_model, 
            device=device,
            C=rate
        )

        server.spin(loader=val_loader)
        
        acc_per_rate[i,:] = server.val_accs
        rate_per_rate[i,:] = server.rates
        load = sum(rate_per_rate[i,:])/t_max
        acc = server.validate_global(loader=test_loader)
        print(f'Test accuracy for rate {rate} = {acc}')
        loads.append(load)
        test_accs.append(acc.cpu().numpy())
    
    """
    Plot results
    """
    T = range(t_max)
    # Plot accuracies
    for acc, rate in zip(acc_per_rate, rates):
        plt.plot(T, acc, label=f'rate={rate:.2f}')
    plt.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
    plt.xlabel('Time Step')
    plt.ylabel('Accuracy')
    plt.title('Validation Set Accuracy - Fully Connected - Learning Rate = 0.001')
    plt.savefig('./images/FedAVG/fc_val.png')
    plt.cla()
    plt.clf()

    for load, acc, rate in zip(loads, test_accs, rates):
        plt.plot(acc, load, label=f'rate={rate:.2f}', marker='x')
    plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
    plt.xlabel('Test Accuracy')
    plt.xlim([0, 1])
    plt.ylim([0,1.1])
    plt.ylabel('Communication Load')
    plt.title('Fully Connected')
    plt.savefig('./images/FedAVG/fc_test_load.png')
    plt.cla()
    plt.clf()

    for rate, r in zip(rate_per_rate, rates):
        plt.plot(T, rate, label=f'rate={r:.2f}')
    plt.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
    plt.xlabel('Time Step')
    plt.ylabel('Rate')
    plt.title('Communication Rate - Fully Connected')
    plt.savefig('./images/FedAVG/fc_comm_rate.png')
    plt.cla()
    plt.clf()

    np.save(file='figure_data/FedAVG/rates_per_rate', arr=rate_per_rate)
    np.save(file='figure_data/FedAVG/accs_per_rate', arr=acc_per_rate)
    np.save(file='figure_data/FedAVG/loads_per_rate', arr=loads)
    np.save(file='figure_data/FedAVG/rates', arr=rates)
