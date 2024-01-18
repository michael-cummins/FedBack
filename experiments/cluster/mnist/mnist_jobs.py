from typing import List
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from admm.models import FCNet
from admm.agents import FedLearn, FedConsensus
from admm.servers import FedAgg, EventADMM
from admm.utils import average_params
import matplotlib.pyplot as plt
import numpy as np


class FedEventJob:

    def __init__(self, train_loaders: List[DataLoader], test_loader: DataLoader, val_loader: DataLoader,
                t_max: int, lr: float, device: str, num_clients: int) -> None:
    
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.t_max = t_max
        self.lr = lr
        self.device = device
        self.num_clients = num_clients

    def run(self) -> None:

        deltas = list(range(0,11,1))
        # deltas = [0]
        acc_per_delta = np.zeros((len(deltas), self.t_max))
        rate_per_delta = np.zeros((len(deltas), self.t_max))
        loads = []
        test_accs = []
        gamma = 1e-4
        rho = 0.01
        global_weight = rho/(rho*self.num_clients - 2*gamma)
        total_samples = sum([len(loader.dataset) for loader in self.train_loaders])
        for i, delta in enumerate(deltas):
            agents = []
            for j, loader in enumerate(self.train_loaders):
                data_ratio = len(loader.dataset)/total_samples
                torch.manual_seed(78)
                model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)
                agents.append(
                    FedConsensus(
                        N=len(self.train_loaders),
                        delta=delta,
                        rho=rho,
                        model=model,
                        loss=nn.CrossEntropyLoss(),
                        train_loader=loader,
                        epochs=1,
                        data_ratio=data_ratio,
                        device=self.device,
                        lr=self.lr,
                        global_weight=global_weight
                    ) 
                )

            
            # Broadcast average to all agents and check if equal
            for agent in agents:
                agent.primal_avg = average_params([agent.model.parameters() for agent in agents])
                # print(f'Agents device = {next(agent.model.parameters()).device}')
            if self.device == 'cuda': torch.cuda.synchronize()

            """
            Run the consensus algorithm
            """

            torch.manual_seed(78)
            global_model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)
            server = EventADMM(clients=agents, t_max=self.t_max, model=global_model, device=self.device)
            server.spin(loader=self.val_loader)
            
            # For plotting purposes
            acc_per_delta[i,:] = server.val_accs
            rate_per_delta[i,:] = server.rates
            load = sum(rate_per_delta[i,:])/self.t_max
            acc = server.validate_global(loader=self.val_loader)
            print(f'Load for delta {delta} = {load} | Test accuracy = {acc}')
            loads.append(load)
            test_accs.append(acc.cpu().numpy())
        
        """
        Plot Results
        """

        T = range(self.t_max)
        
        for acc_per, delta in zip(acc_per_delta, deltas):
            plt.plot(T, acc_per, label=f'rho={delta}:.2f')
        plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Time Step')
        plt.ylabel('Accuracy')
        plt.title('Validation Set Accuracy - Fully Connected - niid')
        plt.savefig('./images/FedEvent/fc_val.png')
        plt.cla()
        plt.clf()

        for rate, delta in zip(rate_per_delta, deltas):
            plt.plot(T, rate, label=f'rate={rho:.2f}')
        plt.legend(loc='center right', bbox_to_anchor=(1, 0.5))
        plt.xlabel('Time Step')
        plt.ylabel('Rate')
        plt.title('Communication Rate - Fully Connected')
        plt.savefig('./images/FedEvent/fc_comm_rate.png')
        plt.cla()
        plt.clf()

        for load, acc, delta in zip(loads, test_accs, deltas):
            plt.plot(acc, load, label=f'rate={delta:.2f}', marker='x')
        plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
        plt.xlabel('Test Accuracy')
        plt.ylabel('Communication Load')
        plt.title('Fully Connected')
        plt.savefig('./images/FedEvent/fc_test_load.png')
        plt.cla()
        plt.clf()
        
        # Save plotting data
        np.save(file='figure_data/FedEvent/rates_per_delta', arr=rate_per_delta)
        np.save(file='figure_data/FedEvent/accs_per_delta', arr=acc_per_delta)
        np.save(file='figure_data/FedEvent/loads_per_delta', arr=loads)
        np.save(file='figure_data/FedEvent/deltas', arr=deltas)


class FedLearnJob:

    def __init__(self, train_loaders: List[DataLoader], test_loader: DataLoader, val_loader: DataLoader,
                t_max: int, lr: float, prox: bool, device: str):
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.t_max = t_max
        self.lr = lr
        self.rho = 0.01 if prox else 0
        self.device = device
        self.prox = prox
        print(f'Prox: {self.prox}')

    def run(self):
        rates = np.arange(start=0.1, stop=0.6, step=0.1)
        rates = [*rates.tolist(), 1]
        print(rates)

        self.agents = []
        self.acc_per_rate = np.zeros((len(rates), self.t_max))
        self.rate_per_rate = np.zeros((len(rates), self.t_max))
        self.loads = []
        self.test_accs = []

        for i, rate in enumerate(rates):
            for loader in self.train_loaders:
                torch.manual_seed(78)
                model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)
                self.agents.append(
                    FedLearn(
                        rho=self.rho,
                        loss=nn.CrossEntropyLoss(),
                        model=model,
                        train_loader=loader,
                        epochs=1,
                        device=self.device,
                        lr=self.lr
                    )
                )
            
            torch.manual_seed(78)
            global_model = FCNet(in_channels=784, hidden1=200, hidden2=None, out_channels=10)
            server = FedAgg(
                clients=self.agents, 
                t_max=self.t_max, 
                model=global_model, 
                device=self.device,
                C=rate
            )

            server.spin(loader=self.val_loader)
        
            self.acc_per_rate[i,:] = server.val_accs
            self.rate_per_rate[i,:] = server.rates
            load = sum(self.rate_per_rate[i,:])/self.t_max
            acc = server.validate_global(loader=self.test_loader)
            print(f'Test accuracy for rate {rate} = {acc}')
            self.loads.append(load)
            self.test_accs.append(acc.cpu().numpy())

        image_dir = './images/FedProx/' if self.prox else './images/FedAVG/'
        figure_data_dir = './figure_data/FedProx/' if self.prox else './figure_data/FedAVG/'

        np.save(file=figure_data_dir+'rates_per_rate', arr=self.rate_per_rate)
        np.save(file=figure_data_dir+'accs_per_rate', arr=self.acc_per_rate)
        np.save(file=figure_data_dir+'loads_per_rate', arr=self.loads)
        np.save(file=figure_data_dir+'rates', arr=rates)

        T = range(self.t_max)
        # Plot accuracies
        for acc, rate in zip(self.acc_per_rate, rates):
            plt.plot(T, acc, label=f'rate={rate:.2f}')
        plt.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
        plt.xlabel('Time Step')
        plt.ylabel('Accuracy')
        plt.title('Validation Set Accuracy - Fully Connected - Learning Rate = 0.001')
        plt.savefig(image_dir+'fc_val.png')
        plt.cla()
        plt.clf()

        for load, acc, rate in zip(self.loads, self.test_accs, rates):
            plt.plot(acc, load, label=f'rate={rate:.2f}', marker='x')
        plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
        plt.xlabel('Test Accuracy')
        plt.xlim([0, 1])
        plt.ylim([0,1.1])
        plt.ylabel('Communication Load')
        plt.title('Fully Connected')
        plt.savefig(image_dir+'fc_test_load.png')
        plt.cla()
        plt.clf()

        for rate, r in zip(self.rate_per_rate, rates):
            plt.plot(T, rate, label=f'rate={r:.2f}')
        plt.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
        plt.xlabel('Time Step')
        plt.ylabel('Rate')
        plt.title('Communication Rate - Fully Connected')
        plt.savefig(image_dir+'fc_comm_rate.png')
        plt.cla()
        plt.clf()
