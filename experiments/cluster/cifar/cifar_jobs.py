from typing import List
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from admm.models import Cifar10CNN
from admm.agents import FedLearn, FedConsensus
from admm.servers import FedAgg, EventADMM
from admm.utils import average_params
import matplotlib.pyplot as plt
import numpy as np


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
                model = Cifar10CNN()
                self.agents.append(
                    FedLearn(
                        rho=self.rho,
                        loss=nn.CrossEntropyLoss(),
                        model=model,
                        train_loader=loader,
                        epochs=5,
                        device=self.device,
                        lr=self.lr
                    )
                )
            
            torch.manual_seed(78)
            global_model = Cifar10CNN()
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
        plt.title('Validation Set Accuracy')
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
