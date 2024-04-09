from typing import List
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from admm.models import Cifar10CNN
from admm.agents import FedLearn, FedConsensus, FedADMM
from admm.servers import FedAgg, EventADMM, InexactADMM
from admm.utils import average_params
import matplotlib.pyplot as plt
from typing import List
import numpy as np

class FedEventJob:

    def __init__(self, train_loaders: List[DataLoader], test_loader: DataLoader, val_loader: DataLoader,
                t_max: int, lr: float, device: str, num_agents: int, epochs: int, 
                rate=None, forward=False, delta=None) -> None:
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.t_max = t_max
        self.lr = lr
        self.forward = forward
        self.epochs = epochs
        self.rho = 0.01
        self.rate = rate
        self.delta = delta
        self.device = device
        self.num_agents = num_agents
        if forward and delta is not None: raise ValueError('Must specify delta if in forward mode')
        if rate is not None and delta is not None: raise ValueError('Must give either a rate or delta')

    def run(self):
        if self.forward: items = self.delta
        elif isinstance(self.rate, list): items = self.rate
        else: items = [self.rate]
        
        acc_per_item = np.zeros((len(items), self.t_max))
        rate_per_item = np.zeros((len(items), self.t_max))
        loads = []
        test_accs = []
        
        gamma = 0
        global_weight = self.rho/(self.rho*self.num_agents - 2*gamma)
        total_samples = sum([len(loader.dataset) for loader in self.train_loaders])

        for i, item in enumerate(items):
            print(f'Testing with K_z = {item} and initialising delta_z = 0 and using leaky PI control')
            
            agents: List[FedConsensus] = []
            for j, loader in enumerate(self.train_loaders):
                data_ratio = len(loader.dataset)/total_samples            
                torch.manual_seed(42)
                model = Cifar10CNN()
                agents.append(
                    FedConsensus(
                        N=len(self.train_loaders),
                        delta=0,
                        rho=self.rho/(data_ratio*self.num_agents),
                        model=model,
                        loss=nn.CrossEntropyLoss(),
                        train_loader=loader,
                        epochs=self.epochs,
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
            torch.manual_seed(42)
            global_model = Cifar10CNN()
            server = EventADMM(clients=agents, t_max=self.t_max, rounds=self.t_max, model=global_model, device=self.device)
            server.spin(loader=self.test_loader, K_x=0, K_z=5, rate_ref=item)
            
            # For plotting purposes
            acc_per_item[i,:] = server.val_accs
            # load = sum(rate_per_item[i,:])/t_max
            acc = server.validate_global(loader=self.test_loader)
            # print(f'Load for {item_key} {item} = {load} | Test accuracy = {acc}')
            loads.append(server.load)
            test_accs.append(acc.cpu().numpy())

            """
            Save Results
            """
            
            # Save plotting data
            if not self.forward:
                np.save(file=f'figure_data/FedBack/rates_{int(item*100)}', arr=rate_per_item)
                np.save(file=f'figure_data/FedBack/accs_{int(item*100)}', arr=acc_per_item)
                np.save(file=f'figure_data/FedBack/loads_{int(item*100)}', arr=loads)

class FedADMMJob:

    def __init__(self, train_loaders: List[DataLoader], test_loader: DataLoader, val_loader: DataLoader,
                t_max: int, lr: float, device: str, num_agents: int, epochs: int, rate: float):
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.t_max = t_max
        self.lr = lr
        self.epochs = epochs
        self.rho = 0.01
        self.rate = rate
        self.device = device
        self.num_agents = num_agents

    def run(self):
        rates=[self.rate]
        print(rates)

        self.agents = []
        self.acc_per_rate = np.zeros((len(rates), self.t_max))
        self.rate_per_rate = np.zeros((len(rates), self.t_max))
        self.loads = []
        self.test_accs = []
        total_samples = sum([len(loader.dataset) for loader in self.train_loaders])

        for i, rate in enumerate(rates):
            for loader in self.train_loaders:
                torch.manual_seed(42)
                model = Cifar10CNN()
                self.agents.append(
                    FedADMM(
                        rho=self.rho,
                        N=self.num_agents,
                        model=model,
                        loss=nn.CrossEntropyLoss(),
                        train_loader=loader,
                        epochs=self.epochs,
                        device=self.device,
                        lr=self.lr,
                        data_ratio=len(loader.dataset)/total_samples
                    )
                )

            torch.manual_seed(42)
            model = Cifar10CNN()
            # k0 = 1 if rate == 1 else 2
            k0 = 1
            server = InexactADMM(clients=self.agents, C=rate, t_max=self.t_max,
                                 model=model, device=self.device, num_clients=self.num_agents, k0=k0)
            server.spin(loader=self.val_loader)

            self.acc_per_rate[i,:] = server.val_accs
            self.rate_per_rate[i,:] = server.rates
            load = sum(self.rate_per_rate[i,:])/self.t_max
            acc = server.validate_global(loader=self.test_loader)
            print(f'Test accuracy for rate {rate} = {acc}')
            self.loads.append(load)
            self.test_accs.append(acc.cpu().numpy())
        
            np.save(file='figure_data/FedADMM/accs_'+str(rate*100), arr=self.acc_per_rate)
            np.save(file='figure_data/FedADMM/loads_'+str(rate*100), arr=self.loads)
        
        print(f'saved: {self.acc_per_rate} and {self.loads}')


class FedLearnJob:

    def __init__(self, train_loaders: List[DataLoader], test_loader: DataLoader, val_loader: DataLoader,
                t_max: int, lr: float, prox: bool, device: str, epochs: int, rate: float):
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.t_max = t_max
        self.lr = lr
        self.rate = rate
        self.rho = 0.01 if prox else 0
        self.device = device
        self.prox = prox
        print(f'Prox: {self.prox}')

    def run(self):
        rates = np.arange(start=0.1, stop=0.6, step=0.1)
        rates = [*rates.tolist(), 1]
        rates=[self.rate]
        print(rates)

        self.agents = []
        self.acc_per_rate = np.zeros((len(rates), self.t_max))
        self.rate_per_rate = np.zeros((len(rates), self.t_max))
        self.loads = []
        self.test_accs = []
        image_dir = './images/FedProx/' if self.prox else './images/FedAVG/'
        figure_data_dir = './figure_data/FedProx/' if self.prox else './figure_data/FedAVG/'

        for i, rate in enumerate(rates):
            for loader in self.train_loaders:
                torch.manual_seed(42)
                model = Cifar10CNN()
                self.agents.append(
                    FedLearn(
                        rho=self.rho,
                        loss=nn.CrossEntropyLoss(),
                        model=model,
                        train_loader=loader,
                        epochs=self.epochs,
                        device=self.device,
                        lr=self.lr
                    )
                )
            
            torch.manual_seed(42)
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

            np.save(file=figure_data_dir+'accs_'+str(rate*100), arr=self.acc_per_rate)
            np.save(file=figure_data_dir+'loads_'+str(rate*100), arr=self.loads)

        print(f'saved: {self.acc_per_rate} and {self.loads}')