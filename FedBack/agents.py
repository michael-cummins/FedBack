import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict
import copy

from fedback.utils import *

class FedConsensus:

    """
    FedBack client routine
    """
    
    def __init__(self, rho: int, N: int, loss: nn.Module, model: nn.Module,
                 train_loader: DataLoader, epochs: int, device: str, 
                 lr: float, data_ratio: float, global_weight: float) -> None:        
        
        self.primal_avg = None
        self.device = device
        self.model = model.to(device)
        self.rho=rho
        self.N=N
        self.broadcast = True
        self.recieve = True
        self.global_weight = global_weight
        self.lr = lr
        self.last_communicated_prime = self.copy_params(self.model.parameters())
        self.residual = self.copy_params(self.model.parameters())
        self.lam = [torch.zeros(param.shape).to(self.device) for param in self.model.parameters()]
        self.last_communicated_lam = [torch.zeros(param.shape).to(self.device) for param in self.model.parameters()]
        self.dual_residual = [torch.zeros(param.shape).to(self.device) for param in self.model.parameters()]

        self.train_loader = train_loader
        try:
            net = self.model.network
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = loss
        self.epochs = epochs
        self.data_ratio = data_ratio
        
        # Get number of params in model
        self.total_params = sum(param.numel() for param in self.model.parameters())
        self.full_loader = DataLoader(train_loader.dataset, batch_size=len(train_loader.dataset))
        self.local_seq = [0]
        self.global_seq = [0]

    def primal_update(self, round, params) -> None:
        
        self.primal_avg = self.copy_params(params)
        if round > 0: self.dual_update()
        
        # Solve argmin problem
        self.model = self.set_parameters(model=self.model, parameters=params)
        for _ in range(self.epochs):
            for i, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
                prox = 0.0
                for param, dual_param, avg in zip(self.model.parameters(), self.lam, self.primal_avg):
                    prox += torch.norm(param - avg.data + dual_param.data, p='fro')**2
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    pred = self.model(data)
                    loss = self.criterion(pred, target) + prox*self.rho/2         
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
        
        self.dual_update()
        self.update_residual()
        self.last_communicated_prime = self.copy_params(self.model.parameters())
        add_params(self.last_communicated_prime, self.lam)
        self.broadcast = True
    
    def dual_update(self) -> None:  
        primal_copy = self.copy_params(self.model.parameters())
        subtract_params(primal_copy, self.primal_avg)
        add_params(self.lam, primal_copy)

    def Lagrangian(self, use_global: bool, params):
       
        model = copy.deepcopy(self.model)
        if use_global: model = self.set_parameters(params, model)        
        
        for data, target in self.full_loader:
            data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
            prox = 0.0
            for param, dual_param, avg in zip(self.model.parameters(), self.lam, self.primal_avg):
                prox += torch.norm(param - avg.data + dual_param.data, p='fro')**2
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                pred = model(data)
                lagrangian = self.criterion(pred, target) + prox*self.rho/2
            break
        
        model.zero_grad()
        lagrangian.backward()
        norm_grad = 0
        for param in model.parameters():
            norm_grad += torch.norm(param.grad, p='fro').item()**2

        return norm_grad, lagrangian.item()
        
    def update_residual(self):
        # Current local z-value
        self.residual = self.copy_params(self.model.parameters())
        add_params(self.residual, self.lam)
        subtract_params(self.residual, self.last_communicated_prime)
        # scale_params(self.residual, a=self.global_weight)

    def copy_params(self, params):
        copy = [torch.zeros(param.shape).to(self.device).copy_(param) for param in params]
        return copy
    
    def get_parameters(self, model):
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters, model: nn.Module) -> nn.Module:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        if params_dict is List[torch.Tensor]: 
            state_dict = OrderedDict({k: v for k, v in params_dict})
        else:
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

class FedLearn:

    """
    Client side routine for FedAVG and FedProx
    """
    
    def __init__(self, loss: nn.Module, model: nn.Module, train_loader: DataLoader, 
                 rho: float, epochs: int, device: str, lr: float) -> None:        
        
        self.device = device
        self.rho = rho
        self.model = model.to(device)
        self.lr = lr
        self.num_samples = len(train_loader.dataset)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.train_loader = train_loader
        self.criterion = loss
        self.epochs = epochs

    def primal_update(self, global_params) -> None:
        
        self.model = self.set_parameters(global_params, model=self.model)
        global_copy = self.copy_params(self.model.parameters())

        # Solve argmin problem
        for _ in range(self.epochs):
            for i, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
                prox = 0.0
                if self.rho > 0:
                    for param, global_param in zip(self.model.parameters(), global_copy):
                        prox += torch.norm(param - global_param.data, p='fro')**2
                loss = self.criterion(self.model(data), target) + prox*self.rho/2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 

    def copy_params(self, params):
        copy = [torch.zeros(param.shape).to(self.device).copy_(param) for param in params]
        return copy

    def set_parameters(self, parameters, model: nn.Module) -> nn.Module:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        if params_dict is List[torch.Tensor]: 
            state_dict = OrderedDict({k: v for k, v in params_dict})
        else:
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model


class FedADMM:

    def __init__(self, rho: int, N: int, loss: nn.Module, model: nn.Module,
                 train_loader: DataLoader, epochs: int, device: str, 
                 lr: float, data_ratio: float) -> None: 
        
        self.primal_avg = None
        self.device = device
        self.model = model.to(device)
        self.rho=rho
        self.N=N
        self.broadcast = False
        self.lr = lr
        self.last_communicated = self.copy_params(self.model.parameters())
        self.residual = self.copy_params(self.model.parameters())
        self.lam = [torch.zeros(param.shape).to(self.device) for param in self.model.parameters()]
        try:
            net = self.model.network
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loader = train_loader
        self.criterion = loss
        self.epochs = epochs
        self.data_ratio = data_ratio

    def update(self, global_params) -> None:
        
        # Solve argmin problem
        self.primal_avg = self.copy_params(global_params)
        self.dual_update()
        
        self.model = self.set_parameters(parameters=self.copy_params(global_params), model=self.model)
        for epoch in range(self.epochs):
            for i, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
                prox = 0.0
                for param, dual_param, avg in zip(self.model.parameters(), self.lam, self.primal_avg):
                    prox += torch.norm(param - avg.data + dual_param.data, p='fro')**2
                loss = self.criterion(self.model(data), target) + prox*self.rho/2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
        
        self.update_residual()

    def dual_update(self) -> None:  
        primal_copy = self.copy_params(self.model.parameters())
        subtract_params(primal_copy, self.primal_avg)
        add_params(self.lam, primal_copy)
    
    def update_residual(self):
        # Current local z-value
        self.residual = self.copy_params(self.model.parameters())
        add_params(self.residual, self.lam)

    def copy_params(self, params):
        copy = [torch.zeros(param.shape).to(self.device).copy_(param) for param in params]
        return copy
    
    def set_parameters(self, parameters, model: nn.Module) -> nn.Module:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        if params_dict is List[torch.Tensor]: 
            state_dict = OrderedDict({k: v for k, v in params_dict})
        else:
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model