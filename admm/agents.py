import numpy as np
import torch.nn as nn
import torch
from admm.utils import *
from admm.models import FCNet
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict
import copy

class FedConsensus:

    """
    Distributed event-based ADMM for federated learning
    """
    
    def __init__(self, rho: int, N: int, delta: int, loss: nn.Module, model: nn.Module,
                 train_loader: DataLoader, epochs: int, device: str, 
                 lr: float, data_ratio: float, global_weight: float) -> None:        
        
        self.primal_avg = None
        self.device = device
        self.model = model.to(device)
        self.rho=rho
        self.N=N
        self.delta = delta
        self.broadcast = True
        self.recieve = True
        self.global_weight = global_weight
        self.lr = lr
        self.last_communicated_prime = self.copy_params(self.model.parameters())
        self.residual = self.copy_params(self.model.parameters())
        self.lam = [torch.zeros(param.shape).to(self.device) for param in self.model.parameters()]
        self.last_communicated_lam = [torch.zeros(param.shape).to(self.device) for param in self.model.parameters()]
        self.dual_residual = [torch.zeros(param.shape).to(self.device) for param in self.model.parameters()]
        # self.primal_avg = [torch.zeros(param.shape).to(self.device) for param in self.model.parameters()]
        # self.optimizer = torch.optim.NAdam(self.model.parameters(), self.lr)
        self.train_loader = train_loader
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = loss
        self.epochs = epochs
        self.data_ratio = data_ratio
        # Get number of params in model
        self.total_params = sum(param.numel() for param in self.model.parameters())
        self.stepper = StepLR(optimizer=self.optimizer, gamma=0.98, step_size=1)
        self.full_loader = DataLoader(train_loader.dataset, batch_size=len(train_loader.dataset))
        self.local_seq = [0]
        self.global_seq = [0]

    def primal_update(self, round, params) -> None:
        
        self.primal_avg = self.copy_params(params)
        if round > 0: self.dual_update()
        
        # local_grad, local_loss = self.Lagrangian(use_global=False, params=params)
        # global_grad, global_loss = self.Lagrangian(use_global=True, params=params)
        # print(f'local diff {local_loss}, global diff: {global_loss}')
        # if local_loss >= global_loss: using_global: int = 1
        # else: using_global: int = 0
        
        using_global = 1
        # Solve argmin problem
        if using_global == 1: self.model = self.set_parameters(model=self.model, parameters=params)
        # self.model = self.set_parameters(model=self.model, parameters=params)
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
        # self.stepper.step()
        # check for how much paramters changed
        delta_prime = 0
        for old_param, updated_param, dual_param in zip(self.last_communicated_prime, self.model.parameters(), self.lam):
            with torch.no_grad():
                delta_prime += torch.norm(old_param.data - updated_param.data - dual_param.data, p='fro').item()**2
        delta_prime = np.sqrt(delta_prime)

        # If "send on delta" then update residual and broadcast to other agents
        if delta_prime >= self.delta:      
            self.update_residual()
            self.last_communicated_prime = self.copy_params(self.model.parameters())
            add_params(self.last_communicated_prime, self.lam)
            self.broadcast = True
        else:
            self.broadcast = False

        return delta_prime, using_global
    
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
    Distributed event-based ADMM for federated learning
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
        # self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.train_loader = train_loader
        self.criterion = loss
        self.epochs = epochs
        self.data_ratio = data_ratio

    def update(self, global_params) -> None:
        # Solve argmin problem
        self.primal_avg = self.copy_params(global_params)
        self.dual_update()
        using_global=True
        if using_global: self.model = self.set_parameters(parameters=self.copy_params(global_params), model=self.model)
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
        # scale_params(self.residual, a=1/self.N)

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
    
class Agent:

    def __init__(self, rho, x_init=None, lam_init=None, z_init=None, nu_init=None) -> None:
        self.rho = rho
        self.x = x_init
        self.lam = lam_init
        self.z = z_init
        self.nu = nu_init
    
    def primal_update(self) -> None:
        raise NotImplementedError('Calling from base class "Agent"')

    def auxillary_update(self) -> None:
        raise NotImplementedError('Calling from base class "Agent"')

    def dual_update(self) -> None:
        raise NotImplementedError('Calling from base class "Agent"')
    

class GlobalConsensus(Agent):

    """
        f(x_i) = x_i ^ 2
        s.t  x_i = z
    """

    def __init__(self, rho, x_init, lam_init=None, z_init=None, nu_init=None) -> None:
        super().__init__(rho, x_init, lam_init, z_init, nu_init)
        self.primal_avg = 0

    def primal_update(self) -> None:
        self.x = (self.rho*self.primal_avg - self.lam)/(2 + self.rho)
    
    def dual_update(self) -> None:
        self.lam = self.lam + self.rho*(self.x - self.primal_avg)


class EventGlobalConsensus(GlobalConsensus):

    """
        f(x_i) = x_i ^ 2
        s.t  x_i = z
    """

    def __init__(self, rho, N, delta, x_init=None, lam_init=None, z_init=None, nu_init=None) -> None:
        super().__init__(rho, x_init, lam_init, z_init, nu_init)
        self.N = N
        self.delta = delta
        self.broadcast = False
        self.C = 0
        self.last_communicated = self.x
        self.residual = 0

    def primal_update(self) -> None:
        
        ######## replace with SGD torch
        self.x = (self.rho*self.primal_avg - self.lam + 2*self.C)/(2+self.rho)
        ########
    
        if np.linalg.norm(self.x-self.last_communicated, ord=2) >= self.delta: 
            self.residual = (self.x - self.last_communicated)/self.N
            self.last_communicated = self.x
            self.broadcast = True
        else:
            self.broadcast = False
    
    def dual_update(self) -> None:
        self.lam = self.lam + self.rho*(self.x - self.primal_avg)