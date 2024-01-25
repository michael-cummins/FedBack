from typing import Dict, Tuple
import flwr as fl
from flwr.common import NDArrays, Scalar
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from admm.utils import *
import torch
import torch.nn as nn
from admm.models import CNN
from collections import OrderedDict
import numpy as np

class FlowerCLient(fl.client.NumPyClient):
    
    def __init__(self, rho: int, delta: int, loss: nn.Module, model: nn.Module,
                 train_loader: DataLoader, epochs: int, device: str, 
                 lr: float, data_ratio: float) -> None:
        super().__init__()

        self.train_loader = train_loader
        self.device = device
        self.model = model.to(self.device)
        self.rho = rho
        self.delta = delta
        self.loss = loss
        self.epochs = epochs
        self.lr = lr
        self.data_ratio = data_ratio
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.stepper = StepLR(optimizer=self.optimizer, gamma=0.975, step_size=1)

        self.last_communicated = self.copy_params(self.model.parameters())
        self.residual = self.copy_params(self.model.parameters())
        self.lam = [torch.zeros(param.shape).to(self.device) for param in self.model.parameters()]
        self.train_length = len(self.train_loader)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict=state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
       
        # copy params into client's local model
        if config["curr_round"] > 0: self.dual_update()
        self.primal_avg = self.copy_params(parameters)

        # Train model
        # Solve argmin problem
        for _ in range(self.epochs):
            for i, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
                prox = 0.0
                for param, dual_param, avg in zip(self.model.parameters(), self.lam, self.primal_avg):
                    prox += torch.norm(param - avg.data + dual_param.data, p='fro')**2
                pred = self.model(data)
                loss = self.criterion(pred, target) + prox*self.rho/2         
                     
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        delta = 0
        for old_param, updated_param, dual_param in zip(self.last_communicated, self.model.parameters(), self.lam):
            with torch.no_grad():
                delta += torch.norm(old_param.data-updated_param.data-dual_param.data,p='fro').item()**2
        d = np.sqrt(delta)

        # If "send on delta" then update residual and broadcast to other agents
        if d >= self.delta:      
            self.update_residual()
            self.last_communicated = self.copy_params(self.model.parameters())
            add_params(self.last_communicated, self.lam)
            self.broadcast = True
            contribution = self.residual
        else:
            self.broadcast = False
            contribution = None

        return contribution, self.train_length, {'broadcast': self.broadcast, 'delta': d}
    
    def dual_update(self) -> None:  
        primal_copy = self.copy_params(self.model.parameters())
        subtract_params(primal_copy, self.primal_avg)
        add_params(self.lam, primal_copy)

    def update_residual(self):
        # Current local z-value
        self.residual = self.copy_params(self.model.parameters())
        add_params(self.residual, self.lam)
        subtract_params(self.residual, self.last_communicated)
        # scale_params(self.residual, a=self.rho/(self.N*self.rho - 2*0.0001))
        # scale_params(self.residual, a=self.global_weight)

    def copy_params(self, params):
        copy = [torch.zeros(param.shape).to(self.device).copy_(param) for param in params]
        return copy