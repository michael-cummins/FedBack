import numpy as np
import torch.nn as nn
import torch
from utils import *
from models import Dummy

class Agent():

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

class EventGlobalConsensusTorch:

    def __init__(self, rho : int, N : int, delta : int, model : nn.Module) -> None:

        """
        x_init/lam_init are generators containting primal/dual parameters, not a model -> similar to model.parameters()
        If a model is passed - the parameters should be initialised within the model rather than passing x_init
        self.primal_avg and self.resiudal are also represented as generators containing model parameters
        """
        self.primal_avg = None
        self.rho=rho
        self.N=N
        self.delta = delta
        self.lr = 0.001
        self.max_iters = 1000
        self.model = model
        self.last_communicated = self.copy_params(self.model.parameters())
        self.residual = self.copy_params(self.model.parameters())
        self.lam = [torch.zeros(param.shape) for param in self.model.parameters()]
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), self.lr)

    def primal_update(self) -> None:

        # SGD with weight decay
        loss = torch.Tensor([np.inf])
        prev_loss = torch.zeros(loss.shape)
        iter = 0
        while np.abs(loss.item() - prev_loss.item()) >= 1e-4 and iter <= self.max_iters:
            prev_loss = loss
            self.optimizer.zero_grad()
            loss = 0
            for param, dual_param, avg in zip(self.model.parameters(), self.lam, self.primal_avg):
                loss += torch.norm(param - avg.data + dual_param.data/self.rho, p=2)**2
            for param in self.model.parameters():
                loss += torch.norm(param-2*torch.ones(param.shape))
            loss.backward()
            self.optimizer.step() 
            iter += 1

        # check for how much paramters changed
        delta = []
        for old_param, updated_param in zip(self.last_communicated, self.model.parameters()):
            delta.append(torch.norm(old_param.data-updated_param.data).item())
        
        # If "send on delta" then update residual and broadcast to other agents
        if any(d >= self.delta for d in delta):       
            self.update_residual()
            self.last_communicated = self.copy_params(self.model.parameters())
            self.broadcast = True
        else:
            self.broadcast = False

    def dual_update(self) -> None:  
        primal_copy = self.copy_params(self.model.parameters())
        subtract_params(primal_copy, self.primal_avg)
        scale_params(primal_copy, a=self.rho)
        add_params(self.lam, primal_copy)

    def update_residual(self):
        self.residual = self.copy_params(self.model.parameters())
        subtract_params(self.residual, self.last_communicated)
        scale_params(self.residual, a=1/self.N)

    def copy_params(self, params):
        copy = [torch.zeros(param.shape).copy_(param) for param in params]
        return copy