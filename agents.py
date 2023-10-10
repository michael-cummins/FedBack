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
        self.residual = (self.x - self.last_communicated)/self.N

        if np.linalg.norm(self.x-self.last_communicated, ord=2) >= self.delta: 
            self.last_communicated = self.x
            self.broadcast = True
        else:
            self.broadcast = False


class EventGlobalConsensusTorch(EventGlobalConsensus):

    def __init__(self, rho : int, N : int, delta : int, model : nn.Module,
                 loss, x_init=None, lam_init=None, z_init=None, nu_init=None) -> None:
        super().__init__(rho, x_init, N, delta, lam_init, z_init, nu_init)

        """
        lam_init is a generator containting dual parameters, not a model -> similar to model.parameters()
        x_init is a generator containting dual parameters, not a model -> similar to model.parameters()
        If a model is passed - the parameters should be initialised within the model rather than passing x_init
        self.primal_avg is also represented as a generator containing model parameters
        """

        self.lr = 0.1
        self.model = model()
        self.loss = loss
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, weight_decay=self.lr*self.rho)
        self.last_communicated = [param for param in self.model.parameters()]

    def primal_update(self) -> None:
        
        old_params = [param for param in self.model.paramaters()]
        
        # SGD with weight decay
        for _ in range(100):
            self.optimizer.zero_grad()
            self.loss(self.model(), p=2).backward()
            self.optimizer.step() 
            add_params(self.model.parameters(), scale_params())

        # calculate residual
        self.residual = [(param - last_comm)/self.N for param, last_comm in zip(self.model.parameters(), self.last_communicated)]

        # check for how much paramters changed
        delta = 0
        for old_param, updated_param in zip(old_params, self.model.parameters()):
            delta += torch.norm(old_param-updated_param, p='fro')
        
        if delta >= self.delta:
            self.last_communicated = [param for param in self.model.parameters()]
            self.broadcast = True
        else:
            self.broadcast = False

        
