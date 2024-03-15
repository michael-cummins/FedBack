from typing import Dict, Tuple
import flwr as fl
from flwr.common import NDArrays, Scalar
import torch
import torch.nn as nn
from admm.models import CNN
from collections import OrderedDict

class FlowerCLient(fl.client.NumPyClient):
    
    def __init__(self, train_loader, val_loader) -> None:
        super().__init__()

        self.train_loader = train_loader
        self.val_loader = val_loader 
        self.model = CNN()
        self.device = 'cpu'
        self.loss = nn.CrossEntropyLoss()

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict=state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        
        # Test model
        self.model.eval()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                out = self.model(data)
                loss += self.loss(out, target)
                _, pred = torch.max(out.data, 1)
                correct += (pred == target).sum().item()
        acc = correct/len(self.val_loader.dataset)
        return float(loss), len(self.val_loader), {'accuracy': acc}

    def fit(self, parameters, config):
       
        # copy params into client's local model
        self.set_parameters(parameters)
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['epochs']
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.model.to(self.device)

        # Train model
        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                out = self.model(data)
                loss = self.loss(out, target)
                optim.zero_grad()
                loss.backward()
                optim.step()

        return self.get_parameters({}), len(self.train_loader), {}
    
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
                loss = self.criterion(self.model(data), target) + prox*self.rho/(2*self.data_ratio)
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