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
    
