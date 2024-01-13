import torch
from typing import List
from admm.utils import sum_params, add_params, sublist_by_fraction
from admm import agents
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import numpy as np
from admm.models import FCNet
from collections import OrderedDict
import statistics

class ServerBase:

    def __init__(self, t_max: int, model: torch.nn.Module, device: str) -> None:
        self.t_max = t_max
        self.pbar = tqdm(range(t_max))
        self.device = device
        self.global_model = model
    
    def set_parameters(self, parameters, model: torch.nn.Module) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    def get_parameters(self):
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]
    
    def validate_global(self, loader: DataLoader) -> float:
        wrong_count = 0
        total = len(loader.dataset)
        for data, target in loader:
            data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
            out = torch.argmax(self.global_model(data), dim=1)
            wrong_count += torch.count_nonzero(out-target)
        global_acc = 1 - wrong_count/total
        return global_acc

class FedAgg(ServerBase):
    def __init__(self, clients: List[agents.Fed], C: float, t_max: int, model: torch.nn.Module, device: str) -> None:
        super().__init__(t_max, model, device)
        self.agents = clients
        self.comm = 0
        self.C = C
        self.N = len(self.agents)
        # For experiment purposes
        self.rates = []
        self.val_accs = []

    def spin(self, loader=None) -> None:
        for _ in self.pbar:
            
            # Sample subset of agents
            sampled_agents = sublist_by_fraction(agents=self.agents, fraction=self.C)
            global_params = self.get_parameters()
            self.comm += len(sampled_agents)

            # Send params to clients and let them train
            m_t = 0
            weighted_local_params = []
            for agent in sampled_agents:
                agent.primal_update(global_params)
                m_t += agent.num_samples

            # Aggregate the new params
            for agent in sampled_agents:
                weighted_local_params.append([param*agent.num_samples/m_t for param in agent.model.parameters()])
            global_params = sum_params(weighted_local_params)
            self.global_model = self.set_parameters(global_params, self.global_model)

            # Validate global model
            acc_descrption = ''
            if loader is not None:
                with torch.no_grad():
                    global_acc = self.validate_global(loader=loader)
                    acc_descrption += f', Global Acc = {global_acc:.4f}'

             # Analyse communication frequency
            freq = self.comm/(self.N)
            self.comm = 0
            self.pbar.set_description(f'Comm: {freq:.3f}' + acc_descrption)

            # For experiment purposes
            self.rates.append(freq)
            self.val_accs.append(global_acc)
        
        self.rates = np.array(self.rates)
        self.val_accs = np.array(self.val_accs)

class EventADMM(ServerBase):

    def __init__(self, clients: List[agents.FedConsensus], t_max: int, model: torch.nn.Module, device: str) -> None:
        super().__init__(t_max, model, device)
        self.agents = clients
        self.comm = 0
        self.N = len(self.agents)
        
        # For experiment purposes
        self.rates = []
        self.val_accs = []
        self.global_model = model

    def spin(self, loader=None) -> None:
        for _ in self.pbar:
    
            # Primal Update
            D = []
            for i, agent in enumerate(self.agents):
                d = agent.primal_update()
                D.append(d)
            delta_description = f', min Delta: {min(D):.8f}, max Delta: {max(D):.8f}, avg: {statistics.median(D):.8f}'

            # Residual update in the case of communication
            C = []
            for agent in self.agents:
                if agent.broadcast: 
                    self.comm += 1
                    C.append(agent.residual)
            if C:
                # If communicaiton set isn't empty
                residuals = [x for x in sum_params(C)]
                for agent in self.agents:
                    add_params(agent.primal_avg, residuals)
            
            # Dual update
            for agent in self.agents:
                agent.dual_update()

            # Test updated params on validation set
            acc_descrption = ''
            if loader is not None:
                # Get gloabl variable Z and copy to a network for validation
                with torch.no_grad():
                    global_params = [param.cpu().numpy() for param in self.agents[0].primal_avg]
                    self.global_model = self.set_parameters(global_params, self.global_model)
                    global_acc = self.validate_global(loader=loader)
                    acc_descrption += f', Global Acc = {global_acc:.4f}'

            # Analyse communication frequency
            freq = self.comm/(self.N)
            self.comm = 0
            self.pbar.set_description(f'Comm: {freq:.3f}' + acc_descrption + delta_description)

            # For experiment purposes
            self.rates.append(freq)
            self.val_accs.append(global_acc.detach().cpu().numpy())


        self.rates = np.array(self.rates)
        self.val_accs = np.array(self.val_accs)

    def validate_agents(self, loader: DataLoader) -> List[float]:
        total = 0
        wrong_count = np.zeros(self.N)
        for data, target in loader:
            total += target.shape[0] 
            with torch.no_grad():
                for i, agent in enumerate(self.agents):
                    data, target = data.to(agent.device), target.type(torch.LongTensor).to(agent.device)
                    out = torch.argmax(agent.model(data), dim=1)
                    wrong_count[i] += torch.count_nonzero(out-target)
        model_accs = [1 - wrong/total for wrong in wrong_count]
        return model_accs
