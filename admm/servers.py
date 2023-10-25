import torch
from typing import Tuple, List
from admm.utils import sum_params, add_params
from admm import agents
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import numpy as np

class EventADMM:

    def __init__(self, clients, t_max: int) -> None:
        self.agents = clients
        self.t_max = t_max
        self.pbar = tqdm(range(t_max))
        self.comm = 0
        self.N = len(self.agents)

    def spin(self, loader=None) -> None:
        
        for t in self.pbar:
    
            # Primal Update
            for agent in self.agents:
                agent.primal_update()
            
            # Test updated params on validation set
            acc_descrption = ''
            if loader is not None:
                accuracies = self.validate(loader=loader)
                for i, acc in enumerate(accuracies):
                    acc_descrption += f', agent {i}: {acc:.2f}'

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
            
            # Analyse communication frequency
            freq = self.comm/((t+1)*self.N)
            self.pbar.set_description(f'Comm frequency: {freq:.3f}' + acc_descrption)
            
            # Dual update
            for agent in self.agents:
                agent.dual_update()
    
    def validate(self, loader: DataLoader) -> List[float]:
        total = 0
        wrong_count = np.zeros(self.N)
        for data, target in loader:
            total += target.shape[0] 
            with torch.no_grad():
                data = data.reshape(-1, 28*28)
                for i, agent in enumerate(self.agents):
                    out = torch.argmax(agent.model(data), dim=1)
                    wrong_count[i] += torch.count_nonzero(out-target)
        model_accs = [1 - wrong/total for wrong in wrong_count]
        return model_accs