import torch
from typing import List
from admm.utils import sum_params, add_params
from admm import agents
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import numpy as np

class EventADMM:

    def __init__(self, clients: List[agents.FedConsensus], t_max: int) -> None:
        self.agents = clients
        self.t_max = t_max
        self.pbar = tqdm(range(t_max))
        self.comm = 0
        self.N = len(self.agents)
        
        # For experiment purposes
        self.rates = []
        self.val_accs = []

    def spin(self, loader=None) -> None:
        overfit = False
        for t in self.pbar:
    
            # Primal Update
            D = []
            for agent in self.agents:
                d = agent.primal_update()
                D.append(d)
            delta_description = f', min Delta: {min(D):.8f}, max Delta: {max(D):.8f}, avg: {(sum(D)/len(D)):.8f}'
           
            # Test updated params on validation set
            acc_descrption = ''
            if loader is not None:
                accuracies = self.validate(loader=loader)
                avg_acc = sum(accuracies)/len(accuracies)
                acc_descrption = f', Min: {min(accuracies):.4f}, Max: {max(accuracies):.4f}, Avg Acc: {avg_acc:.4f}'

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
            self.pbar.set_description(f'Comm: {freq:.3f}' + acc_descrption + delta_description)
            
            # Dual update
            for agent in self.agents:
                agent.dual_update()

            # For experiment purposes
            self.rates.append(freq)
            self.val_accs.append(sum(accuracies)/len(accuracies))

        self.rates = np.array(self.rates)
        self.val_accs = np.array(self.val_accs)

    def validate(self, loader: DataLoader) -> List[float]:
        total = 0
        wrong_count = np.zeros(self.N)
        for data, target in loader:
            total += target.shape[0] 
            with torch.no_grad():
                for i, agent in enumerate(self.agents):
                    data, target = data.to(agent.device), target.to(agent.device)
                    out = torch.argmax(agent.model(data), dim=1)
                    wrong_count[i] += torch.count_nonzero(out-target)
        model_accs = [1 - wrong/total for wrong in wrong_count]
        return model_accs