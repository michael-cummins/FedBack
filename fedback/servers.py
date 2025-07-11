import torch
from typing import List
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import numpy as np
from collections import OrderedDict
import statistics
import copy

from fedback.utils import sublist_by_fraction
from fedback.utils import sum_params, add_params, average_params
from fedback import agents



class EventADMM:
    """
    Server-side routine for FedBack
    """

    def __init__(self, clients: List[agents.FedConsensus], t_max: int, rounds: int, model: torch.nn.Module, device: str) -> None:
        self.agents = clients
        self.t_max = t_max
        self.rounds = rounds
        self.pbar = tqdm(range(t_max))
        self.comm = 0
        self.N = len(self.agents)
        self.device = device
        
        # For experiment purposes
        self.rates = []
        self.val_accs = []
        self.global_model = model.to(self.device)
        self.global_params = self.get_parameters(self.global_model)
        self.last_communicated = [torch.zeros(param.shape).to(self.device).copy_(param) for param in self.global_params]
        self.local_res = [self.get_parameters(self.global_model) for _ in range(self.N)]
        self.cumm_global = 0
        self.delta_z = np.zeros(self.N)
    
    def spin(self, K, rate_ref, loader=None) -> None:

        global_comm = []
        local_comm = []

        alpha = 0.9
        p_meas = np.zeros(self.N)
        global_freq = np.ones(self.N)

        for round in self.pbar:
            
            global_comm.append(sum(global_freq)/len(global_freq))
            
            # Primal Update
            for i, agent in enumerate(self.agents):
                if agent.recieve:
                    agent.primal_update(round, params=self.global_params)
                    agent.recieve = False
            if self.device == 'cuda': torch.cuda.synchronize()
                
            # Residual update in the case of communication
            comm_list=[]
            self.comm = 0
            for i, agent in enumerate(self.agents):
                if agent.broadcast: 
                    comm_list.append(i)
                    self.comm += 1
                    add_params(self.local_res[i], agent.residual)
                    agent.broadcast = False
            
            # Aggregate parameters from clients
            self.global_params = average_params(self.local_res)
            local_freq = self.comm/self.N
            if self.device == 'cuda': torch.cuda.synchronize()
            local_comm.append(local_freq)

            # Calculating delta based on new server parameters
            global_freq = np.zeros(self.N)
            for i, res in enumerate(self.local_res):
                d_z = 0
                for old_z, new_z in zip(res, self.global_params):
                    with torch.no_grad():
                        d_z += torch.norm(new_z - old_z, p='fro').item()**2
                d_z = np.sqrt(d_z)
                if d_z >= self.delta_z[i]: 
                    # Since client i meets the threshold, they participate in the next round
                    self.agents[i].recieve=True
                    global_freq[i] = 1
                else: self.agents[i].recieve=False
                        
            # Test updated params on validation set
            if loader is not None:
                # Get gloabl variable Z and copy to a network for validation
                with torch.no_grad():
                    self.global_model = self.set_parameters(self.global_params, self.global_model)
                    global_acc = self.validate_global(loader=loader)
                    acc_descrption = f'{global_acc:.4f}'
            if self.device == 'cuda': torch.cuda.synchronize()
            self.pbar.set_description(f'Accuracy: {acc_descrption}')
            self.val_accs.append(global_acc.detach().cpu().numpy())
            
            # Update delta for each client
            p_meas = (1-alpha)*p_meas + alpha*global_freq
            self.rates.append(np.mean(global_freq))
            self.delta_z = self.delta_z + K*(p_meas - rate_ref*np.ones(p_meas.shape))
            for i, dz in enumerate(self.delta_z):
                if dz <=0: self.delta_z[i] = 0
            
            gc = [(g_elem + l_elem)/2 for g_elem, l_elem in zip(global_comm, local_comm) if g_elem != 0]
            if len(gc) >= self.rounds: break
        
        self.load = statistics.mean(gc)
        self.val_accs = np.array(self.val_accs)
        print(f'Total communication load = {self.load}, alpha = {alpha}, computed from rates = {np.sum(self.rates)/(round+1)}')

    def set_parameters(self, parameters, model: torch.nn.Module) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model
    
    def get_parameters(self, model):
        """Return the parameters of the current net."""
        model = copy.deepcopy(model)
        copied_params = [torch.zeros(param.shape).to(self.device).copy_(param) for param in model.parameters()]
        return copied_params
    
    def validate_global(self, loader: DataLoader) -> float:
        wrong_count = 0
        total = len(loader.dataset)
        for data, target in loader:
            data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
            out = torch.argmax(self.global_model(data), dim=1)
            wrong_count += torch.count_nonzero(out-target)
        global_acc = 1 - wrong_count/total
        return global_acc

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
    

class ServerBase:

    def __init__(self, t_max: int, model: torch.nn.Module, device: str) -> None:
        self.t_max = t_max
        self.pbar = tqdm(range(t_max))
        self.device = device
        self.global_model = model.to(device)
    
    def set_parameters(self, parameters, model: torch.nn.Module) -> None:
        """Change the parameters of the model using the given ones."""
        model_copy = copy.deepcopy(model)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model_copy.load_state_dict(state_dict, strict=True)
        return model_copy

    def get_parameters(self, model):
        """Return the parameters of the current net."""
        model = copy.deepcopy(model)
        copied_params = [torch.zeros(param.shape).to(self.device).copy_(param) for param in model.parameters()]
        return copied_params
    
    def validate_global(self, loader: DataLoader) -> float:
        wrong_count = 0
        total = len(loader.dataset)
        for data, target in loader:
            data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
            out = torch.argmax(self.global_model(data), dim=1)
            wrong_count += torch.count_nonzero(out-target)
        global_acc = 1 - wrong_count/total
        return global_acc
    

class InexactADMM(ServerBase):

    """
    Server-side routine for Inexact ADMM
    """
    
    def __init__(self, clients: List[agents.FedADMM], C: float, t_max: int, 
                model: torch.nn.Module, device: str, num_clients: int, k0: int) -> None:
        super().__init__(t_max, model, device)
        self.agents = clients
        self.comm = 0
        self.C = C
        self.N = len(self.agents)
        self.k0 = k0
        # For experiment purposes
        self.rates = []
        self.val_accs = []
        self.num_clients = num_clients

    def spin(self, loader=None) -> None:
        sampled_agents = self.agents
        for i in self.pbar:
            
            # Collect params from sublist of clients 
            if i%self.k0==0:
                # Sample subset of agents
                sampled_agents = sublist_by_fraction(agents=self.agents, fraction=self.C)
                self.comm += len(sampled_agents)
                global_params = average_params([agent.residual for agent in self.agents])

            for clients in sampled_agents:
                clients.update(global_params=global_params)

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
            self.val_accs.append(global_acc.detach().cpu().numpy())
        
        self.rates = np.array(self.rates)
        self.val_accs = np.array(self.val_accs)


class FedAgg(ServerBase):

    """
    Server-side routine for FedAvg And FedProx
    """
    
    def __init__(self, clients: List[agents.FedLearn], C: float, t_max: int,
                  model: torch.nn.Module, device: str) -> None:
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
            global_params = self.get_parameters(self.global_model)
            self.comm = len(sampled_agents)

            # Send params to clients and let them train
            m_t = 0
            weighted_local_params = []
            for agent in sampled_agents:
                agent.primal_update(global_params)
                m_t += agent.num_samples
            if self.device == 'cuda': torch.cuda.synchronize()
            
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
            self.pbar.set_description(f'Comm: {freq:.3f}' + acc_descrption)

            # For experiment purposes
            self.rates.append(freq)
            self.val_accs.append(global_acc.detach().cpu().numpy())
        
        self.rates = np.array(self.rates)
        self.val_accs = np.array(self.val_accs)