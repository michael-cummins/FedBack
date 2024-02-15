import torch
from typing import List
from admm.utils import sum_params, add_params, average_params, scale_params
from admm import agents
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import numpy as np
from collections import OrderedDict
import statistics
import subprocess
import re
from admm.utils import sublist_by_fraction
import math
import copy


class EventADMM:

    def __init__(self, clients: List[agents.FedConsensus], t_max: int, model: torch.nn.Module, device: str) -> None:
        self.agents = clients
        self.t_max = t_max
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
    
    def spin(self, K_x, K_z, rate_ref, loader=None) -> None:

        adaptive_delta=True
        delta = self.agents[0].delta
        global_comm = []
        local_comm = []
        integral = 0
        window_length = 10
        train = True
        global_freq = 1
        alpha = 0.9
        p_meas = np.zeros(self.N)
        global_freq = np.ones(self.N)
        # p_meas = 0

        for round in self.pbar:

            global_desc = ''
            for elem in global_freq:
                global_desc += str(int(elem))
            
            global_comm.append(sum(global_freq)/len(global_freq))
            
            # Primal Update
            D = []
            delta_description=', '
            for i, agent in enumerate(self.agents):
                if agent.recieve:
                    d , global_indicator = agent.primal_update(round, params=self.global_params)
                    agent.recieve=False
                else: 
                    d = 0
                D.append(d)            
            delta_description += f', d_x:{delta:.1f}, min: {min(D):.2f}, max: {max(D):.2f}, med: {statistics.median(D):.2f}'
            if self.device == 'cuda': torch.cuda.synchronize()
                
            # Residual update in the case of communication
            comm_list=[]
            self.comm = 0
            for i, agent in enumerate(self.agents):
                if agent.broadcast: 
                    comm_list.append(i)
                    self.comm += 1
                    add_params(self.local_res[i], agent.residual)
                    agent.broadcast=False
            
            self.global_params = average_params(self.local_res)
            local_freq = self.comm/self.N
            if self.device == 'cuda': torch.cuda.synchronize()
            local_comm.append(local_freq)

            # Calculating delta_z
            global_freq = np.zeros(self.N)
            # global_freq = 0
            for i, res in enumerate(self.local_res):
                d_z = 0
                for old_z, new_z in zip(res, self.global_params):
                    with torch.no_grad():
                        d_z += torch.norm(new_z - old_z, p='fro').item()**2
                d_z = np.sqrt(d_z)
                if d_z >= self.delta_z[i]: 
                    # self.last_communicated = [torch.zeros(param.shape).to(self.device).copy_(param) for param in self.global_params]
                    self.agents[i].recieve=True
                    # global_freq[i] = 1
                    global_freq[i] = 1
                else: self.agents[i].recieve=False
            
            print(p_meas)
            
            # Test updated params on validation set
            acc_descrption = ''
            if loader is not None:
                # Get gloabl variable Z and copy to a network for validation
                with torch.no_grad():
                    self.global_model = self.set_parameters(self.global_params, self.global_model)
                    global_acc = self.validate_global(loader=loader)
                    acc_descrption += f', Global Acc = {global_acc:.4f}'
            if self.device == 'cuda': torch.cuda.synchronize()
            
            # Check RAM
            if self.device == 'cuda':
                command = 'nvidia-smi'
                p = subprocess.check_output(command)
                ram_using = re.findall(r'\b\d+MiB+ /', str(p))[0][:-5]
                GPU_desctiption = f', ram = {ram_using}'
            else: GPU_desctiption = ''
            
            # Analyse communication frequency and log stats
            freq = (global_freq + local_freq)*0.5
            # freq = local_freq

            delta_z_desc = ', dz: ('
            for dz in self.delta_z:
                delta_z_desc += f'{dz:.1f} '
            delta_z_desc + ')'
            self.pbar.set_description(f'global: ' + global_desc + acc_descrption + delta_description + delta_z_desc)

            # For experiment purposes
            self.rates.append(freq)
            self.val_accs.append(global_acc.detach().cpu().numpy())
            
            if adaptive_delta:
                delta = delta + K_x*(local_freq - rate_ref)
                if delta <= 0: delta = 0
                # Assign delta to clients and server
                for agent in self.agents: agent.delta = delta
                # self.delta_z = delta
                # integral += global_comm/(round+1) - rate_ref
                # self.delta_z = K_z*(global_comm/(round+1) - rate_ref) #+ 0.01*integral
                
                # if round < 10:
                #     self.delta_z += K_z*(sum(global_comm[-window_length:])/(round + 1) - rate_ref)
                # else:
                #     self.delta_z += K_z*(sum(global_comm[-window_length:])/10 - rate_ref)
                
                # self.delta_z = K_z*integral
                # self.delta_z = 7
                p_meas = (1-alpha)*p_meas + alpha*global_freq
                # self.delta_z += K_z*(p_meas - rate_ref)

                self.delta_z = self.delta_z + K_z*(p_meas - rate_ref*np.ones(p_meas.shape))
                for i, dz in enumerate(self.delta_z):
                    if dz <=0: self.delta_z[i] = 0

        load = statistics.mean(global_comm)*0.5 + statistics.mean(local_comm)*0.5
        print(f'Total communication load = {load}, alpha = {alpha}')
        self.rates = np.array(self.rates)
        self.val_accs = np.array(self.val_accs)


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
    def __init__(self, clients: List[agents.FedLearn], C: float, t_max: int, model: torch.nn.Module, device: str) -> None:
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