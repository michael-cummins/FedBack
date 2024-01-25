import flwr as fl
from flwr.server.strategy import Strategy
import torch
import torch
from typing import List
from admm.utils import sum_params, add_params, average_params, scale_params, flower
from admm import agents
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import numpy as np
from collections import OrderedDict
import statistics
from admm.utils import sublist_by_fraction
import math
from typing import Callable, Union, Tuple, Dict, Optional

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import torch.nn as nn

class FedConsensus(Strategy):

    def __init__(self, num_clients, model: nn.Module, device: str, val_laoder: DataLoader):
        super().__init__(self)

    def __repr__(self) -> str:
        return "FedConsensus"
        
    def initialize_parameters(self, client_manager):
        # Your implementation here
        pass

    def configure_fit(self, server_round, parameters, client_manager):
        # Your implementation here
        pass

    def aggregate_fit(self, server_round, results, failures):
        # Your implementation here
        pass

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Your implementation here
        pass

    def aggregate_evaluate(self, server_round, results, failures):
        # Your implementation here
        pass

    def evaluate(self, parameters):
        # Your implementation here
        pass

    ### Utility Functions from admm.servers ###

    def validate_global(self, loader: DataLoader) -> float:
        wrong_count = 0
        total = len(loader.dataset)
        for data, target in loader:
            data, target = data.to(self.device), target.type(torch.LongTensor).to(self.device)
            out = torch.argmax(self.global_model(data), dim=1)
            wrong_count += torch.count_nonzero(out-target)
        global_acc = 1 - wrong_count/total
        return global_acc

    def set_parameters(self, parameters, model: torch.nn.Module) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model