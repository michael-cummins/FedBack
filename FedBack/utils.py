import torch
import torch.nn as nn
import random
import numpy as np

# from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays
from typing import Tuple, List
from functools import reduce

# def flwr_aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
#     """Compute weighted average."""
#     # Calculate the total number of examples used during training
#     # num_examples_total = sum(num_examples for (_, num_examples) in results)
#     num_clients = len(results)
#     print(f'Aggregating for {num_clients} clients')

#     # Create a list of weights, each multiplied by the related number of examples
#     weighted_weights = [
#         [layer for layer in weights] for weights, _ in results
#     ]
#     # Compute average weights of each layer
#     weights_prime: NDArrays = [
#         reduce(np.add, layer_updates) / num_clients
#         for layer_updates in zip(*weighted_weights)
#     ]
#     return weights_prime

def split_dataset(dataset, train_ratio, val_ratio):
    dataset_length = len(dataset)
    train_length = int(train_ratio * dataset_length)
    val_length = int(val_ratio * dataset_length)
    test_length = dataset_length - train_length - val_length
    train_dataset_fc, val_dataset_fc, test_dataset_fc = torch.utils.data.random_split(
        dataset, 
        [train_length, val_length, test_length]
    )   
    return train_dataset_fc, val_dataset_fc, test_dataset_fc

# --------- Functions ---------

def sublist_by_fraction(agents, fraction: float):
    n = max(int(len(agents) * fraction), 0) # ensure non-negative number of samples
    sample_indices = set(random.sample(range(len(agents)), n))
    return [agent for i, agent in enumerate(agents) if i in sample_indices]

def scale_params(model_params, a):
    for param in model_params:
        param.data = a*param

def add_params(model1_params, model2_params):
    """
    Adds the parameters of model2 to model1
    """
    for param1, param2 in zip(model1_params, model2_params):
        param1.data =  param1 + param2

def subtract_params(model1_params, model2_params):
    """
    Subtracts the parameters of model2 from model1
    """
    for param1, param2 in zip(model1_params, model2_params):
        param1.data =  param1 - param2

def average_params(model_params):
    N = len(model_params)
    return [sum(params)/N for params in zip(*model_params)]

def sum_params(model_params):
    return [sum(params) for params in zip(*model_params)]

       
# --------- Generators ---------

def add_params_gen(model1_params, model2_params):
    """
    Adds the parameters of model2 to model1
    """
    for param1, param2 in zip(model1_params, model2_params):
        yield  param1 + param2

def subtract_params_gen(model1_params, model2_params):
    """
    Subtracts the parameters of model2 from model1
    """
    for param1, param2 in zip(model1_params, model2_params):
        yield param1 - param2

def scale_params_gen(model_params, a):
    """
    Scales params of model by a factor of "a"
    """
    for param1 in model_params:
        yield a*param1

