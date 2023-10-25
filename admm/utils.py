import torch
import torch.nn as nn

# --------- Functions ---------

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

def average_params(model_params):
    N = len(model_params)
    return [sum(params)/N for params in zip(*model_params)]
       

def sum_params(model_params):
    for params in zip(*model_params):
        yield sum(params)

