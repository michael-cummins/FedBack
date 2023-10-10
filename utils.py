import torch
import torch.nn as nn

# --------- Functions ---------

def scale_params(model_params, a):
    for param in model_params:
        param.data = a*param
    return model_params

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
    return model1_params


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