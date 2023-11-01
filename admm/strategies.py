import flwr as fl
from flwr.server.strategy import Strategy
import torch

class FedConsensus(Strategy):

    def __init__(self):
        super().__init__()
        
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