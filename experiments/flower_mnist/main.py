import hydra
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
import torch
from data import get_mnist_data
from admm.clients import FlowerCLient
import flwr as fl
from admm.models import CNN

def generate_client_fn(train_loaders, val_loader):
    def client_fn(cid: str):
        return FlowerCLient(
            train_loader=train_loaders[int(cid)], 
            val_loader=val_loader
        )
    return client_fn

def get_on_fit_config(config: DictConfig):
    
    """
    Server round may be useful for some cases but not in this one. 
    Must still be included in func definitions
    """

    def fit_config_fn(server_round: int):
        return {'lr': config.lr, 'momentum': config.momentum, 'epochs': config.epochs}
    return fit_config_fn

def get_evaluate_fn(test_loader):
    def evaluate_fn(server_round: int, parameters, config):
        model = CNN()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict=state_dict, strict=True)
        model.eval()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for data, target in test_loader:
                out = model(data)
                loss += loss(out, target)
                _, pred = torch.max(out.data, 1)
                correct += (pred == target).sum().item()
        acc = correct/len(test_loader.dataset)
        return loss, {'accuracy': acc}
    return evaluate_fn

@hydra.main(config_path='conf', config_name='base', version_base=None)
def main(cfg: DictConfig):
        
    train_loaders, val_loader, test_loader = get_mnist_data(cfg.batch_size, root='../data/mnist_data/')
    client_fn = generate_client_fn(train_loaders, val_loader)
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.000001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.000001,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(config=cfg.config_fit),
        evaluate_fn=get_evaluate_fn(test_loader=test_loader)
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy
    )

if __name__ == '__main__':
    main()