import torch
import torch.nn as nn

class FCNet(nn.Module):

    def __init__(self, in_channels, hidden1, out_channels, hidden2=None) -> None:
        super().__init__()

        layers = [nn.Linear(in_features=in_channels, out_features=hidden1), nn.ReLU()]
        if hidden2 is not None:
            layers.append(nn.Linear(in_features=hidden1, out_features=hidden2))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(in_features=hidden2, out_features=out_channels))
        else:
            layers.append(nn.Linear(in_features=hidden1, out_features=out_channels))
        layers.append(nn.Softmax())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Dummy(nn.Module):

    def __init__(self, x) -> None:
        super().__init__()
        self.params = nn.Parameter(x)

    def forward(self) -> torch.Tensor:
        return self.params

class NN(nn.Module):

    def __init__(self, in_channels, hidden, out_channels) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            # nn.ReLU(),
            nn.Linear(hidden, out_channels)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.parameters()