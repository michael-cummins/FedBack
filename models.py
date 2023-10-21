import torch
import torch.nn as nn

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