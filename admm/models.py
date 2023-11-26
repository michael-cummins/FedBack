import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, num_classes=10) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=4,            
                kernel_size=2,       
                stride=2                      
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=4,              
                out_channels=10,            
                kernel_size=2,                      
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.fc = FCNet(in_channels=196, hidden1=50, out_channels=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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
        layers.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Dummy(nn.Module):

    def __init__(self, x) -> None:
        super().__init__()
        self.params = nn.Parameter(x)

    def forward(self) -> torch.Tensor:
        return self.params