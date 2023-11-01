import torch
import torch.nn as nn
from torchvision import datasets, transforms
from admm.utils import split_dataset

def get_mnist_data(batch_size, root: str):
    train_ratio = 0.8
    val_ratio = 0.2

    # Define transformationsw
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # Load the full MNIST dataset and filter for 1's and 2's
    dataset = datasets.MNIST(root=root, train=True, transform=transform, download=False)
    test_set = datasets.MNIST(root=root, train=False, transform=transform, download=False)
    filtered_dataset = [data for data in dataset if data[1] == 0 or data[1] == 1] # For CNN
    filtered_test_set = [data for data in test_set if data[1] == 0 or data[1] == 1] # For CNN

    # Split into train, val and test for CNN
    train_dataset, val_dataset, _ = split_dataset(dataset=filtered_dataset, train_ratio=train_ratio, val_ratio=val_ratio)
    digit_1_train_dataset = [data for data in train_dataset if data[1] == 0]
    digit_2_train_dataset = [data for data in train_dataset if data[1] == 1]
    digit_1_train_loader = torch.utils.data.DataLoader(digit_1_train_dataset, batch_size=batch_size, shuffle=True)
    digit_2_train_loader = torch.utils.data.DataLoader(digit_2_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(filtered_test_set, batch_size=batch_size, shuffle=True)

    return [digit_1_train_loader, digit_2_train_loader], val_loader, test_loader 
