import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from admm.models import Cifar10CNN
from admm.data import split_dataset
import seaborn as sns
sns.set_theme()

if torch.cuda.is_available(): 
    device = 'cuda'
    print('GPU available')
else:
    raise Exception('GPU not available')


def get_data():
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar_trainset = datasets.CIFAR10(
        root='./data/cifar10', train=True,
        download=True, transform=cifar_transform
    )
    cifar_testset = datasets.CIFAR10(
        root='./data/cifar10', train=False,
        download=True, transform=cifar_transform
    )
    train_dataset, val_dataset, _ = split_dataset(dataset=cifar_trainset, train_ratio=0.8, val_ratio=0.2)
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    test_loader = DataLoader(cifar_testset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, val_loader

if __name__ == '__main__':
    train_loader, test_loader, val_loader  = get_data()
    torch.manual_seed(42)
    model = Cifar10CNN().to(device=device)
    num_epochs = 15
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    pbar  = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []

    model.train()
    for epoch in pbar:
        for (image, target), (val_image, val_target) in zip(train_loader, val_loader):
            model.train()
            image, target = image.to(device), target.to(device)
            loss = loss_func(model(image), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                val_image, val_target = val_image.to(device), val_target.to(device)
                v_loss = loss_func(model(val_image), val_target)
                val_losses.append(v_loss.cpu().numpy())
            
            pbar.set_description(f'Epoch {epoch} | training loss = {loss.item():.4f}, validation loss = {v_loss.item():.4f}')
            # pbar.set_description(f'Epoch {epoch} | training loss = {loss.item():.4f}')
            train_losses.append(loss.detach().cpu().numpy())

    print('\nFinished training - Working on test set\n')
    with torch.no_grad():
        model.eval()
        wrong_count = 0
        total = len(val_loader.dataset)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = torch.argmax(model(data), dim=1)
            wrong_count += torch.count_nonzero(out-target)
        global_acc = 1 - wrong_count/total
    
    print(f'Final accuracy of model = {global_acc}')

    # plt.plot(range(len(val_losses)), val_losses, label='validation')
    # plt.plot(range(len(train_losses)), train_losses, label='training')
    # plt.xlabel('Gradient Step')
    # plt.ylabel('Loss')
    # plt.title('Sudmitted Job')
    # plt.savefig('images/cifar/training_loss.png')    





