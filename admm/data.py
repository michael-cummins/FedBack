import torch
import torch.nn as nn

from typing import List, Optional, Tuple
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision.datasets import MNIST
from admm.moon_dataset import get_dataloader, partition_data

def get_cifar_data(num_clients: int = 100, batch_size: int = 20):
    (
        _,
        _,
        _,
        _,
        net_dataidx_map,
    ) = partition_data(
        partition='noniid',
        num_clients=num_clients,
        beta=0.5,
        num_labels=10
    )
    _, test_global_dl, _, _ = get_dataloader(
        datadir='./data/cifar10',
        train_bs=32,
        test_bs=32,
    )
    trainloaders = []
    for idx in range(num_clients):
        train_dl, _, _, _ = get_dataloader(
            './data/cifar10', batch_size, 32, net_dataidx_map[idx]
        )
        trainloaders.append(train_dl)
    trainsets = [loader.dataset for loader in trainloaders]
    
    for i, dataset in enumerate(trainsets):
        labels = np.zeros(10)
        dummy_loader = DataLoader(dataset, batch_size=1)
        for data, target in dummy_loader:
            labels[int(target.item())] += 1
        print(f'Dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    labels = np.zeros(10)
    dummy_loader = DataLoader(test_global_dl.dataset, batch_size=1)
    for data, target in dummy_loader:
        labels[int(target.item())] += 1
    print(f'Validation dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    return trainloaders, test_global_dl

def get_mnist_data(num_clients: int = 100, batch_size: int = 42):
    mnist_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(lambda x: torch.flatten(x))
    ])

    mnist_trainset = datasets.MNIST(
        root='./data/mnist_data', train=True,
        download=True, transform=mnist_transform, 
    )
    mnist_testset = datasets.MNIST(
        root='./data/mnist_data', train=False,
        download=True, transform=mnist_transform
    )

    train_dataset, val_dataset, _ = split_dataset(dataset=mnist_trainset, train_ratio=0.8, val_ratio=0.2)

    trainsets = mnist_partition_data(
        num_clients=num_clients,
        iid=False,
        balance=True,
        power_law=False,
        seed=42,
        trainset=train_dataset.dataset,
        labels_per_partition=1
    )

    """
    Print data to evaluate heterogeneity
    """
    for i, dataset in enumerate(trainsets):
        labels = np.zeros(10)
        dummy_loader = DataLoader(dataset, batch_size=1)
        for data, target in dummy_loader:
            labels[target.item()] += 1
        print(f'Dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    labels = np.zeros(10)
    dummy_loader = DataLoader(val_dataset, batch_size=1)
    for data, target in dummy_loader:
        labels[target.item()] += 1
    print(f'Validation dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    labels = np.zeros(10)
    dummy_loader = DataLoader(mnist_testset, batch_size=1)
    for data, target in dummy_loader:
        labels[target.item()] += 1
    print(f'Validation dataset {i} distribution: {labels} - num_samples = {labels.sum()}')

    """
    Build DataLoaders
    """

    train_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) for dataset in trainsets
    ]
    test_loader = DataLoader(mnist_testset, batch_size=100, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True, num_workers=0)

    return train_loaders, test_loader, val_loader


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

def _download_data() -> Tuple[Dataset, Dataset]:
    """Download (if necessary) and returns the MNIST dataset.

    Returns
    -------
    Tuple[MNIST, MNIST]
        The dataset for training and the dataset for testing MNIST.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    testset = MNIST("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


# pylint: disable=too-many-locals
def mnist_partition_data(
    num_clients,
    labels_per_partition: int,
    iid: Optional[bool] = False,
    power_law: Optional[bool] = True,
    balance: Optional[bool] = False,
    seed: Optional[int] = 42,
    trainset: Optional[Dataset] = None
) -> Tuple[List[Dataset], Dataset]:
    """Split training set into iid or non iid partitions to simulate the federated.

    setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    iid : bool, optional
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed
        by chunks to each client (used to test the convergence in a worst case scenario)
        , by default False
    power_law: bool, optional
        Whether to follow a power-law distribution when assigning number of samples
        for each client, defaults to True
    balance : bool, optional
        Whether the dataset should contain an equal number of samples in each class,
        by default False
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing
        the model.
    """

    if balance:
        trainset = _balance_classes(trainset, seed)

    partition_size = int(len(trainset) / num_clients)
    lengths = [partition_size] * num_clients

    if iid:
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
    else:
        if power_law:
            trainset_sorted = _sort_by_class(trainset)
            datasets = _power_law_split(
                trainset_sorted,
                num_partitions=num_clients,
                num_labels_per_partition=labels_per_partition,
                min_data_per_partition=10,
                mean=0.0,
                sigma=2.0,
            )
        else:
            shard_size = int(partition_size / 2)
            if trainset.targets is not torch.Tensor:
                idxs = torch.Tensor(trainset.targets).argsort()
            else:
                idxs = trainset.targets.argsort()
            sorted_data = Subset(trainset, idxs)
            tmp = []
            for idx in range(num_clients * 2):
                tmp.append(
                    Subset(
                        sorted_data, np.arange(shard_size * idx, shard_size * (idx + 1))
                    )
                )
            idxs_list = torch.randperm(
                num_clients * 2, generator=torch.Generator().manual_seed(seed)
            )
            datasets = [
                ConcatDataset((tmp[idxs_list[2 * i]], tmp[idxs_list[2 * i + 1]]))
                for i in range(num_clients)
            ]

    return datasets


def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    """Balance the classes of the trainset.

    Trims the dataset so each class contains as many elements as the
    class that contained the least elements.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be balanced.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42.

    Returns
    -------
    Dataset
        The balanced training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    if type(trainset.targets) is list:
        trainset.targets = torch.Tensor(trainset.targets)
    idxs = trainset.targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in np.cumsum(class_counts):
        tmp.append(Subset(trainset, idxs[int(count) : int(count + smallest)]))
        tmp_targets.append(trainset.targets[idxs[int(count) : int(count + smallest)]])
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = Subset(unshuffled, shuffled_idxs)
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled


def _sort_by_class(
    trainset: Dataset,
) -> Dataset:
    """Sort dataset by class/label.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be sorted.

    Returns
    -------
    Dataset
        The sorted training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    if trainset.targets is not torch.Tensor:
        idxs = torch.Tensor(trainset.targets).argsort()
    else:
        idxs = trainset.targets.argsort()  # sort targets in ascending order
    tmp = []  # create subset of smallest class
    tmp_targets = []  # same for targets

    start = 0
    for count in np.cumsum(class_counts):
        tmp.append(
            Subset(trainset, idxs[start : int(count + start)])
        )  # add rest of classes

        if trainset.targets is not torch.Tensor:
            targets = torch.Tensor(trainset.targets)
        else: targets = trainset.targets
        
        tmp_targets.append(targets[idxs[start : int(count + start)]])
        start += count
    sorted_dataset = ConcatDataset(tmp)  # concat dataset
    sorted_dataset.targets = torch.cat(tmp_targets)  # concat targets
    return sorted_dataset


# pylint: disable=too-many-locals, too-many-arguments
def _power_law_split(
    sorted_trainset: Dataset,
    num_partitions: int,
    num_labels_per_partition: int = 10,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> Dataset:
    """Partition the dataset following a power-law distribution. It follows the.

    implementation of Li et al 2020: https://arxiv.org/abs/1812.06127 with default
    values set accordingly.

    Parameters
    ----------
    sorted_trainset : Dataset
        The training dataset sorted by label/class.
    num_partitions: int
        Number of partitions to create
    num_labels_per_partition: int
        Number of labels to have in each dataset partition. For
        example if set to two, this means all training examples in
        a given partition will be long to the same two classes. default 2
    min_data_per_partition: int
        Minimum number of datapoints included in each partition, default 10
    mean: float
        Mean value for LogNormal distribution to construct power-law, default 0.0
    sigma: float
        Sigma value for LogNormal distribution to construct power-law, default 2.0

    Returns
    -------
    Dataset
        The partitioned training dataset.
    """
    targets = sorted_trainset.targets
    full_idx = list(range(len(targets)))

    class_counts = np.bincount(sorted_trainset.targets)
    labels_cs = np.cumsum(class_counts)
    labels_cs = [0] + labels_cs[:-1].tolist()

    partitions_idx: List[List[int]] = []
    num_classes = len(np.bincount(targets))
    hist = np.zeros(num_classes, dtype=np.int32)

    # assign min_data_per_partition
    min_data_per_class = int(min_data_per_partition / num_labels_per_partition)
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition):
            # label for the u_id-th client
            cls = (u_id + cls_idx) % num_classes
            # record minimum data
            indices = list(
                full_idx[
                    labels_cs[cls]
                    + hist[cls] : labels_cs[cls]
                    + hist[cls]
                    + min_data_per_class
                ]
            )
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class

    # add remaining images following power-law
    probs = np.random.lognormal(
        mean,
        sigma,
        (num_classes, int(num_partitions / num_classes), num_labels_per_partition),
    )
    remaining_per_class = class_counts - hist
    # obtain how many samples each partition should be assigned for each of the
    # labels it contains
    # pylint: disable=too-many-function-args
    probs = (
        remaining_per_class.reshape(-1, 1, 1)
        * probs
        / np.sum(probs, (1, 2), keepdims=True)
    )

    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            count = int(probs[cls, u_id // num_classes, cls_idx])

            # add count of specific class to partition
            indices = full_idx[
                labels_cs[cls] + hist[cls] : labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    # construct subsets
    partitions = [Subset(sorted_trainset, p) for p in partitions_idx]
    return partitions
