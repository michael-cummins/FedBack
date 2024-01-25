from torch.utils import data
import torchvision
from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
import os

def load_cifar10_data(datadir='./data/cifar10/'):
    """Load CIFAR10 dataset."""
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10Sub(
        datadir, train=True, download=True, transform=transform
    )
    cifar10_test_ds = CIFAR10Sub(
        datadir, train=False, download=True, transform=transform
    )

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

class CIFAR10Sub(data.Dataset):
    """CIFAR-10 dataset with idxs."""

    def __init__(
        self,
        root='./data/cifar10/',
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_sub_dataset__()

    def __build_sub_dataset__(self):
        """Build sub dataset given idxs."""
        cifar_dataobj = CIFAR10(
            self.root, self.train, self.transform, self.target_transform, self.download
        )

        if torchvision.__version__ == "0.2.1":
            if self.train:
                # pylint: disable=redefined-outer-name
                data, target = cifar_dataobj.train_data, np.array(
                    cifar_dataobj.train_labels
                )
            else:
                # pylint: disable=redefined-outer-name
                data, target = cifar_dataobj.test_data, np.array(
                    cifar_dataobj.test_labels
                )
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """Get item by index.

        Args:
            index (int): Index.

        Returns
        -------
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Length.

        Returns
        -------
            int: length of data
        """
        return len(self.data)

def partition_data(partition, num_clients, beta, num_labels: int):
    """Partition data into train and test sets for IID and non-IID experiments."""
    X_train, y_train, X_test, y_test = load_cifar10_data()

    n_train = y_train.shape[0]
    np.random.seed(42)
    if partition in ("homo", "iid"):
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, num_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}

    elif partition in ("noniid-labeldir", "noniid"):
        min_size = 0
        min_require_size = 10
        K = num_labels

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return (X_train, y_train, X_test, y_test, net_dataidx_map)


def get_dataloader(datadir, train_bs, test_bs, dataidxs=None):
    """Get dataloader for a given dataset."""
    datadir='./data/cifar10'
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Lambda(
            #     lambda x: F.pad(
            #         Variable(x.unsqueeze(0), requires_grad=False),
            #         (4, 4, 4, 4),
            #         mode="reflect",
            #     ).data.squeeze()
            # ),
            # transforms.ToPILImage(),
            # transforms.ColorJitter(brightness=noise_level),
            # transforms.RandomCrop(32),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            normalize,
        ]
    )
    # data prep for test set
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    train_ds = CIFAR10Sub(
        datadir,
        dataidxs=dataidxs,
        train=True,
        transform=transform_train,
        download=False,
    )
    test_ds = CIFAR10Sub(datadir, train=False, transform=transform_test, download=False)

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True
    )
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl, train_ds, test_ds