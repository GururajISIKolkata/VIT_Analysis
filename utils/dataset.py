from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_transforms(train=True):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def get_cifar10(data_dir, train=True):
    return datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=get_transforms(train=train)
    )


def get_subset(dataset, fraction):
    size = int(len(dataset) * fraction)
    indices = np.random.permutation(len(dataset))[:size]
    return Subset(dataset, indices)


def get_dataloaders(data_dir, batch_size=128, fraction=1.0):
    train_dataset = get_cifar10(data_dir, train=True)
    test_dataset = get_cifar10(data_dir, train=False)

    if fraction < 1.0:
        train_dataset = get_subset(train_dataset, fraction)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader