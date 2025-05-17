#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_cifar_dataset(num_classes, batch_size=128, num_workers=4):
    """
    Get CIFAR10 or CIFAR100 dataset with standard data augmentations.
    
    Args:
        num_classes: Number of classes (10 for CIFAR10, 100 for CIFAR100)
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if num_classes not in [10, 100]:
        raise ValueError("num_classes must be either 10 (CIFAR10) or 100 (CIFAR100)")
    
    # Define transforms for training (with augmentation)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    # Define transforms for validation/testing (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load the appropriate CIFAR dataset
    if num_classes == 10:
        cifar_class = torchvision.datasets.CIFAR10
    else:  # num_classes == 100
        cifar_class = torchvision.datasets.CIFAR100
    
    # Load training data
    full_trainset = cifar_class(
        root='./data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    
    # Split training data into train and validation sets (90% train, 10% validation)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(
        full_trainset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # For validation set, we want to use the test transform (no augmentation)
    valset.dataset = cifar_class(
        root='./data', 
        train=True, 
        download=False,
        transform=transform_test
    )
    
    # Only use the validation indices
    valset = torch.utils.data.Subset(valset.dataset, valset.indices)
    
    # Load test data
    testset = cifar_class(
        root='./data', 
        train=False,
        download=True, 
        transform=transform_test
    )
    
    # Create data loaders
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    valloader = DataLoader(
        valset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    testloader = DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return trainloader, valloader, testloader