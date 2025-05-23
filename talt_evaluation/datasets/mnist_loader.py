import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist(batch_size: int = 64, num_workers: int = 4, data_dir: str = "./data"):
    """
    Load the MNIST dataset with train/validation/test splits.

    Args:
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for data loading.
        data_dir (str): Directory to store the MNIST dataset.

    Returns:
        tuple: Training, validation, and test DataLoaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full training dataset
    full_trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    
    # Split training data into train and validation sets (90% train, 10% validation)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(
        full_trainset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Load test dataset
    testset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Create data loaders with persistent workers for better performance
    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        valset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader, test_loader
