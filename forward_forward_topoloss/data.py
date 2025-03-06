import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from utils import overlay_y_on_x 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def prepare_data():
    """Load MNIST dataset and preprocess batch for training."""
    train_loader, test_loader = MNIST_loaders()
    x_train, y_train = next(iter(train_loader))
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_pos_train = overlay_y_on_x(x_train, y_train)
    rnd = torch.randperm(x_train.size(0))
    x_neg_train = overlay_y_on_x(x_train, y_train[rnd])

    x_val, y_val = next(iter(test_loader))  
    x_val, y_val = x_val.to(device), y_val.to(device)
    return train_loader, test_loader, x_train, y_train, x_pos_train, x_neg_train, x_val, y_val
