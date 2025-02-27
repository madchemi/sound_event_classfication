import os
from numpy import random
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader 

def data_loader(data_root):
    dataset = ImageFolder(data_root,transform=transforms.Compose([transforms.ToTensor(),]))

    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random.split(dataset,[train_size, val_size])

    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle = True)
    val_data_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
    
    return train_data_loader, val_data_loader


