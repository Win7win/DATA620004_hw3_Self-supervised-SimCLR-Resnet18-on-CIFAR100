import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

def load_stl10_data(batchsize):
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.STL10(root='data', split='unlabeled', download=True, transform=data_transforms)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4)
    
    return data_loader
