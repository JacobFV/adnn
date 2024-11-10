import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SyntheticDataset(Dataset):
    def __init__(self, num_samples, N):
        self.inputs = torch.randn(num_samples, N)
        self.targets = torch.sin(self.inputs) + 0.1 * torch.randn_like(self.inputs)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class WaveDataset(Dataset):
    def __init__(self, num_samples, N):
        t = torch.linspace(0, 10, N)
        self.inputs = torch.zeros(num_samples, N)
        self.targets = torch.zeros(num_samples, N)
        
        for i in range(num_samples):
            freq = np.random.uniform(0.1, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            self.inputs[i] = torch.sin(2*np.pi*freq*t + phase)
            self.targets[i] = torch.sin(2*np.pi*freq*t + phase + np.pi/4)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def get_dataset(dataset_name, num_samples, N, batch_size):
    datasets = {
        'synthetic': SyntheticDataset,
        'wave': WaveDataset
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(datasets.keys())}")
    
    dataset = datasets[dataset_name](num_samples, N)
    
    # Split into training and validation sets
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader 