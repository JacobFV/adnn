import torch
from .base import BaseDataset

class SyntheticDataset(BaseDataset):
    def __init__(self, num_samples, N):
        super().__init__()
        self.inputs = torch.randn(num_samples, N)
        self.targets = torch.sin(self.inputs) + 0.1 * torch.randn_like(self.inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def get_input_dim(self):
        return self.inputs.shape[1]

    def get_output_dim(self):
        return self.targets.shape[1] 