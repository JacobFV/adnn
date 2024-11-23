import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Base class for all datasets."""

    def __init__(self):
        super().__init__()

    def get_input_dim(self):
        """Return the dimension of the input data."""
        raise NotImplementedError

    def get_output_dim(self):
        """Return the dimension of the output data."""
        raise NotImplementedError 