import torch.nn as nn
from .base import BaseTask

class RegressionTask(BaseTask):
    """Regression task with MSE loss."""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.metrics = [self.mse]

    def mse(self, outputs, targets):
        """Mean Squared Error metric."""
        return nn.functional.mse_loss(outputs, targets).item() 