class BaseTrainer:
    """Base class for all trainers."""

    def __init__(self, model, task, optimizer, config):
        self.model = model
        self.task = task
        self.optimizer = optimizer
        self.config = config

    def train_epoch(self, train_loader):
        """Train the model for one epoch."""
        raise NotImplementedError

    def validate(self, val_loader):
        """Validate the model."""
        raise NotImplementedError

    def train(self, train_loader, val_loader):
        """Full training loop."""
        raise NotImplementedError 