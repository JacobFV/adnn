class BaseTask:
    """Base class for all tasks."""

    def __init__(self):
        self.loss_fn = None
        self.metrics = []

    def compute_loss(self, outputs, targets):
        """Compute the loss between outputs and targets."""
        return self.loss_fn(outputs, targets)

    def compute_metrics(self, outputs, targets):
        """Compute evaluation metrics."""
        results = {}
        for metric in self.metrics:
            results[metric.__name__] = metric(outputs, targets) 