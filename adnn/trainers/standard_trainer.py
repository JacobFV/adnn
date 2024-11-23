import torch
from .base import BaseTrainer
from adnn.utils.logging import get_logger

logger = get_logger(__name__)

class StandardTrainer(BaseTrainer):
    """Standard training loop."""

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.config['device'])
            targets = targets.to(self.config['device'])

            self.optimizer.zero_grad()

            self.model.initialize_latent(inputs, torch.zeros_like(inputs))
            outputs, _ = self.model.forward(self.config['t_span'], self.config['dt'])
            outputs = outputs[-1]

            loss = self.task.compute_loss(outputs, targets)
            loss.backward()

            if self.config.get('clip_grad', False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))

            self.optimizer.step()
            total_loss += loss.item() * inputs.size(0)

            if batch_idx % self.config.get('log_interval', 10) == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.config['device'])
                targets = targets.to(self.config['device'])

                self.model.initialize_latent(inputs, torch.zeros_like(inputs))
                outputs, _ = self.model.forward(self.config['t_span'], self.config['dt'])
                outputs = outputs[-1]

                loss = self.task.compute_loss(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(val_loader.dataset)
        return avg_loss

    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(train_loader)
            logger.info(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}')

            if (epoch + 1) % self.config.get('eval_interval', 1) == 0:
                val_loss = self.validate(val_loader)
                logger.info(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = self.config['model_dir'] / 'best_model.pth'
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f'Saved best model to {save_path}') 