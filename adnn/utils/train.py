import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from .eval import evaluate

logger = logging.getLogger(__name__)

def train(model, train_loader, val_loader, config):
    """
    Training function with configuration dictionary
    """
    device = config['device']
    writer = SummaryWriter(log_dir=config['log_dir'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            model.initialize_latent(inputs, torch.zeros_like(inputs))
            L_real_traj, L_imag_traj = model.forward(config['t_span'], config['dt'])
            
            outputs = L_real_traj[-1]
            loss = criterion(outputs, targets)
            loss.backward()
            
            if config['clip_grad']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            
            if batch_idx % config['log_interval'] == 0:
                logger.info(f'Epoch [{epoch+1}/{config["num_epochs"]}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_loader) + batch_idx)
        
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        logger.info(f'Epoch [{epoch+1}/{config["num_epochs"]}] Average Training Loss: {avg_epoch_loss:.4f}')
        
        if (epoch + 1) % config['eval_interval'] == 0:
            val_loss = evaluate(model, val_loader, config['t_span'], config['dt'], criterion, device)
            logger.info(f'Epoch [{epoch+1}/{config["num_epochs"]}] Validation Loss: {val_loss:.4f}')
            writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = Path(config['model_dir']) / 'best_model.pth'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
    
    writer.close()
    return model
