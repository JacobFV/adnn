import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from adnn.methods.adnn_linear import AdaptiveDynamicsNeuralNetwork
from adnn.utils.synthetic_data import SyntheticDataset

def create_datasets(N, num_samples, batch_size):
    dataset = SyntheticDataset(num_samples, N)
    
    # Split into training and validation sets
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def check_gradients(model, logger):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            logger.debug(f'Gradient norm for {name}: {grad_norm:.4f}')
            
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                logger.warning(f'Gradient issue in {name}: NaN or Inf detected.')

def train_epoch(model, train_loader, optimizer, criterion, device, writer, epoch, num_epochs, logger):
    model.train()
    epoch_loss = 0.0
    t_span = [0.0, 1.0]
    dt = 0.01

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        model.initialize_latent(inputs, torch.zeros_like(inputs))
        L_real_traj, L_imag_traj = model.forward(t_span, dt)
        
        outputs = L_real_traj[-1]
        loss = criterion(outputs, targets)
        loss.backward()
        
        check_gradients(model, logger)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)

        if batch_idx % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_loader) + batch_idx)
    
    return epoch_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device, t_span, dt):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            model.initialize_latent(inputs, torch.zeros_like(inputs))
            L_real_traj, L_imag_traj = model.forward(t_span, dt)
            
            outputs = L_real_traj[-1]
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    return val_loss / len(val_loader.dataset)

def main():
    # Setup parameters
    N = 1024
    num_samples = 1000
    batch_size = 32
    num_epochs = 20
    eval_interval = 2
    t_span = [0.0, 1.0]
    dt = 0.01
    
    # Setup device and model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AdaptiveDynamicsNeuralNetwork(N, device=device).to(device)
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(log_dir='runs/ADNN_experiment')
    
    # Create datasets
    train_loader, val_loader = create_datasets(N, num_samples, batch_size)
    
    # Training loop
    for epoch in range(num_epochs):
        avg_epoch_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                   device, writer, epoch, num_epochs, logger)
        logger.info(f'Epoch [{epoch+1}/{num_epochs}] Average Training Loss: {avg_epoch_loss:.4f}')
        
        if (epoch + 1) % eval_interval == 0:
            avg_val_loss = validate(model, val_loader, criterion, device, t_span, dt)
            logger.info(f'Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}')
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
    
    writer.close()

if __name__ == "__main__":
    main()