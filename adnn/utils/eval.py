def evaluate(model, data_loader, t_span, dt, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Initialize the latent state
            model.initialize_latent(inputs, torch.zeros_like(inputs))

            # Simulate the system
            L_real_traj, L_imag_traj = model.forward(t_span, dt)

            # Compute outputs and loss
            outputs = L_real_traj[-1]
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss

# Save the model
torch.save(model.state_dict(), 'adnn_model.pth')

# Load the model
model.load_state_dict(torch.load('adnn_model.pth'))
model.to(device)