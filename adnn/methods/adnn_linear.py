import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FlattenedMultiStreamSystem(nn.Module):
    def __init__(self, N, device='cpu'):
        super(FlattenedMultiStreamSystem, self).__init__()
        self.N = N  # Total dimensionality
        self.device = device
        logger.debug(f'Initializing FlattenedMultiStreamSystem with N={N} on device={device}')

        # Initialize learnable parameters
        self.A_real = nn.Parameter(torch.zeros(N, device=device))
        self.A_imag = nn.Parameter(torch.zeros(N, device=device))
        self.w_acc_real = nn.Parameter(torch.zeros(N, device=device))
        self.w_acc_imag = nn.Parameter(torch.zeros(N, device=device))
        self.theta = nn.Parameter(torch.ones(N, device=device))  # Thresholds
        self.W_filter_real = nn.Parameter(torch.randn(N, N, device=device) * 0.01)
        self.W_filter_imag = nn.Parameter(torch.randn(N, N, device=device) * 0.01)

        # Initialize latent state
        self.L_real = torch.zeros(N, device=device)
        self.L_imag = torch.zeros(N, device=device)
        logger.debug('Parameters initialized successfully')

    def forward(self, t_span, dt):
        """
        Simulate the system over the time span t_span with time step dt.
        """
        logger.debug(f'Forward pass started with t_span={t_span}, dt={dt}')
        num_steps = int((t_span[1] - t_span[0]) / dt)
        times = torch.linspace(t_span[0], t_span[1], num_steps, device=self.device)

        # Record latent states over time
        L_real_traj = []
        L_imag_traj = []

        for t in times:
            logger.debug(f'Processing time step t={t:.4f}')
            # Update latent state between collapses
            self.update_latent(dt)

            # Check for collapses
            collapse_indices = self.check_collapse()

            if collapse_indices.numel() > 0:
                logger.debug(f'Collapse detected at indices: {collapse_indices}')
                # Perform collapse operation
                self.collapse(collapse_indices)

            # Record the latent state
            L_real_traj.append(self.L_real.clone())
            L_imag_traj.append(self.L_imag.clone())

        # Stack the recorded latent states
        L_real_traj = torch.stack(L_real_traj)
        L_imag_traj = torch.stack(L_imag_traj)
        logger.debug(f'Forward pass completed. Trajectory shapes: real={L_real_traj.shape}, imag={L_imag_traj.shape}')

        return L_real_traj, L_imag_traj

    def update_latent(self, dt):
        """
        Update the latent state using the dynamics:
        dL/dt = A * L
        """
        logger.debug('Updating latent state')
        # Compute complex A
        A_complex = self.A_real + 1j * self.A_imag

        # Compute the exponential of A * dt
        exp_At = torch.exp(A_complex * dt)

        # Update latent state
        L_complex = self.L_real + 1j * self.L_imag
        L_complex = L_complex * exp_At

        # Update real and imaginary parts
        self.L_real = L_complex.real
        self.L_imag = L_complex.imag
        logger.debug(f'Latent state updated. Max values - real: {self.L_real.max():.4f}, imag: {self.L_imag.max():.4f}')

    def check_collapse(self):
        """
        Check which dimensions satisfy the collapse condition.
        """
        logger.debug('Checking for collapse conditions')
        # Compute complex accumulator weights
        w_acc_complex = self.w_acc_real + 1j * self.w_acc_imag

        # Compute accumulator values
        L_complex = self.L_real + 1j * self.L_imag
        a = (w_acc_complex.conj() * L_complex).real

        # Find indices where accumulator exceeds threshold
        collapse_indices = torch.nonzero(a >= self.theta).squeeze()
        logger.debug(f'Found {collapse_indices.numel()} collapse conditions')

        return collapse_indices

    def collapse(self, indices):
        """
        Perform the collapse operation on specified indices.
        """
        logger.debug(f'Performing collapse on {len(indices)} indices')
        # Extract relevant parts of W_filter
        W_filter_real_sub = self.W_filter_real[indices, :]
        W_filter_imag_sub = self.W_filter_imag[indices, :]

        # Compute the filter
        L_complex = self.L_real + 1j * self.L_imag
        W_filter_complex = W_filter_real_sub + 1j * W_filter_imag_sub
        F_complex = torch.matmul(W_filter_complex, L_complex)

        # Apply activation function (e.g., complex ReLU)
        F_complex_activated = self.complex_relu(F_complex)

        # Update latent state for the collapsed indices
        self.L_real[indices] = self.L_real[indices] * F_complex_activated.real
        self.L_imag[indices] = self.L_imag[indices] * F_complex_activated.imag
        logger.debug('Collapse operation completed')

    def complex_relu(self, z):
        """
        Complex ReLU activation function.
        """
        logger.debug('Applying complex ReLU')
        return torch.relu(z.real) + 1j * z.imag

    def initialize_latent(self, L_real_init, L_imag_init):
        """
        Initialize the latent state.
        """
        logger.debug('Initializing latent state')
        self.L_real = L_real_init.clone().to(self.device)
        self.L_imag = L_imag_init.clone().to(self.device)
        logger.debug(f'Latent state initialized with shapes - real: {self.L_real.shape}, imag: {self.L_imag.shape}')