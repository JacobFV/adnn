import unittest

import torch

from adnn.methods.adnn_linear import AdaptiveDynamicsNeuralNetwork

class TestAdaptiveDynamicsNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.N = 10
        self.model = AdaptiveDynamicsNeuralNetwork(self.N)
        self.dt = 0.1
        self.t_span = [0.0, 0.5]
        self.device = 'cpu'

    def test_update_latent(self):
        self.model.initialize_latent(torch.ones(self.N), torch.zeros(self.N))
        self.model.update_latent(self.dt)
        self.assertIsNotNone(self.model.L_real)
        self.assertIsNotNone(self.model.L_imag)

    def test_check_collapse(self):
        self.model.initialize_latent(torch.ones(self.N), torch.zeros(self.N))
        collapse_indices = self.model.check_collapse()
        self.assertIsInstance(collapse_indices, torch.Tensor)

    def test_forward(self):
        self.model.initialize_latent(torch.ones(self.N), torch.zeros(self.N))
        L_real_traj, L_imag_traj = self.model.forward(self.t_span, self.dt)
        self.assertEqual(L_real_traj.size(0), int((self.t_span[1] - self.t_span[0]) / self.dt) + 1)
        self.assertEqual(L_real_traj.size(1), self.N)

if __name__ == '__main__':
    unittest.main()