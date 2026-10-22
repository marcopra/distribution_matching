import unittest
import torch
from agent.rover_networks import CNNEncoder, Encoder


class EncoderInputDtypeTest(unittest.TestCase):
    def test_encoder_input_follows_module_dtype_not_global_default(self):
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            cnn = CNNEncoder((3, 84, 84), feature_dim=8).float()
            cnn_output = cnn.encode_and_project(
                torch.zeros((2, 3, 84, 84), dtype=torch.float64)
            )
            self.assertEqual(cnn_output.dtype, torch.float32)

            mlp = Encoder((4,), hidden_dim=8, feature_dim=3).float()
            mlp_output = mlp(torch.zeros((2, 4), dtype=torch.float64))
            self.assertEqual(mlp_output.dtype, torch.float32)
        finally:
            torch.set_default_dtype(previous_dtype)
