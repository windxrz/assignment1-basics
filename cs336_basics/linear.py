import torch
import torch.nn.functional as F
import math

from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        sigma = math.sqrt(2 / (in_features + out_features))
        if dtype is None:
            self.weights = torch.zeros((out_features, in_features))
        else:
            self.weights = torch.zeros((out_features, in_features), dtype=dtype)
        self.weights = torch.nn.init.trunc_normal_(self.weights, 0, sigma, -3 * sigma, 3 * sigma)

        if device is not None:
            self.weights = self.weights.to(device)

        self.weights = torch.nn.Parameter(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "... d_out d_in, ... d_in -> ... d_out")
