import torch
import math

from einops import einsum, rearrange, reduce


class Linear(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        sigma = math.sqrt(2 / (in_features + out_features))
        if dtype is None:
            weight = torch.zeros((out_features, in_features))
        else:
            weight = torch.zeros((out_features, in_features), dtype=dtype)
        weight = torch.nn.init.trunc_normal_(weight, 0, sigma, -3 * sigma, 3 * sigma)

        if device is not None:
            weight = weight.to(device)

        self.weight = torch.nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "... d_out d_in, ... d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if dtype is None:
            weight = torch.zeros((num_embeddings, embedding_dim))
        else:
            weight = torch.zeros((num_embeddings, embedding_dim), dtype=dtype)

        weight = torch.nn.init.trunc_normal_(weight, 0, 1, -3, 3)
        if device is not None:
            weight = weight.to(device)
        self.weight = torch.nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        if dtype is None:
            weight = torch.ones(d_model)
        else:
            weight = torch.ones(d_model, dtype=dtype)
        
        if device is not None:
            weight = weight.to(device)

        self.weight = torch.nn.Parameter(weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = reduce(x * x, "... d_model -> ...", "mean")
        rms = 1 / torch.sqrt(rms + self.eps)
        x_scaled = einsum(rms, x, "..., ... d_model -> ... d_model")
        res = einsum(x_scaled, self.weight, "... d_model, d_model -> ... d_model")
        return res


class FFNSwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int = 0, device=None, dtype=None):
        super().__init__()
        if d_ff == 0:
            d_ff = int(d_model / 3 * 8 / 64) * 64
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def _SiLU(self, x):
        return x / (1 + torch.exp(-x))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._SiLU(self.w1(x))
        x2 = self.w3(x)
        res = self.w2(einsum(x1, x2, "... d_ff, ... d_ff -> ... d_ff"))
        return res
