import torch
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
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        if dtype is None:
            weight = torch.zeros((num_embeddings, embedding_dim))
        else:
            weight = torch.zeros((num_embeddings, embedding_dim), dtype=dtype)

        weight = torch.nn.init.trunc_normal_(weight, 0, 1, -3, 3)
        if device is not None:
            weight = weight.to(device)
        self.weight = torch.nn.Parameter(weight)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        res = self.weight[token_ids]
        return res
