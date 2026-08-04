import torch
import math
import numpy.typing as npt
from jaxtyping import Bool, Float, Int
from torch import Tensor

from einops import einsum, rearrange, reduce, repeat


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    x = x - torch.max(x, dim=i, keepdim=True).values
    x = torch.exp(x)
    x = x / torch.sum(x, dim=i, keepdim=True)
    return x


def scaled_dot_product_attention(Q, K, V, mask = None):
    mat = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(Q.shape[-1])
    if mask is not None:
        mat = mat.masked_fill(~mask, -torch.inf)
    prob = softmax(mat, -1)
    res = einsum(prob, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return res


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


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        idx_i = torch.arange(0.0, max_seq_len).to(device)
        idx_k = torch.pow(theta, -torch.arange(0.0, d_k // 2) * 2 / d_k).to(device)
        thetas = einsum(idx_i, idx_k, "l, d_k -> l d_k")
        self.register_buffer("sin_value", torch.sin(thetas), persistent=False)
        self.register_buffer("cos_value", torch.cos(thetas), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_value[token_positions]
        sin = self.sin_value[token_positions]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        res_even = einsum(x_even, cos, "... d_k_2, ... d_k_2 -> ... d_k_2") - einsum(x_odd, sin, "... d_k_2, ... d_k_2 -> ... d_k_2")
        res_odd = einsum(x_even, sin, "... d_k_2, ... d_k_2 -> ... d_k_2") + einsum(x_odd, cos, "... d_k_2, ... d_k_2 -> ... d_k_2")
        res = rearrange([res_even, res_odd], "pair ... d_k_2 -> ... (d_k_2 pair)")
        return res


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 10000, theta: float | None = None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len), diagonal=0).bool()
        self.register_buffer("mask", mask, persistent=False)

        self.theta = theta
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        seq_len = x.shape[-2]
        mask = self.mask[:seq_len, :seq_len]
        Q = rearrange(Q, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = self.num_heads)
        K = rearrange(K, "... seq_len (num_head d_k) -> ... num_head seq_len d_k", num_head = self.num_heads)

        if self.theta is not None:
            if token_positions is None:
                token_positions = torch.arange(0, seq_len, dtype=torch.int)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        V = rearrange(V, "... seq_len (num_head d_v) -> ... num_head seq_len d_v", num_head = self.num_heads)
        res = scaled_dot_product_attention(Q, K, V, mask)
        res = rearrange(res, "... num_head seq_len d_v -> ... seq_len (num_head d_v)")
        res = self.output_proj(res)
        return res


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int,
                 max_seq_len: int = 10000, rope_theta: float | None = None,
                 eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope_theta, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = FFNSwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int,
                 context_length: int, vocab_size: int, num_layers: int,
                 rope_theta: float | None = None, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = [
            TransformerBlock(d_model, d_ff, num_heads, context_length, rope_theta, eps, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.layers = torch.nn.ModuleList(self.layers)
        self.ln_final = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.token_embeddings(x)
        for block in self.layers:
            emb = block(emb)
        res = self.ln_final(emb)
        res = self.lm_head(res)
        return res
