import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention block.

    Examples
    --------
    >>> import torch
    >>> from braindecode.modules import MultiHeadAttention
    >>> module = MultiHeadAttention(emb_size=32, num_heads=4, dropout=0.1)
    >>> inputs = torch.randn(2, 10, 32)
    >>> outputs = module(inputs)
    >>> outputs.shape
    torch.Size([2, 10, 32])
    """

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        self.rearrange_stack = Rearrange(
            "b n (h d) -> b h n d",
            h=num_heads,
        )
        self.rearrange_unstack = Rearrange(
            "b h n d -> b n (h d)",
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        queries = self.rearrange_stack(self.queries(x))
        keys = self.rearrange_stack(self.keys(x))
        values = self.rearrange_stack(self.values(x))
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = float("-inf")
            energy = energy.masked_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = self.rearrange_unstack(out)
        out = self.projection(out)
        return out


class SimpleAttention(nn.Module):
    """SimA attention block from Koohpayegani and Pirsiavash (2024).

    This implements the softmax-free attention proposed in
    https://arxiv.org/html/2206.08898v2 by L1-normalizing queries and keys
    across the token axis and dynamically selecting the cheapest association
    order during the matrix product.

    Examples
    --------
    >>> import torch
    >>> module = SimpleAttention(emb_size=32, num_heads=4, dropout=0.1)
    >>> inputs = torch.randn(2, 10, 32)
    >>> outputs = module(inputs)
    >>> outputs.shape
    torch.Size([2, 10, 32])
    """

    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError(
                f"emb_size ({emb_size}) must be divisible by num_heads ({num_heads})."
            )

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        self.rearrange_stack = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.rearrange_unstack = Rearrange("b h n d -> b n (h d)")

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            raise NotImplementedError(
                "SimpleAttention does not implement masking."
            )

        queries = self.rearrange_stack(self.queries(x))
        keys = self.rearrange_stack(self.keys(x))
        values = self.rearrange_stack(self.values(x))

        # SimA normalizes each feature channel across tokens with L1 norm.
        queries = F.normalize(queries, p=1.0, dim=-2)
        keys = F.normalize(keys, p=1.0, dim=-2)

        n_tokens = queries.size(-2)
        if n_tokens < self.head_dim:
            out = torch.matmul(torch.matmul(queries, keys.transpose(-2, -1)), values)
        else:
            out = torch.matmul(queries, torch.matmul(keys.transpose(-2, -1), values))

        out = self.att_drop(out)
        out = self.rearrange_unstack(out)
        out = self.projection(out)
        return out
