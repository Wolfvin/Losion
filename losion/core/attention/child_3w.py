"""
Child-3W Routing — MoE at the Attention QKV Level.

Implementation of the architecture document's "Router + Child-3W" concept
(Sections 5-6): routing at the Wq/Wk/Wv level with multiple child attention
parameter sets.

Core Concept:
    Instead of one set of Wq/Wk/Wv that must compromise for all contexts,
    provide MANY child-3W sets, each specializing in different types of
    attention patterns. A router selects which child-3W sets to activate.

    This is MORE GRANULAR than standard MoE:
    - MoE: router selects experts at the FFN layer (output-level specialization)
    - Child-3W: router selects experts at the attention level (representation-level
      specialization at Q, K, V projections)

    Key Insight: The router doesn't need to be designed manually. Just like
    attention heads spontaneously specialize in large LLMs, the Child-3W
    router will discover its own routing patterns through backpropagation —
    as long as the architecture provides separate pathways.

    Multiple Child-3W sets can be active simultaneously:
    - Generalist: blend all Child-3W sets with equal weights
    - Specialist: one Child-3W set dominant
    - Multi-domain: several Child-3W sets active with different weights

Comparison with MoE:
    | MoE Standard            | Router + Child-3W                      |
    |-------------------------|----------------------------------------|
    | Router at FFN layer     | Router at attention (3W) level         |
    | Separation at output    | Separation at Q/K/V representation    |
    | One expert per token    | Multiple children active simultaneously|

Credits & References:
    - Losion Architecture Document Sections 5-6: Router + Child-3W
    - Mixtral, GPT-4: MoE at FFN level (inspiration for granularity)
    - MoonshotAI: MoBA (attention-level routing, but different approach)
    - Multi-head attention spontaneous specialization in LLMs

Hardware: Pure PyTorch. Compatible with CUDA, ROCm, and CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Child3WConfig:
    """Configuration for Child-3W routing.

    Attributes:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads per child.
        d_kv: Dimension per attention head.
        num_children: Number of Child-3W sets (default 4).
        top_k_children: Number of active children per token (default 2).
        use_mla: Whether to use Multi-head Latent Attention compression.
        mla_latent_dim: MLA latent dimension (default 0 = no compression).
        dropout: Dropout rate.
        load_balance_weight: Weight for auxiliary load balancing loss.
    """
    d_model: int = 2048
    n_heads: int = 8
    d_kv: int = 256
    num_children: int = 4
    top_k_children: int = 2
    use_mla: bool = False
    mla_latent_dim: int = 0
    dropout: float = 0.0
    load_balance_weight: float = 0.01


class Child3WSet(nn.Module):
    """A single Child-3W set: independent Wq, Wk, Wv projections.

    Each child has its own set of query, key, value projection matrices,
    allowing it to specialize in different attention patterns.

    Args:
        d_model: Model dimension.
        n_heads: Number of heads.
        d_kv: Dimension per head.
    """

    def __init__(self, d_model: int, n_heads: int, d_kv: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_kv = d_kv

        self.wq = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.wk = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.wv = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.wo = nn.Linear(n_heads * d_kv, d_model, bias=False)

        # Initialize with small weights for stability
        nn.init.normal_(self.wq.weight, std=0.02)
        nn.init.normal_(self.wk.weight, std=0.02)
        nn.init.normal_(self.wv.weight, std=0.02)
        nn.init.zeros_(self.wo.weight)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention with this child's Wq/Wk/Wv.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            attention_mask: Optional attention mask.

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.n_heads, self.d_kv).transpose(1, 2)

        # Standard attention
        scale = math.sqrt(self.d_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_mask is not None:
            scores = scores + attention_mask
        else:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(output)


class Child3WRouter(nn.Module):
    """Router for Child-3W sets: selects which children to activate.

    Produces routing weights for each token, selecting the top-K children
    with the highest affinity. Uses bias-based load balancing (DeepSeek-V3 style).

    Args:
        d_model: Model dimension.
        num_children: Number of Child-3W sets.
        top_k: Number of active children per token.
    """

    def __init__(self, d_model: int, num_children: int, top_k: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_children = num_children
        self.top_k = min(top_k, num_children)

        # Router projection
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_children, bias=False),
        )

        # Bias for load balancing (DeepSeek-V3 style)
        self.bias = nn.Parameter(torch.zeros(num_children))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights for Child-3W sets.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Tuple (weights, indices, router_logits):
            - weights: (batch, seq_len, top_k) — routing weights
            - indices: (batch, seq_len, top_k) — child indices
            - router_logits: (batch, seq_len, num_children) — raw logits
        """
        logits = self.gate(x) + self.bias.unsqueeze(0).unsqueeze(0)

        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices, logits


class Child3WAttention(nn.Module):
    """Child-3W Attention: MoE routing at the QKV level.

    Multiple Child-3W sets, each with independent Wq/Wk/Wv projections.
    A router selects which children to activate for each token.
    Active children's outputs are merged with weighted combination.

    This replaces standard multi-head attention with a more flexible,
    context-adaptive approach where different attention patterns are
    used for different types of content.

    Usage: Drop-in replacement for standard attention in LosionModelV2.

    Args:
        config: Child3WConfig instance.
    """

    def __init__(self, config: Optional[Child3WConfig] = None) -> None:
        super().__init__()
        self.config = config or Child3WConfig()
        self.d_model = self.config.d_model
        self.num_children = self.config.num_children
        self.top_k = self.config.top_k_children

        # MLA compression (optional)
        if self.config.use_mla and self.config.mla_latent_dim > 0:
            self.mla_compress = nn.Linear(self.d_model, self.config.mla_latent_dim, bias=False)
            self.mla_decompress = nn.Linear(self.config.mla_latent_dim, self.d_model, bias=False)
            child_d_model = self.config.mla_latent_dim
        else:
            self.mla_compress = None
            self.mla_decompress = None
            child_d_model = self.d_model

        # Child-3W sets
        self.child_sets = nn.ModuleList([
            Child3WSet(
                d_model=child_d_model,
                n_heads=self.config.n_heads,
                d_kv=self.config.d_kv,
            )
            for _ in range(self.config.num_children)
        ])

        # Router
        self.router = Child3WRouter(
            d_model=self.d_model,
            num_children=self.config.num_children,
            top_k=self.config.top_k_children,
        )

        # Output norm
        self.output_norm = nn.RMSNorm(self.d_model)

        # Dropout
        self.dropout = nn.Dropout(self.config.dropout) if self.config.dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: route tokens to Child-3W sets and merge.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            attention_mask: Optional attention mask.

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Optional MLA compression
        if self.mla_compress is not None:
            child_input = self.mla_compress(x)
        else:
            child_input = x

        # Route tokens to children
        weights, indices, router_logits = self.router(x)

        # Compute each child's output
        child_outputs = []
        for child in self.child_sets:
            child_out = child(child_input, attention_mask)
            child_outputs.append(child_out)

        # Stack: (num_children, batch, seq_len, d_model)
        all_child_outputs = torch.stack(child_outputs, dim=0)

        # Select and merge based on routing
        output = torch.zeros(batch, seq_len, self.d_model, device=x.device, dtype=x.dtype)

        # Expand child outputs if MLA was used
        if self.mla_decompress is not None:
            decompressed = torch.stack([
                self.mla_decompress(all_child_outputs[i]) for i in range(self.num_children)
            ], dim=0)
        else:
            decompressed = all_child_outputs

        for k in range(self.top_k):
            # Get weights and indices for this top-k position
            w = weights[:, :, k:k+1]  # (batch, seq_len, 1)
            idx = indices[:, :, k]  # (batch, seq_len)

            # Gather child outputs
            for child_id in range(self.num_children):
                mask = (idx == child_id)  # (batch, seq_len)
                if mask.any():
                    child_out = decompressed[child_id]  # (batch, seq_len, d_model)
                    output[mask] += (w * child_out)[mask]

        # Norm and optional dropout
        output = self.output_norm(output)
        if self.dropout is not None:
            output = self.dropout(output)

        # Store router logits for auxiliary loss
        self._last_router_logits = router_logits

        return output

    def compute_load_balance_loss(self) -> torch.Tensor:
        """Compute auxiliary load balancing loss (DeepSeek-V3 style).

        Encourages all children to receive roughly equal routing probability.

        Returns:
            Scalar loss.
        """
        if not hasattr(self, '_last_router_logits'):
            return torch.tensor(0.0)

        logits = self._last_router_logits  # (batch, seq, num_children)
        probs = F.softmax(logits, dim=-1)

        # Switch-style load balance loss
        assignments = probs.argmax(dim=-1)
        one_hot = F.one_hot(assignments, self.num_children).float()
        f = one_hot.mean(dim=(0, 1))  # Fraction of tokens per child
        P = probs.mean(dim=(0, 1))    # Mean routing probability

        loss = self.num_children * (f * P).sum() * self.config.load_balance_weight
        return loss
