"""
Attention Residuals (AttnRes) — Learned selective layer aggregation.

Implementation based on MoonshotAI (2026):
https://github.com/MoonshotAI/Attention-Residuals

Core insight: Standard residual connections use fixed weight = 1 for all
layers, causing PreNorm dilution and magnitude swelling as depth increases.
AttnRes replaces this with *learned* attention-based aggregation where each
layer selectively attends to the outputs of all previous layers.

Key Concepts
------------
1. **Learned Aggregation** — Instead of h_l = h_{l-1} + f(h_{l-1}), each
   layer computes h_l = Σ α_{i→l} · v_i, where α are learned attention
   weights over previous layer outputs.

2. **Block AttnRes** — Full AttnRes requires O(L·d) memory (storing all
   layer outputs). Block AttnRes partitions layers into N blocks, accumulates
   with standard residuals within blocks, then applies attention only between
   block representations. With ~8 blocks, captures most of Full AttnRes's
   benefit at minimal overhead.

3. **Pseudo-Query Per Layer** — Each layer has a learned query vector that
   determines which previous layers it should attend to. The attention
   mechanism is: α_{i→l} = softmax(q_l · k_i / √d), where q_l is the
   pseudo-query for layer l and k_i is a projection of layer i's output.

4. **Two-Dimension Generalization** — The same principle applies to both
   the layer dimension (depth) and the token dimension (sequence length):
   - Layer dimension: "How should layer L access info from layer 1?"
   - Token dimension: "How should token 1M access info from token 1?"
   Both are the same problem: selective aggregation → compression → fixed-size state.

Results (Kimi Linear 48B, 1.4T tokens):
- GPQA-Diamond: 36.9 → 44.4 (+7.5)
- Math: 53.5 → 57.1 (+3.6)
- HumanEval: 59.1 → 62.2 (+3.1)
- MMLU: 73.5 → 74.6 (+1.1)

Block AttnRes matches the performance of a baseline trained with 1.25×
more compute.

Credits & References:
- MoonshotAI, "Attention Residuals" (2026)
  https://github.com/MoonshotAI/Attention-Residuals
- Kimi Linear 48B benchmark results
- Analogous to Mamba's selective state updates, but in the layer dimension
  instead of the token dimension.

Hardware: Pure PyTorch. Compatible with CUDA, ROCm, and CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================


class AttnResMode(str, Enum):
    """AttnRes operation mode.

    Attributes:
        FULL: Store and attend to all previous layer outputs. O(L·d) memory.
        BLOCK: Partition layers into blocks, standard residual within blocks,
            attention between block representatives. O(N·d) memory where N
            is the number of blocks.
        HYBRID: Use Block AttnRes for the first half of layers, then Full
            AttnRes for the remaining layers (more memory, more precision
            for deeper layers).
    """
    FULL = "full"
    BLOCK = "block"
    HYBRID = "hybrid"


@dataclass
class AttnResConfig:
    """Configuration for Attention Residuals module.

    Attributes:
        d_model: Model hidden dimension.
        n_layers: Total number of layers in the model.
        mode: AttnRes operation mode (full, block, or hybrid).
        num_blocks: Number of blocks for Block AttnRes mode.
        dropout: Dropout rate for attention weights (default 0.0).
        use_gate: Whether to use a gating mechanism after aggregation
            (default True). Helps with training stability.
        temperature: Temperature for attention softmax (default 1.0).
        compression_dim: If > 0, compress layer representations to this
            dimension before storing, reducing memory. 0 means no compression.
    """
    d_model: int = 2048
    n_layers: int = 32
    mode: str = "block"
    num_blocks: int = 8
    dropout: float = 0.0
    use_gate: bool = True
    temperature: float = 1.0
    compression_dim: int = 0


# ============================================================================
# Full Attention Residuals
# ============================================================================


class FullAttnRes(nn.Module):
    """Full Attention Residuals — attend to all previous layer outputs.

    For each layer l, computes:
        h_l = Σ_{i=0}^{l} α_{i→l} · v_i

    where α_{i→l} = softmax(q_l · k_i / √d) and q_l is a learned
    pseudo-query for layer l, k_i is a projection of layer i's output.

    Memory: O(L·d) — stores all layer outputs.

    Args:
        config: AttnResConfig instance.
    """

    def __init__(self, config: AttnResConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_layers = config.n_layers

        d_store = config.compression_dim if config.compression_dim > 0 else config.d_model

        # Pseudo-query per layer: (n_layers, d_model)
        self.pseudo_queries = nn.Parameter(
            torch.randn(n_layers, config.d_model) * 0.02
        )

        # Key projection: d_model → d_store
        self.key_proj = nn.Linear(config.d_model, d_store, bias=False)

        # Value projection: d_model → d_store
        self.value_proj = nn.Linear(config.d_model, d_store, bias=False)

        # Output projection: d_store → d_model (if compressed)
        if config.compression_dim > 0:
            self.output_proj = nn.Linear(d_store, config.d_model, bias=False)
        else:
            self.output_proj = None

        # Optional gate for stability
        if config.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(config.d_model, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Scale factor
        self.scale = math.sqrt(d_store)

        # Layer output storage (populated during forward)
        self._layer_outputs: List[torch.Tensor] = []

    def reset(self) -> None:
        """Reset stored layer outputs for a new forward pass."""
        self._layer_outputs = []

    def store_layer_output(self, layer_idx: int, output: torch.Tensor) -> None:
        """Store a layer's output for later attention.

        Args:
            layer_idx: Index of the layer (0-indexed).
            output: Layer output tensor (batch, seq_len, d_model).
        """
        # Store with gradient flow when not using checkpointing
        # Gradients flow through the attention mechanism
        if layer_idx < len(self._layer_outputs):
            self._layer_outputs[layer_idx] = output
        else:
            self._layer_outputs.append(output)

    def forward(
        self,
        current_output: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute attention-weighted aggregation over previous layer outputs.

        Args:
            current_output: Current layer's output (batch, seq_len, d_model).
            layer_idx: Index of the current layer (0-indexed).

        Returns:
            Aggregated output (batch, seq_len, d_model).
        """
        if layer_idx == 0 or len(self._layer_outputs) == 0:
            # First layer: no previous outputs to attend to
            return current_output

        batch, seq_len, _ = current_output.shape
        device = current_output.device
        dtype = current_output.dtype

        # Get pseudo-query for this layer
        query = self.pseudo_queries[layer_idx]  # (d_model,)

        # Stack all previous layer outputs: (n_prev, batch, seq_len, d_model)
        prev_outputs = torch.stack(self._layer_outputs[:layer_idx], dim=0)
        n_prev = prev_outputs.shape[0]

        # Project to keys and values
        # Reshape for efficient projection: (n_prev * batch * seq_len, d_model)
        flat_prev = prev_outputs.reshape(-1, self.d_model)
        keys = self.key_proj(flat_prev).reshape(n_prev, batch, seq_len, -1)
        vals = self.value_proj(flat_prev).reshape(n_prev, batch, seq_len, -1)

        # Compute attention scores: query attends to keys
        # query: (d_model,) → project to d_store if needed
        d_store = keys.shape[-1]

        # Mean-pool each previous layer output for key representation
        # (n_prev, d_store) — one key per previous layer
        key_repr = keys.mean(dim=(1, 2))  # (n_prev, d_store)

        # Attention scores: (n_prev,)
        # Use the pseudo-query projected through key_proj space
        if d_store == self.d_model:
            q = query
        else:
            # Project query through key_proj
            q = self.key_proj(query.unsqueeze(0)).squeeze(0)  # (d_store,)

        scores = torch.matmul(key_repr, q) / self.scale  # (n_prev,)

        # Apply temperature
        scores = scores / self.config.temperature

        # Softmax over previous layers
        attn_weights = F.softmax(scores, dim=0)  # (n_prev,)

        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of value representations
        # Mean-pool values per layer: (n_prev, d_store)
        val_repr = vals.mean(dim=(1, 2))  # (n_prev, d_store)

        # Weighted aggregation
        aggregated = torch.matmul(attn_weights.unsqueeze(0), val_repr)  # (1, d_store)
        aggregated = aggregated.squeeze(0)  # (d_store,)

        # Project back to d_model if compressed
        if self.output_proj is not None:
            aggregated = self.output_proj(aggregated)  # (d_model,)

        # Expand to (batch, seq_len, d_model)
        aggregated = aggregated.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)

        # Optional gating
        if self.gate is not None:
            gate_val = self.gate(current_output)  # (batch, seq_len, 1)
            aggregated = gate_val * aggregated

        # Residual: combine with current output
        output = current_output + aggregated

        return output.to(dtype)


# ============================================================================
# Block Attention Residuals
# ============================================================================


class BlockAttnRes(nn.Module):
    """Block Attention Residuals — efficient approximation with block partitioning.

    Partitions layers into N blocks. Within each block, uses standard residual
    connections. Between blocks, applies attention-based aggregation where each
    block representative attends to previous block representatives.

    Memory: O(N·d) where N is the number of blocks. With ~8 blocks, captures
    most of Full AttnRes's benefit at minimal overhead.

    Algorithm:
    1. Partition layers [0, L) into N blocks of size block_size = L / N
    2. Within block b: standard residual h_l = h_{l-1} + f(h_{l-1})
    3. At block boundary: compute block representative r_b = mean(h_l for l in block b)
    4. Apply attention: h_b_aggregated = Σ α_{i→b} · r_i for i < b
    5. Initialize next block with aggregated representation

    This is provably efficient: with N = O(log L) blocks, captures most of
    the benefit of Full AttnRes while using O(log L · d) memory.

    Args:
        config: AttnResConfig instance.
    """

    def __init__(self, config: AttnResConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.num_blocks = config.num_blocks

        d_store = config.compression_dim if config.compression_dim > 0 else config.d_model

        # Block size
        self.block_size = max(1, config.n_layers // config.num_blocks)

        # Actual number of blocks (might differ due to rounding)
        self.actual_blocks = math.ceil(config.n_layers / self.block_size)

        # Pseudo-query per block: (actual_blocks, d_model)
        self.block_queries = nn.Parameter(
            torch.randn(self.actual_blocks, config.d_model) * 0.02
        )

        # Key/value projections
        self.key_proj = nn.Linear(config.d_model, d_store, bias=False)
        self.value_proj = nn.Linear(config.d_model, d_store, bias=False)

        # Output projection (if compressed)
        if config.compression_dim > 0:
            self.output_proj = nn.Linear(d_store, config.d_model, bias=False)
        else:
            self.output_proj = None

        # Gate
        if config.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(config.d_model, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        self.scale = math.sqrt(d_store)

        # Block representatives storage
        self._block_representatives: List[torch.Tensor] = []
        self._current_block_outputs: List[torch.Tensor] = []
        self._current_block_idx: int = 0

    def reset(self) -> None:
        """Reset for a new forward pass."""
        self._block_representatives = []
        self._current_block_outputs = []
        self._current_block_idx = 0

    def _get_block_idx(self, layer_idx: int) -> int:
        """Get block index for a given layer."""
        return layer_idx // self.block_size

    def _is_block_boundary(self, layer_idx: int) -> bool:
        """Check if layer_idx is at the end of a block."""
        next_block = self._get_block_idx(layer_idx + 1)
        current_block = self._get_block_idx(layer_idx)
        return next_block != current_block or layer_idx == self.n_layers - 1

    def store_layer_output(self, layer_idx: int, output: torch.Tensor) -> None:
        """Store layer output and compute block representative at boundaries.

        Args:
            layer_idx: Layer index (0-indexed).
            output: Layer output tensor.
        """
        self._current_block_outputs.append(output)

        if self._is_block_boundary(layer_idx):
            # Compute block representative: mean of all outputs in this block
            block_repr = torch.stack(self._current_block_outputs, dim=0).mean(dim=0)
            self._block_representatives.append(block_repr)
            self._current_block_outputs = []

    def forward(
        self,
        current_output: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute block attention aggregation at block boundaries.

        Within a block, returns current_output unchanged (standard residual
        is handled by the model). At block boundaries, aggregates information
        from previous blocks via attention.

        Args:
            current_output: Current layer output.
            layer_idx: Current layer index.

        Returns:
            Aggregated output at block boundaries, or current_output unchanged.
        """
        if not self._is_block_boundary(layer_idx):
            return current_output

        current_block_idx = self._get_block_idx(layer_idx)

        if current_block_idx == 0 or len(self._block_representatives) <= 1:
            # First block: no previous blocks to attend to
            return current_output

        batch, seq_len, _ = current_output.shape
        dtype = current_output.dtype

        # Previous block representatives (exclude current block)
        prev_reprs = self._block_representatives[:-1]  # Exclude current
        n_prev = len(prev_reprs)

        if n_prev == 0:
            return current_output

        # Stack previous representatives: (n_prev, batch, seq_len, d_model)
        stacked = torch.stack(prev_reprs, dim=0)

        # Compute key representations per block: (n_prev, d_store)
        flat = stacked.reshape(-1, self.d_model)
        keys = self.key_proj(flat).reshape(n_prev, stacked.shape[1], stacked.shape[2], -1)
        key_repr = keys.mean(dim=(1, 2))  # (n_prev, d_store)

        # Get query for current block
        q_idx = min(current_block_idx, self.actual_blocks - 1)
        query = self.block_queries[q_idx]  # (d_model,)

        d_store = key_repr.shape[-1]
        if d_store != self.d_model:
            q = self.key_proj(query.unsqueeze(0)).squeeze(0)
        else:
            q = query

        # Attention scores
        scores = torch.matmul(key_repr, q) / self.scale
        scores = scores / self.config.temperature
        attn_weights = F.softmax(scores, dim=0)  # (n_prev,)

        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights)

        # Value projection and aggregation
        vals = self.value_proj(flat).reshape(n_prev, stacked.shape[1], stacked.shape[2], -1)
        val_repr = vals.mean(dim=(1, 2))  # (n_prev, d_store)

        aggregated = torch.matmul(attn_weights.unsqueeze(0), val_repr).squeeze(0)

        if self.output_proj is not None:
            aggregated = self.output_proj(aggregated)

        # Expand: (batch, seq_len, d_model)
        aggregated = aggregated.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)

        # Gate
        if self.gate is not None:
            gate_val = self.gate(current_output)
            aggregated = gate_val * aggregated

        output = current_output + aggregated

        return output.to(dtype)


# ============================================================================
# Token-Dimension AttnRes + Compression
# ============================================================================


class TokenAttnResCompression(nn.Module):
    """AttnRes + Compression applied in the token (sequence) dimension.

    This is the key innovation from the architecture document: apply the
    same AttnRes principle (selective attention + compression) in the token
    dimension to achieve O(n) complexity with intelligent forgetting.

    Instead of Mamba's forced forgetting (A < 1), this module:
    1. Applies attention to select which past tokens are most relevant
    2. Compresses the selected representation to a fixed-size hidden state
    3. The hidden state serves as "long-term memory" with O(d) size

    Three compression options:
    - Linear: h_compressed = W · h_attnres (simple, uniform info loss)
    - Gated: gate = sigmoid(W_g · h), h_compressed = gate * h (selective)
    - SSM: Use Mamba-style SSM as compressor (AttnRes + SSM together)

    Args:
        d_model: Model hidden dimension.
        d_state: Compressed hidden state dimension.
        compression_type: "linear", "gated", or "ssm".
        chunk_size: Size of chunks for chunked processing.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 256,
        compression_type: str = "gated",
        chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.compression_type = compression_type
        self.chunk_size = chunk_size

        # Learned query for token-level attention (projected to key space)
        self.token_query = nn.Parameter(torch.randn(d_model // 4) * 0.02)

        # Key projection
        self.key_proj = nn.Linear(d_model, d_model // 4, bias=False)

        # Value projection
        self.value_proj = nn.Linear(d_model, d_model // 4, bias=False)

        # Compression layer
        if compression_type == "linear":
            self.compress = nn.Linear(d_model // 4, d_state, bias=False)
        elif compression_type == "gated":
            self.compress_gate = nn.Linear(d_model // 4, d_state, bias=False)
            self.compress_value = nn.Linear(d_model // 4, d_state, bias=False)
        elif compression_type == "ssm":
            # SSM-based compressor: uses SSM state transitions
            self.compress_proj = nn.Linear(d_model // 4, d_state, bias=False)
            self.ssm_A = nn.Parameter(torch.randn(d_state) * 0.5 - 1.0)  # Stable init
            self.ssm_B = nn.Linear(d_model // 4, d_state, bias=False)
            self.ssm_C = nn.Linear(d_state, d_model // 4, bias=False)

        # Output projection from compressed state
        self.decompress = nn.Linear(d_state, d_model, bias=False)

        # Gate for combining with current representation
        self.output_gate = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),
            nn.Sigmoid(),
        )

        self.scale = math.sqrt(d_model // 4)

    def _compress_gated(self, x: torch.Tensor) -> torch.Tensor:
        """Gated compression."""
        gate = torch.sigmoid(self.compress_gate(x))
        value = self.compress_value(x)
        return gate * value

    def _update_ssm_state(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Update SSM compressed state.

        h(t) = A · h(t-1) + B · x(t)
        """
        dt = 0.1  # Fixed small dt for stability
        A = -torch.exp(self.ssm_A)  # Ensure negative (stable)
        B = self.ssm_B(x)
        new_state = state * torch.exp(dt * A.unsqueeze(0)) + dt * B
        return new_state

    def forward(
        self,
        x: torch.Tensor,
        compressed_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: selective attention + compression.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            compressed_state: Previous compressed state (batch, d_state).
                If None, initialized to zeros.

        Returns:
            Tuple of:
            - output: (batch, seq_len, d_model) — enriched representation
            - new_state: (batch, d_state) — updated compressed state
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # Initialize state if needed
        if compressed_state is None:
            compressed_state = torch.zeros(
                batch, self.d_state, device=device, dtype=dtype
            )

        # Project to keys and values
        keys = self.key_proj(x)  # (batch, seq_len, d_model//4)
        vals = self.value_proj(x)  # (batch, seq_len, d_model//4)

        # Compute attention: query attends to keys
        q = self.token_query.to(dtype=dtype)
        # Key representation: use the projected keys directly
        scores = torch.matmul(keys, q.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
        scores = scores / self.scale
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum of values
        attn_output = torch.matmul(
            attn_weights.unsqueeze(1), vals
        ).squeeze(1)  # (batch, d_model//4)

        # Compress
        if self.compression_type == "linear":
            compressed = self.compress(attn_output)
        elif self.compression_type == "gated":
            compressed = self._compress_gated(attn_output)
        elif self.compression_type == "ssm":
            compressed = self._update_ssm_state(compressed_state, attn_output)
        else:
            compressed = self.compress_proj(attn_output) if hasattr(self, 'compress_proj') else compressed_state

        # Decompress back to d_model
        decompressed = self.decompress(compressed)  # (batch, d_model)

        # Expand to sequence length and gate
        decompressed = decompressed.unsqueeze(1).expand(-1, seq_len, -1)
        gate = self.output_gate(x)  # (batch, seq_len, 1)
        enriched = x + gate * decompressed

        return enriched, compressed


# ============================================================================
# AttnRes Manager — Coordinates Full/Block/Hybrid modes
# ============================================================================


class AttnResManager(nn.Module):
    """Manages AttnRes across all layers of the model.

    Provides a unified interface for storing layer outputs and computing
    attention-based residual aggregation. Supports Full, Block, and Hybrid
    modes.

    Usage in LosionModelV2:
        # Before layer loop:
        attn_res.reset()

        # After each layer:
        output = layer(x)
        output = attn_res(output, layer_idx)
        attn_res.store_layer_output(layer_idx, output)

    Args:
        config: AttnResConfig instance.
    """

    def __init__(self, config: AttnResConfig) -> None:
        super().__init__()
        self.config = config
        self.mode = AttnResMode(config.mode)

        if self.mode == AttnResMode.FULL:
            self.attn_res = FullAttnRes(config)
        elif self.mode == AttnResMode.BLOCK:
            self.attn_res = BlockAttnRes(config)
        elif self.mode == AttnResMode.HYBRID:
            # Use Block for first half, Full for second half
            block_config = AttnResConfig(**{**config.__dict__, 'mode': 'block'})
            full_config = AttnResConfig(**{**config.__dict__, 'mode': 'full'})
            self.attn_res_block = BlockAttnRes(block_config)
            self.attn_res_full = FullAttnRes(full_config)
            self.attn_res = None  # Not used directly in hybrid mode
        else:
            raise ValueError(f"Unknown AttnRes mode: {config.mode}")

    def reset(self) -> None:
        """Reset for a new forward pass."""
        if self.mode == AttnResMode.HYBRID:
            self.attn_res_block.reset()
            self.attn_res_full.reset()
        else:
            self.attn_res.reset()

    def store_layer_output(self, layer_idx: int, output: torch.Tensor) -> None:
        """Store a layer's output."""
        if self.mode == AttnResMode.HYBRID:
            self.attn_res_block.store_layer_output(layer_idx, output)
            self.attn_res_full.store_layer_output(layer_idx, output)
        else:
            self.attn_res.store_layer_output(layer_idx, output)

    def forward(
        self,
        current_output: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute AttnRes aggregation for the current layer.

        Args:
            current_output: Current layer's output.
            layer_idx: Current layer index.

        Returns:
            Aggregated output.
        """
        if self.mode == AttnResMode.HYBRID:
            half = self.config.n_layers // 2
            if layer_idx < half:
                return self.attn_res_block(current_output, layer_idx)
            else:
                return self.attn_res_full(current_output, layer_idx)
        else:
            return self.attn_res(current_output, layer_idx)

    def get_stats(self) -> Dict[str, object]:
        """Get statistics about the AttnRes module."""
        return {
            "mode": self.config.mode,
            "n_layers": self.config.n_layers,
            "num_blocks": self.config.num_blocks if self.config.mode == "block" else "N/A",
            "compression_dim": self.config.compression_dim,
            "use_gate": self.config.use_gate,
        }
