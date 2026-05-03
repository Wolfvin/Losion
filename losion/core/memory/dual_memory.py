"""
Two-Level Memory System — Working Memory + Long-Term Memory.

Implementation of the architecture document's insight (Section 11.4):
AttnRes + Compression naturally produces a two-level memory system:

    Working Memory:  direct access to recent layer/token outputs (high detail)
    Long-Term Memory: compressed hidden state from AttnRes (selective summary)

Analogous to human memory:
    Working memory  ↔  recent outputs (direct, detailed access)
    Long-term memory ↔  compressed AttnRes state (selective, persistent)
    Memory consolidation ↔  AttnRes attention weights (selective compression)

Credits & References:
    - Losion Architecture Document Section 11.4: Two-Level Memory
    - MoonshotAI Attention Residuals (2026)
    - Human memory systems: Baddeley's working memory model

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
class DualMemoryConfig:
    """Configuration for Two-Level Memory System."""
    d_model: int = 2048
    working_memory_size: int = 512
    long_term_memory_dim: int = 256
    consolidation_method: str = "attention"
    retrieval_method: str = "attention"
    n_retrieval_heads: int = 4
    dropout: float = 0.0


class WorkingMemory(nn.Module):
    """Working Memory: direct access to recent token/layer outputs.

    Stores the most recent representations with full fidelity (ring buffer).
    Like human working memory: high detail, limited capacity, direct access.
    """

    def __init__(self, d_model: int, capacity: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self.capacity = capacity
        self.register_buffer("buffer", torch.zeros(capacity, d_model), persistent=False)
        self.register_buffer("occupation", torch.zeros(capacity, dtype=torch.bool), persistent=False)
        self._write_ptr: int = 0
        self._count: int = 0

    def write(self, entries: torch.Tensor) -> None:
        n = entries.shape[0]
        for i in range(n):
            idx = (self._write_ptr + i) % self.capacity
            self.buffer[idx] = entries[i].detach()
            self.occupation[idx] = True
        self._write_ptr = (self._write_ptr + n) % self.capacity
        self._count = min(self._count + n, self.capacity)

    def read_all(self) -> torch.Tensor:
        if not self.occupation.any():
            return torch.zeros(0, self.d_model, device=self.buffer.device)
        indices = self.occupation.nonzero(as_tuple=True)[0]
        return self.buffer[indices]

    def read_recent(self, n: int) -> torch.Tensor:
        if self._count == 0:
            return torch.zeros(0, self.d_model, device=self.buffer.device)
        n = min(n, self._count)
        indices = [(self._write_ptr - 1 - i) % self.capacity for i in range(n)]
        return self.buffer[torch.tensor(indices, device=self.buffer.device)]

    def clear(self) -> None:
        self.buffer.zero_()
        self.occupation.zero_()
        self._write_ptr = 0
        self._count = 0

    def get_occupation_ratio(self) -> float:
        return self._count / self.capacity


class LongTermMemory(nn.Module):
    """Long-Term Memory: compressed, persistent hidden state.

    Stores compressed representations from AttnRes + Compression.
    Like human long-term memory: selective, persistent, compressed.
    """

    def __init__(self, d_model: int, d_state: int = 256, consolidation_method: str = "attention") -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.consolidation_method = consolidation_method

        self.state_proj = nn.Linear(d_model, d_state, bias=False)

        if consolidation_method == "attention":
            self.query = nn.Parameter(torch.randn(d_state) * 0.02)
            self.key_proj = nn.Linear(d_model, d_state, bias=False)
            self.value_proj = nn.Linear(d_model, d_state, bias=False)
            self.scale = math.sqrt(d_state)

        if consolidation_method == "gated":
            self.gate = nn.Sequential(
                nn.Linear(d_state, d_state, bias=False),
                nn.Sigmoid(),
            )

        self.output_proj = nn.Linear(d_state, d_model, bias=False)
        self.register_buffer("compressed_state", torch.zeros(d_state), persistent=False)

    def consolidate(self, working_memory_entries: torch.Tensor) -> torch.Tensor:
        if working_memory_entries.shape[0] == 0:
            return self.compressed_state

        if self.consolidation_method == "attention":
            keys = self.key_proj(working_memory_entries)
            values = self.value_proj(working_memory_entries)
            q = self.query
            scores = torch.matmul(keys, q) / self.scale
            attn = F.softmax(scores, dim=0)
            new_state = torch.matmul(attn.unsqueeze(0), values).squeeze(0)
        elif self.consolidation_method == "gated":
            projected = self.state_proj(working_memory_entries.mean(dim=0))
            gate = self.gate(projected)
            new_state = gate * projected + (1 - gate) * self.compressed_state
        else:
            new_state = self.state_proj(working_memory_entries.mean(dim=0))

        self.compressed_state = 0.9 * self.compressed_state + 0.1 * new_state
        return self.compressed_state

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        retrieved = self.output_proj(self.compressed_state)
        if query.dim() == 3:
            retrieved = retrieved.unsqueeze(0).unsqueeze(0).expand_as(query)
        elif query.dim() == 2:
            retrieved = retrieved.unsqueeze(0).expand_as(query)
        return retrieved


class DualMemorySystem(nn.Module):
    """Two-Level Memory System: Working Memory + Long-Term Memory.

    Coordinates both memory levels with consolidation and retrieval.
    Integration with LosionModelV2: write after each layer, consolidate
    at block boundaries, retrieve when generating.
    """

    def __init__(self, config: Optional[DualMemoryConfig] = None) -> None:
        super().__init__()
        self.config = config or DualMemoryConfig()
        self.d_model = self.config.d_model

        self.working_memory = WorkingMemory(
            d_model=self.d_model,
            capacity=self.config.working_memory_size,
        )
        self.long_term_memory = LongTermMemory(
            d_model=self.d_model,
            d_state=self.config.long_term_memory_dim,
            consolidation_method=self.config.consolidation_method,
        )

        self.retrieval_gate = nn.Sequential(
            nn.Linear(self.d_model, 2, bias=False),
            nn.Softmax(dim=-1),
        )
        self.working_retrieve_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def write(self, x: torch.Tensor) -> None:
        if x.dim() == 3:
            entries = x.reshape(-1, self.d_model).detach()
        else:
            entries = x.detach()
        self.working_memory.write(entries)

    def consolidate(self) -> None:
        entries = self.working_memory.read_all()
        if entries.shape[0] > 0:
            self.long_term_memory.consolidate(entries)

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object]]:
        wm_entries = self.working_memory.read_recent(64)
        if wm_entries.shape[0] > 0:
            if query.dim() == 3:
                q = query.mean(dim=(0, 1))
            elif query.dim() == 2:
                q = query.mean(dim=0)
            else:
                q = query
            scores = F.cosine_similarity(q.unsqueeze(0), wm_entries, dim=-1)
            best_idx = scores.argmax()
            wm_output = self.working_retrieve_proj(wm_entries[best_idx])
        else:
            wm_output = torch.zeros(self.d_model, device=query.device)

        ltm_output = self.long_term_memory.retrieve(query)

        gate_input = query.reshape(-1, self.d_model) if query.dim() != 2 else query
        flat = gate_input.mean(dim=0 if gate_input.dim() == 2 else 0)
        gates = self.retrieval_gate(flat)
        wm_weight, ltm_weight = gates[0], gates[1]

        if query.dim() == 3:
            combined = wm_weight * wm_output.unsqueeze(0).unsqueeze(0).expand_as(query) + \
                       ltm_weight * ltm_output
        elif query.dim() == 2:
            combined = wm_weight * wm_output.unsqueeze(0).expand_as(query) + \
                       ltm_weight * ltm_output
        else:
            combined = wm_weight * wm_output + ltm_weight * ltm_output

        info = {
            "working_memory_occupation": self.working_memory.get_occupation_ratio(),
            "wm_weight": wm_weight.item(),
            "ltm_weight": ltm_weight.item(),
        }
        return combined, info

    def clear(self) -> None:
        self.working_memory.clear()

    def get_stats(self) -> Dict[str, object]:
        return {
            "working_memory_occupation": self.working_memory.get_occupation_ratio(),
            "long_term_memory_norm": self.long_term_memory.compressed_state.norm().item(),
        }
