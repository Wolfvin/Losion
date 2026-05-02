"""
FlowMatchingDecoder — Optional refinement untuk output.

Diadaptasi dari CogView4/GLM-Image: flow matching hanya memerlukan
2-3 langkah refinement karena starting point sudah bermakna
(berkat AR stage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowMatchingOutput:
    """Output dari Flow Matching Decoder."""

    refined_logits: torch.Tensor  # [batch, seq, vocab_size] — logits setelah refinement
    num_steps: int  # Jumlah langkah refinement yang dilakukan
    trajectory: Optional[List[torch.Tensor]]  # Trajectory intermediate (opsional)


class FlowStep(nn.Module):
    """
    Satu langkah flow matching.

    Memprediksi velocity field (arah perbaikan) dari state saat ini
    pada waktu t. Berbeda dari DDPM yang memprediksi noise,
    flow matching memprediksi velocity = (target - current).

    Args:
        d_model: Model dimension
        vocab_size: Vocabulary size
        time_embed_dim: Dimensi time embedding (default: d_model // 4)
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        time_embed_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.time_embed_dim = time_embed_dim or d_model // 4

        # Time embedding: sinusoidal → MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Velocity prediction network
        # Input: current_state + time_embedding
        self.velocity_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Output projection ke vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    @staticmethod
    def sinusoidal_embedding(
        t: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """
        Hitung sinusoidal time embedding.

        Args:
            t: Time value [batch, 1] atau [batch]
            dim: Embedding dimension

        Returns:
            Time embedding [batch, dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [batch, 1]

        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb
        )
        emb = t * emb.unsqueeze(0)  # [batch, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [batch, dim]

        # Handle odd dim
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity field pada waktu t.

        Args:
            x: Current state [batch, seq, d_model]
            t: Time value [batch] — nilai antara 0 dan 1

        Returns:
            Velocity prediction [batch, seq, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Time embedding
        t_emb = self.sinusoidal_embedding(t, self.time_embed_dim)  # [batch, time_dim]
        t_emb = self.time_mlp(t_emb)  # [batch, d_model]

        # Expand time embedding ke sequence
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, d_model]

        # Predict velocity
        velocity_input = torch.cat([x, t_emb], dim=-1)  # [batch, seq, d_model*2]
        velocity = self.velocity_net(velocity_input)  # [batch, seq, d_model]

        return velocity


class FlowMatchingDecoder(nn.Module):
    """
    Flow Matching Decoder — optional refinement untuk output.
    Diadaptasi dari CogView4/GLM-Image.

    Alih-alih DDPM yang memerlukan 100-1000 sampling steps,
    flow matching hanya memerlukan 2-3 langkah karena
    starting point sudah bermakna (berkat AR stage).

    Arsitektur:
    - Input: logits dari AR stage (starting point yang sudah bermakna)
    - Flow steps memprediksi velocity field untuk refinement
    - Euler integration untuk sampling
    - Output: refined logits dengan kualitas lebih tinggi

    Flow matching formula:
    - x_0 = initial logits (dari AR stage)
    - x_1 = refined logits (target)
    - dx/dt = v(x, t) — velocity field
    - x_{t+dt} = x_t + v(x_t, t) * dt — Euler step

    Args:
        d_model: Model dimension
        vocab_size: Vocabulary size
        num_steps: Number of refinement steps (default 3)
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        num_steps: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_steps = max(1, num_steps)

        # Logit ↔ hidden state converters
        self.logits_to_hidden = nn.Linear(vocab_size, d_model, bias=False)
        self.hidden_to_logits = nn.Linear(d_model, vocab_size, bias=False)

        # Flow step networks
        # Beberapa step bisa berbagi parameter (weight sharing)
        # atau memiliki parameter masing-masing
        self.flow_steps = nn.ModuleList(
            [FlowStep(d_model, vocab_size) for _ in range(self.num_steps)]
        )

        # Layer normalization
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        # Scheduler: menentukan time points untuk setiap step
        # Uniform schedule: langkah merata dari t=0 ke t=1
        self.register_buffer(
            "time_schedule",
            torch.linspace(0, 1, self.num_steps + 1),
        )

    def forward(
        self,
        initial_logits: torch.Tensor,
        return_trajectory: bool = False,
    ) -> FlowMatchingOutput:
        """
        Forward pass: refine logits melalui flow matching.

        Proses:
        1. Konversi logits → hidden state
        2. Iterative refinement dengan flow steps
        3. Konversi hidden state → refined logits

        Args:
            initial_logits: Logits dari AR stage [batch, seq, vocab_size]
            return_trajectory: Jika True, return intermediate states

        Returns:
            FlowMatchingOutput dengan refined logits
        """
        if initial_logits.dim() != 3:
            raise ValueError(
                f"Input harus 3D [batch, seq, vocab_size], mendapat {initial_logits.dim()}D"
            )

        batch_size, seq_len, vocab_size = initial_logits.shape

        if vocab_size != self.vocab_size:
            raise ValueError(
                f"Vocab size mismatch: input={vocab_size}, model={self.vocab_size}"
            )

        # Konversi logits → hidden state
        x = self.logits_to_hidden(initial_logits)  # [batch, seq, d_model]
        x = self.input_norm(x)

        # Trajectory recording
        trajectory: List[torch.Tensor] = []
        if return_trajectory:
            trajectory.append(x.clone())

        # Iterative refinement
        for step_idx in range(self.num_steps):
            # Time values untuk step ini
            t_start = self.time_schedule[step_idx]
            t_end = self.time_schedule[step_idx + 1]
            dt = t_end - t_start

            # Current time (sama untuk seluruh batch)
            t = t_start.expand(batch_size).to(x.device)

            # Predict velocity
            velocity = self.flow_steps[step_idx](x, t)

            # Euler integration step
            x = x + velocity * dt

            if return_trajectory:
                trajectory.append(x.clone())

        # Normalize output
        x = self.output_norm(x)

        # Konversi hidden state → refined logits
        refined_logits = self.hidden_to_logits(x)

        return FlowMatchingOutput(
            refined_logits=refined_logits,
            num_steps=self.num_steps,
            trajectory=trajectory if return_trajectory else None,
        )

    def compute_loss(
        self,
        initial_logits: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hitung flow matching training loss.

        Loss = MSE antara predicted velocity dan target velocity.
        Target velocity = (target - current) / (1 - t)

        Training strategy:
        - Sample random time t ∈ [0, 1]
        - Interpolate: x_t = (1 - t) * x_0 + t * x_1
          di mana x_0 = initial, x_1 = target
        - Target velocity = x_1 - x_0
        - Predicted velocity = flow_step(x_t, t)
        - Loss = MSE(predicted, target)

        Args:
            initial_logits: Logits dari AR stage [batch, seq, vocab_size]
            target_hidden: Target hidden state [batch, seq, d_model]

        Returns:
            Loss scalar
        """
        batch_size = initial_logits.shape[0]
        device = initial_logits.device
        dtype = initial_logits.dtype

        # Konversi logits → hidden state
        x_0 = self.logits_to_hidden(initial_logits)  # [batch, seq, d_model]
        x_0 = self.input_norm(x_0)
        x_1 = target_hidden  # [batch, seq, d_model]

        # Sample random time
        t = torch.rand(batch_size, device=device, dtype=dtype)

        # Interpolate: x_t = (1 - t) * x_0 + t * x_1
        t_expand = t.view(-1, 1, 1)  # [batch, 1, 1]
        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        # Target velocity = x_1 - x_0 (constant velocity interpolation)
        target_velocity = x_1 - x_0

        # Predict velocity (gunakan step pertama, atau sample random step)
        step_idx = torch.randint(0, self.num_steps, (1,)).item()
        predicted_velocity = self.flow_steps[step_idx](x_t, t)

        # MSE loss
        loss = F.mse_loss(predicted_velocity, target_velocity)

        return loss

    def refine_single_step(
        self,
        hidden_state: torch.Tensor,
        t: float,
        step_idx: int = 0,
    ) -> torch.Tensor:
        """
        Single refinement step untuk inference yang lebih granular.

        Args:
            hidden_state: Current hidden state [batch, seq, d_model]
            t: Current time value (0.0 - 1.0)
            step_idx: Which flow step to use

        Returns:
            Velocity prediction [batch, seq, d_model]
        """
        batch_size = hidden_state.shape[0]
        device = hidden_state.device
        dtype = hidden_state.dtype

        t_tensor = torch.full(
            (batch_size,), t, device=device, dtype=dtype
        )

        step_idx = min(step_idx, self.num_steps - 1)
        velocity = self.flow_steps[step_idx](hidden_state, t_tensor)

        return velocity
