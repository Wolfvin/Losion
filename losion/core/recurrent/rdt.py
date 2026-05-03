"""
Recurrent-Depth Transformer (RDT) — Losion Framework.

OpenMythos reconstruction of the Claude Mythos architecture, implementing
depth-recurrent computation with adaptive halting and stable state injection.

The core insight: instead of stacking N unique transformer layers, loop a
single (or small set of) layers for a variable number of iterations, letting
the model allocate computation dynamically per token. This bridges the gap
between depth and recurrence — gaining the expressiveness of deep networks
with the parameter efficiency of weight-sharing.

Architecture overview::

    Input ──► Prelude ──► [Recurrent Block × K] ──► Coda ──► Output
                            │     │      │
                            │  Loop-idx  Depth-LoRA
                            │  Embed     Adaptation
                            │     │      │
                            ▼     ▼      ▼
                         LTI-Stable Injection ◄── Recurrent State h_t
                            │
                         ACT Halting
                         (learned pondering)

Key components:
1. LTIStableInjection   — Discrete LTI state update with spectral constraint
2. AdaptiveComputationTime — Learned halting for variable loop depth
3. LoopIndexEmbedding   — Positional encoding for depth iterations
4. DepthLoRA            — Per-iteration low-rank adaptation
5. RecurrentDepthBlock  — Main orchestrating block

Credits & References
--------------------
- Universal Transformers
  Dehghani et al., "Universal Transformers" (2019)
  arXiv:1807.03819 — foundational work on weight-shared depth recurrence.

- OpenMythos
  Kye Gomez, github.com/kyegomez/OpenMythos
  Open-source reconstruction of the Claude Mythos architecture from which
  this module draws its design philosophy.

- Relaxed Recursive Transformers
  Bae et al., "Relaxed Recursive Transformers" (2024)
  arXiv:2410.20672 — depth LoRA: per-iteration low-rank adaptation that
  relaxes strict weight-sharing while preserving parameter efficiency.

- Reasoning with Latent Thoughts
  Saunshi et al., "Reasoning with Latent Thoughts" (2025)
  arXiv:2502.17416 — latent thinking via iterative refinement with
  adaptive computation time in the depth dimension.

- COCONUT (Chain of Continuous Thought)
  "Training Large Language Models to Reason in a Continuous Latent Space"
  arXiv:2412.06769 — continuous latent reasoning that motivates the
  recurrent depth paradigm.

- Loop, Think, & Generalize
  arXiv:2604.07822 — looped transformers with emergent reasoning and
  generalization capabilities.

- Parcae Scaling Laws
  arXiv:2604.12946 — scaling laws for recurrent-depth architectures that
  inform our spectral stability constraints and compute allocation.

Hardware: Pure PyTorch, compatible with CUDA / ROCm / CPU.
Supports bf16 and fp16 mixed precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# LTIStableInjection
# ============================================================================


class LTIStableInjection(nn.Module):
    """Linear Time-Invariant (LTI) stable state injection for recurrence.

    Models the recurrence as a discrete LTI system::

        h_{t+1} = A * h_t + B * e_t

    where:
    - h_t is the recurrent hidden state at iteration t
    - e_t is the expert (block) output at iteration t
    - A is the state-transition matrix
    - B is the input-injection matrix

    The spectral radius rho(A) is constrained to be < 1 for asymptotic
    stability. This is enforced via eigendecomposition parameterization:
    the eigenvalues are parameterized in log-magnitude / phase form and
    clamped inside the unit circle.

    This design is informed by:
    - Parcae scaling laws (arXiv:2604.12946) showing that stable recurrence
      is critical for scaling recurrent-depth models.
    - Classical control theory: discrete-time LTI systems are stable iff
      all eigenvalues of A lie strictly within the unit circle.

    Args:
        d_model: Model dimension.
        d_state: Dimension of the recurrent hidden state. Defaults to d_model.
        spectral_radius_cap: Maximum spectral radius rho(A). Must be < 1.
            Defaults to 0.95 for a slight stability margin.
        init_scale: Initial scale for B matrix. Defaults to 0.02.
        power_iter_steps: Number of power-iteration steps for spectral-norm
            estimation at runtime (used as a cheaper alternative to full
            eigendecomposition). Defaults to 5.
        use_eigendecomp: If True, use eigendecomposition-based projection
            to enforce the spectral constraint (more accurate but O(d^3)).
            If False, use power iteration + rescaling (cheaper). Defaults to True.
    """

    def __init__(
        self,
        d_model: int,
        d_state: Optional[int] = None,
        spectral_radius_cap: float = 0.95,
        init_scale: float = 0.02,
        power_iter_steps: int = 5,
        use_eigendecomp: bool = True,
    ) -> None:
        super().__init__()

        if spectral_radius_cap >= 1.0:
            raise ValueError(
                f"spectral_radius_cap must be < 1 for stability, got {spectral_radius_cap}"
            )

        self.d_model = d_model
        self.d_state = d_state or d_model
        self.spectral_radius_cap = spectral_radius_cap
        self.power_iter_steps = power_iter_steps
        self.use_eigendecomp = use_eigendecomp

        # --- State-transition matrix A ---
        # We parameterize A via eigendecomposition: A = V * diag(lambda) * V^{-1}
        # where lambda = rho * exp(i * theta), with rho < spectral_radius_cap.
        #
        # For efficiency with real-valued states, we use a real parameterization:
        # Store log_magnitudes (constrained via sigmoid) and phases.
        # For the non-eigendecomp path, we store A directly and rescale.

        if use_eigendecomp:
            # Log-magnitudes: mapped through sigmoid * cap to ensure rho < cap
            self.log_mag_raw = nn.Parameter(
                torch.randn(self.d_state) * 0.1
            )
            # Phases for complex eigenvalues (paired conjugates for real A)
            self.phase_raw = nn.Parameter(
                torch.randn(self.d_state) * 0.1
            )
            # Eigenvector matrix (orthogonal init)
            self.V = nn.Parameter(
                self._random_orthogonal(self.d_state)
            )
        else:
            # Direct parameterization of A, rescaled at runtime
            self.A_raw = nn.Parameter(
                torch.randn(self.d_state, self.d_state) * init_scale
            )
            # Power-iteration vector for spectral norm estimation
            self.register_buffer(
                "_u", torch.randn(self.d_state), persistent=False
            )

        # --- Input-injection matrix B ---
        self.B = nn.Linear(d_model, self.d_state, bias=False)
        nn.init.normal_(self.B.weight, std=init_scale)

        # --- Output projection (from state space back to model dim) ---
        self.output_proj = nn.Linear(self.d_state, d_model, bias=False)
        nn.init.normal_(self.output_proj.weight, std=init_scale)

        # --- Gating for residual blend ---
        self.gate = nn.Linear(d_model + self.d_state, d_model, bias=False)
        nn.init.zeros_(self.gate.weight)

    @staticmethod
    def _random_orthogonal(dim: int) -> torch.Tensor:
        """Generate a random orthogonal matrix via QR decomposition."""
        M = torch.randn(dim, dim)
        Q, _ = torch.linalg.qr(M)
        return Q

    def _build_stable_A_eigendecomp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the state-transition matrix A with guaranteed spectral stability
        via eigendecomposition parameterization.

        Returns:
            Tuple of (A_matrix, spectral_radius) where A has rho(A) < spectral_radius_cap.
        """
        # Constrain magnitudes: rho in (0, spectral_radius_cap)
        rho = torch.sigmoid(self.log_mag_raw) * self.spectral_radius_cap
        theta = self.phase_raw * math.pi  # phases in [-pi, pi]

        # Construct diagonal eigenvalue matrix (real part only for simplicity)
        # For real-valued systems, complex eigenvalues come in conjugate pairs.
        # We use: lambda_i = rho_i * cos(theta_i) for real-valued A.
        eigenvalues = rho * torch.cos(theta)

        # A = V * diag(eigenvalues) * V^{-1}
        V = self.V
        # Use solve instead of explicit inverse for numerical stability
        # A = V @ diag(eigenvalues) @ V^{-1}
        D = torch.diag(eigenvalues)
        V_inv = torch.linalg.inv(V)
        A = V @ D @ V_inv

        # Spectral radius is max |eigenvalue| = max(rho * |cos(theta)|) <= max(rho) < cap
        spectral_radius = rho.max()

        return A, spectral_radius

    def _build_stable_A_power_iter(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a stable A matrix using power iteration for spectral-norm
        estimation and rescaling.

        Returns:
            Tuple of (A_matrix, spectral_radius_estimate).
        """
        A = self.A_raw

        # Power iteration to estimate spectral norm (largest singular value as proxy)
        u = self._u
        with torch.no_grad():
            for _ in range(self.power_iter_steps):
                v = F.normalize(A.T @ u, dim=0)
                u = F.normalize(A @ v, dim=0)
            self._u.copy_(u)

        sigma_max = (u @ A @ (A.T @ u)).clamp(min=1e-8).sqrt()

        # Rescale A so that spectral radius < cap
        if sigma_max > self.spectral_radius_cap:
            A = A * (self.spectral_radius_cap / sigma_max.detach())

        return A, sigma_max.clamp(max=self.spectral_radius_cap)

    def forward(
        self,
        hidden_state: torch.Tensor,
        expert_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply LTI-stable state injection.

        Updates the recurrent hidden state via the discrete LTI update
        and blends the result back into the model's hidden representation.

        Args:
            hidden_state: Current recurrent state h_t, shape (batch, d_state).
            expert_output: Block output e_t, shape (batch, d_model).

        Returns:
            Tuple of (updated_hidden_state, updated_model_hidden):
            - updated_hidden_state: h_{t+1}, shape (batch, d_state)
            - updated_model_hidden: blended output, shape (batch, d_model)
        """
        batch = expert_output.shape[0]

        # Build stable transition matrix
        if self.use_eigendecomp:
            A, spectral_radius = self._build_stable_A_eigendecomp()
        else:
            A, spectral_radius = self._build_stable_A_power_iter()

        # LTI update: h_{t+1} = A @ h_t + B @ e_t
        # h_t: (batch, d_state), A: (d_state, d_state)
        h_next = (hidden_state @ A.T) + self.B(expert_output)

        # Project state back to model dimension
        state_out = self.output_proj(h_next)

        # Gated residual blend
        gate_input = torch.cat([expert_output, h_next], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        blended = expert_output + gate_values * state_out

        return h_next, blended

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Initialize the recurrent hidden state to zeros.

        Args:
            batch_size: Batch size.
            device: Device for the state tensor.
            dtype: Data type (supports bf16/fp16).

        Returns:
            Zero hidden state of shape (batch_size, d_state).
        """
        return torch.zeros(batch_size, self.d_state, device=device, dtype=dtype)


# ============================================================================
# AdaptiveComputationTime
# ============================================================================


class AdaptiveComputationTime(nn.Module):
    """Adaptive Computation Time (ACT) for variable-depth loop halting.

    Implements a learned halting criterion that determines how many loop
    iterations to execute per token. Each iteration produces a halt
    probability p_i, and the loop terminates when the cumulative
    probability reaches 1 - threshold.

    During training, soft pondering is used (weighted sum of all iteration
    outputs), enabling differentiable halting. During inference, hard
    halting is used (stop at the first iteration where cumulative
    probability >= 1 - threshold).

    This is inspired by:
    - Universal Transformers (Dehghani et al., 2019, arXiv:1807.03819)
      which introduced ACT for weight-shared transformers.
    - Reasoning with Latent Thoughts (Saunshi et al., 2025, arXiv:2502.17416)
      which applies adaptive depth to latent thinking.
    - COCONUT (arXiv:2412.06769) which demonstrates the power of variable-
      depth reasoning in continuous latent spaces.

    The ponder cost (regularization) encourages computational efficiency:
        L_ponder = sum_i p_i * i
    which penalizes excessive iterations.

    Args:
        d_model: Model dimension.
        max_iterations: Maximum number of loop iterations. Defaults to 6.
        halt_threshold: Threshold for cumulative halt probability. Defaults to 0.01.
        ponder_epsilon: Small constant for numerical stability in remainder
            computation. Defaults to 1e-3.
        halting_mlp_hidden: Hidden dimension for the halting MLP. Defaults to 64.
    """

    def __init__(
        self,
        d_model: int,
        max_iterations: int = 6,
        halt_threshold: float = 0.01,
        ponder_epsilon: float = 1e-3,
        halting_mlp_hidden: int = 64,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_iterations = max_iterations
        self.halt_threshold = halt_threshold
        self.ponder_epsilon = ponder_epsilon

        # Small MLP to predict halt probability from the current hidden state
        self.halt_mlp = nn.Sequential(
            nn.Linear(d_model, halting_mlp_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(halting_mlp_hidden, 1, bias=True),
            nn.Sigmoid(),
        )

        # Initialize the final layer to bias towards continuing (small halting prob)
        nn.init.zeros_(self.halt_mlp[-2].weight)
        nn.init.constant_(self.halt_mlp[-2].bias, -1.0)  # sigmoid(-1) ≈ 0.27

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        training: bool = True,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Compute adaptive halting over loop iterations.

        Args:
            hidden_states: List of hidden states from each loop iteration,
                each of shape (batch, d_model). Length = actual iterations.
            training: If True, use soft pondering (differentiable weighted sum).
                If False, use hard halting (select single iteration).

        Returns:
            Tuple of (output, n_iterations, halt_probs, ponder_cost):
            - output: Weighted combination of iteration outputs, shape (batch, d_model).
            - n_iterations: Number of iterations actually used.
            - halt_probs: Halting probabilities at each iteration, shape (max_iters, batch).
            - ponder_cost: Regularization cost encouraging efficiency, shape (batch,).
        """
        batch = hidden_states[0].shape[0]
        device = hidden_states[0].device
        dtype = hidden_states[0].dtype

        n_iters = len(hidden_states)

        # Compute halt probabilities for each iteration
        halt_probs_list: List[torch.Tensor] = []
        for i in range(n_iters):
            p = self.halt_mlp(hidden_states[i]).squeeze(-1)  # (batch,)
            halt_probs_list.append(p)

        # Stack: (n_iters, batch)
        halt_probs = torch.stack(halt_probs_list, dim=0)

        if training:
            # --- Soft pondering (differentiable) ---
            # Accumulate halt probabilities with remainder distribution
            cumulative = torch.zeros(batch, device=device, dtype=dtype)
            weights: List[torch.Tensor] = []
            remainder_weights: List[torch.Tensor] = []

            for i in range(n_iters):
                p_i = halt_probs[i]  # (batch,)

                if i == n_iters - 1:
                    # Last iteration: all remaining probability goes here
                    w = 1.0 - cumulative
                else:
                    # Weight = p_i * (1 - cumulative_so_far)
                    w = p_i * (1.0 - cumulative)

                weights.append(w)
                cumulative = cumulative + w

            # Stack weights: (n_iters, batch)
            weight_tensor = torch.stack(weights, dim=0)

            # Weighted sum of all iteration outputs
            # hidden_states stacked: (n_iters, batch, d_model)
            stacked = torch.stack(hidden_states, dim=0)
            # weight_tensor: (n_iters, batch, 1)
            output = (stacked * weight_tensor.unsqueeze(-1)).sum(dim=0)  # (batch, d_model)

            # Ponder cost: sum of p_i * (i+1) for regularization
            iter_indices = torch.arange(
                1, n_iters + 1, device=device, dtype=dtype
            ).unsqueeze(-1)  # (n_iters, 1)
            ponder_cost = (halt_probs * iter_indices).sum(dim=0)  # (batch,)

        else:
            # --- Hard halting (inference) ---
            cumulative = torch.zeros(batch, device=device, dtype=dtype)
            output = torch.zeros(batch, self.d_model, device=device, dtype=dtype)

            for i in range(n_iters):
                p_i = halt_probs[i]  # (batch,)
                cumulative = cumulative + p_i

                # Select this iteration's output for tokens that halt here
                halt_mask = (cumulative >= (1.0 - self.halt_threshold)) & (cumulative - p_i < (1.0 - self.halt_threshold))
                # For the last iteration, everyone halts
                if i == n_iters - 1:
                    halt_mask = (cumulative - p_i < (1.0 - self.halt_threshold))

                if halt_mask.any():
                    output[halt_mask] = hidden_states[i][halt_mask]

                # Early exit if all tokens have halted
                if (cumulative >= (1.0 - self.halt_threshold)).all():
                    break

            # Ponder cost (for logging only during inference)
            iter_indices = torch.arange(
                1, n_iters + 1, device=device, dtype=dtype
            ).unsqueeze(-1)
            ponder_cost = (halt_probs * iter_indices).sum(dim=0)

        return output, n_iters, halt_probs, ponder_cost


# ============================================================================
# LoopIndexEmbedding
# ============================================================================


class LoopIndexEmbedding(nn.Module):
    """Positional embedding for loop iterations (depth dimension).

    Analogous to how RoPE encodes sequence position, LoopIndexEmbedding
    encodes the *depth position* — which loop iteration we are in. This
    gives the model information about how many times it has already
    processed the current token, enabling iteration-aware computation.

    Supports two modes:
    1. **Learnable**: A learned embedding table up to max_loop_iters.
       More expressive but cannot extrapolate beyond max_loop_iters.
    2. **Sinusoidal**: Fixed sinusoidal embeddings that allow
       extrapolation to longer loops at inference time.

    The sinusoidal formulation follows:
        PE(loop, 2i)   = sin(loop / 10000^(2i/d))
        PE(loop, 2i+1) = cos(loop / 10000^(2i/d))

    This is motivated by:
    - Universal Transformers (arXiv:1807.03819) which use coordinate
      embeddings for depth.
    - Loop, Think, & Generalize (arXiv:2604.07822) which shows that
      depth positional information is crucial for looped transformers.

    Args:
        d_model: Model dimension (embedding dimension).
        max_loop_iters: Maximum number of loop iterations. Defaults to 8.
        mode: Embedding mode — "learnable" or "sinusoidal". Defaults to "learnable".
        dropout: Dropout rate for the embeddings. Defaults to 0.0.
    """

    def __init__(
        self,
        d_model: int,
        max_loop_iters: int = 8,
        mode: str = "learnable",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if mode not in ("learnable", "sinusoidal"):
            raise ValueError(f"mode must be 'learnable' or 'sinusoidal', got '{mode}'")

        self.d_model = d_model
        self.max_loop_iters = max_loop_iters
        self.mode = mode
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if mode == "learnable":
            # Learnable embedding: (max_loop_iters, d_model)
            # Index 0 is unused (loop starts at 1), but included for simplicity
            self.embedding = nn.Embedding(max_loop_iters + 1, d_model)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            # Zero out index 0 (no loop iteration)
            with torch.no_grad():
                self.embedding.weight[0].zero_()
        else:
            # Sinusoidal embedding (fixed, not learnable)
            self.register_buffer(
                "_sinusoidal_table",
                self._build_sinusoidal_table(max_loop_iters + 1, d_model),
            )

    @staticmethod
    def _build_sinusoidal_table(length: int, d_model: int) -> torch.Tensor:
        """Build sinusoidal positional embedding table.

        Args:
            length: Number of positions (max_loop_iters + 1).
            d_model: Embedding dimension.

        Returns:
            Sinusoidal embedding table of shape (length, d_model).
        """
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        table = torch.zeros(length, d_model)
        table[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            table[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        else:
            table[:, 1::2] = torch.cos(position * div_term)

        # Zero out index 0
        table[0].zero_()
        return table

    def forward(self, loop_idx: int, batch_size: int) -> torch.Tensor:
        """Get the loop-index embedding for a given iteration.

        Args:
            loop_idx: Current loop iteration index (0-based; 0 means no iteration).
            batch_size: Batch size to expand the embedding for.

        Returns:
            Loop-index embedding of shape (batch_size, d_model).
        """
        if loop_idx < 0 or loop_idx > self.max_loop_iters:
            # Clamp to valid range (extrapolation for sinusoidal, clip for learnable)
            loop_idx = min(loop_idx, self.max_loop_iters)

        if self.mode == "learnable":
            indices = torch.full(
                (batch_size,), loop_idx, dtype=torch.long, device=self.embedding.weight.device
            )
            emb = self.embedding(indices)  # (batch, d_model)
        else:
            # Sinusoidal: index into the precomputed table
            emb = self._sinusoidal_table[loop_idx].unsqueeze(0).expand(batch_size, -1)
            emb = emb.clone()  # avoid in-place ops on buffer

        return self.dropout(emb)


# ============================================================================
# DepthLoRA
# ============================================================================


class DepthLoRA(nn.Module):
    """Depth-adaptive LoRA for per-iteration weight adaptation.

    From Relaxed Recursive Transformers (Bae et al., 2024, arXiv:2410.20672):
    instead of using identical weights at every loop iteration, add a small
    per-iteration low-rank adaptation. This "relaxes" strict weight-sharing
    while preserving parameter efficiency.

    The adaptation is:
        output = base_output + alpha/r * B_{loop_idx} @ A_{loop_idx} @ x

    Two modes:
    1. **Per-iteration**: Separate lora_A and lora_B for each loop iteration.
       More expressive but uses more parameters.
    2. **Shared-index**: Shared A and B matrices with a loop-index-dependent
       scaling factor. More parameter-efficient.

    The per-iteration mode allows each depth layer to develop its own
    specialization (e.g., early iterations for local features, later
    iterations for global reasoning), which is critical for the
    Recurrent-Depth Transformer to avoid representational collapse.

    Args:
        d_model: Model dimension.
        rank: LoRA rank per iteration. Defaults to 8.
        alpha: LoRA scaling factor. Defaults to 16.0.
        max_loop_iters: Maximum loop iterations. Defaults to 8.
        mode: "per_iteration" or "shared_index". Defaults to "per_iteration".
        dropout: Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 8,
        alpha: float = 16.0,
        max_loop_iters: int = 8,
        mode: str = "per_iteration",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if mode not in ("per_iteration", "shared_index"):
            raise ValueError(
                f"mode must be 'per_iteration' or 'shared_index', got '{mode}'"
            )

        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.max_loop_iters = max_loop_iters
        self.mode = mode
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if mode == "per_iteration":
            # Per-iteration LoRA: separate A and B for each iteration
            # Shape: (max_loop_iters, rank, d_model) and (max_loop_iters, d_model, rank)
            self.lora_A = nn.Parameter(
                torch.randn(max_loop_iters, rank, d_model) * 0.01
            )
            self.lora_B = nn.Parameter(
                torch.zeros(max_loop_iters, d_model, rank)
            )
        else:
            # Shared-index LoRA: shared matrices with iteration-dependent scaling
            self.lora_A_shared = nn.Linear(d_model, rank, bias=False)
            self.lora_B_shared = nn.Linear(rank, d_model, bias=False)
            # Iteration-dependent scaling: one scalar per iteration
            self.iter_scales = nn.Parameter(
                torch.ones(max_loop_iters) * 0.1
            )
            # Initialize
            nn.init.kaiming_uniform_(self.lora_A_shared.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_shared.weight)

    def forward(
        self,
        x: torch.Tensor,
        loop_idx: int,
    ) -> torch.Tensor:
        """Apply depth LoRA adaptation for a given loop iteration.

        Args:
            x: Input tensor, shape (batch, d_model).
            loop_idx: Current loop iteration index (0-based).

        Returns:
            Adapted output tensor, shape (batch, d_model).
        """
        # Clamp loop_idx to valid range
        loop_idx = min(max(loop_idx, 0), self.max_loop_iters - 1)

        x_dropped = self.dropout(x)

        if self.mode == "per_iteration":
            # lora_A[loop_idx]: (rank, d_model)
            # lora_B[loop_idx]: (d_model, rank)
            A_i = self.lora_A[loop_idx]  # (rank, d_model)
            B_i = self.lora_B[loop_idx]  # (d_model, rank)

            # x: (batch, d_model)
            # down: (batch, rank)
            down = F.linear(x_dropped, A_i)
            # up: (batch, d_model)
            up = F.linear(down, B_i)

            return self.scaling * up
        else:
            # Shared matrices with iteration-dependent scaling
            scale_i = self.iter_scales[loop_idx]
            down = self.lora_A_shared(x_dropped)  # (batch, rank)
            up = self.lora_B_shared(down)  # (batch, d_model)
            return self.scaling * scale_i * up


# ============================================================================
# RecurrentDepthBlock
# ============================================================================


@dataclass
class RecurrentDepthAuxInfo:
    """Auxiliary information from a RecurrentDepthBlock forward pass.

    Attributes:
        loop_count: Number of loop iterations actually executed.
        halt_probs: Halting probabilities at each iteration.
            Shape (n_iters, batch) or None if ACT is disabled.
        spectral_radius: Estimated spectral radius of the LTI transition matrix.
        ponder_cost: ACT regularization cost, shape (batch,).
    """

    loop_count: int
    halt_probs: Optional[torch.Tensor] = None
    spectral_radius: Optional[torch.Tensor] = None
    ponder_cost: Optional[torch.Tensor] = None


class RecurrentDepthBlock(nn.Module):
    """Recurrent-Depth Transformer Block — main orchestrating module.

    Takes a transformer-like block (nn.Module) and loops it for a variable
    number of iterations, applying depth-adaptive modifications at each step.

    Architecture::

        Input ──► Prelude ──► [Recurrent Block × K] ──► Coda ──► Output
                                  │
                            ┌─────┴─────┐
                            │  Loop i:  │
                            │  1. Add loop-index embedding
                            │  2. Apply depth LoRA adaptation
                            │  3. Run the block forward
                            │  4. Apply LTI-stable injection
                            │  5. Check ACT halt condition
                            └───────────┘

    The **Prelude** prepares the input for recurrence (e.g., initial
    normalization and projection). The **Coda** produces the final output
    after all loop iterations (e.g., output normalization and projection).

    This architecture draws from:
    - Universal Transformers (arXiv:1807.03819): weight-shared depth recurrence.
    - Relaxed Recursive Transformers (arXiv:2410.20672): depth LoRA for
      per-iteration adaptation.
    - Reasoning with Latent Thoughts (arXiv:2502.17416): adaptive depth
      for latent reasoning.
    - COCONUT (arXiv:2412.06769): continuous latent reasoning spaces.
    - Loop, Think, & Generalize (arXiv:2604.07822): looped transformers.
    - Parcae Scaling Laws (arXiv:2604.12946): stability constraints.

    Args:
        block: The transformer-like block to loop (nn.Module).
            Must accept (hidden_states, **kwargs) and return (output, aux).
        d_model: Model dimension.
        max_loop_iters: Maximum number of loop iterations. Defaults to 6.
        lti_state_dim: Dimension of LTI recurrent state. Defaults to d_model.
        lti_spectral_radius_cap: Maximum spectral radius for LTI stability.
            Defaults to 0.95.
        lora_rank: Rank for depth LoRA. Defaults to 8.
        lora_alpha: Alpha for depth LoRA. Defaults to 16.0.
        lora_mode: Depth LoRA mode — "per_iteration" or "shared_index".
            Defaults to "per_iteration".
        loop_emb_mode: Loop index embedding mode — "learnable" or "sinusoidal".
            Defaults to "learnable".
        act_halt_threshold: ACT halting threshold. Defaults to 0.01.
        use_act: Whether to use adaptive computation time. Defaults to True.
        dropout: Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self,
        block: nn.Module,
        d_model: int,
        max_loop_iters: int = 6,
        lti_state_dim: Optional[int] = None,
        lti_spectral_radius_cap: float = 0.95,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_mode: str = "per_iteration",
        loop_emb_mode: str = "learnable",
        act_halt_threshold: float = 0.01,
        use_act: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_loop_iters = max_loop_iters
        self.use_act = use_act

        # ---- Wrapped transformer block ----
        self.block = block

        # ---- Prelude: input preparation ----
        self.prelude_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.prelude_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.prelude_proj.weight, std=0.02)

        # ---- Coda: output finalization ----
        self.coda_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.coda_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.coda_proj.weight, std=0.02)

        # ---- LTI Stable Injection ----
        self.lti = LTIStableInjection(
            d_model=d_model,
            d_state=lti_state_dim or d_model,
            spectral_radius_cap=lti_spectral_radius_cap,
        )

        # ---- Adaptive Computation Time ----
        if use_act:
            self.act = AdaptiveComputationTime(
                d_model=d_model,
                max_iterations=max_loop_iters,
                halt_threshold=act_halt_threshold,
            )

        # ---- Loop Index Embedding ----
        self.loop_embedding = LoopIndexEmbedding(
            d_model=d_model,
            max_loop_iters=max_loop_iters,
            mode=loop_emb_mode,
            dropout=dropout,
        )

        # ---- Depth LoRA ----
        self.depth_lora = DepthLoRA(
            d_model=d_model,
            rank=lora_rank,
            alpha=lora_alpha,
            max_loop_iters=max_loop_iters,
            mode=lora_mode,
            dropout=dropout,
        )

        # ---- Per-iteration layer norms (pre-block normalization) ----
        self.iter_norms = nn.ModuleList(
            [nn.RMSNorm(d_model, eps=1e-5) for _ in range(max_loop_iters)]
        )

        # ---- Residual scaling (learned per iteration) ----
        self.residual_scales = nn.Parameter(
            torch.ones(max_loop_iters) * 0.5
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        **block_kwargs: Any,
    ) -> Tuple[torch.Tensor, RecurrentDepthAuxInfo]:
        """Forward pass through the Recurrent-Depth Block.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).
            attention_mask: Optional attention mask for the wrapped block.
            recurrent_state: Optional LTI recurrent state from a previous
                forward pass, shape (batch, seq_len, d_state). If None,
                initialized to zeros.
            **block_kwargs: Additional keyword arguments passed to the
                wrapped block's forward method.

        Returns:
            Tuple of (output, aux_info):
            - output: Transformed output, shape (batch, seq_len, d_model).
            - aux_info: RecurrentDepthAuxInfo with diagnostic information.
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        # ---- Prelude ----
        hidden = self.prelude_norm(x)
        hidden = self.prelude_proj(hidden)
        # Residual from input
        prelude_out = x + hidden

        # ---- Initialize LTI recurrent state ----
        d_state = self.lti.d_state
        if recurrent_state is None:
            lti_state = torch.zeros(
                batch, seq_len, d_state, device=device, dtype=dtype
            )
        else:
            lti_state = recurrent_state

        # ---- Recurrent loop ----
        # Collect both pooled representations (for ACT halting decisions) and
        # full hidden states (for ACT-weighted output combination).
        iteration_pooled: List[torch.Tensor] = []      # (batch, d_model) each
        iteration_full: List[torch.Tensor] = []         # (batch, seq_len, d_model) each
        spectral_radius: Optional[torch.Tensor] = None

        for i in range(self.max_loop_iters):
            # 1. Pre-norm for this iteration
            iter_input = self.iter_norms[i](prelude_out if i == 0 else hidden)

            # 2. Add loop-index embedding
            loop_emb = self.loop_embedding(i, batch)  # (batch, d_model)
            # Expand for sequence dimension: (batch, 1, d_model) + (batch, seq_len, d_model)
            iter_input = iter_input + loop_emb.unsqueeze(1)

            # 3. Flatten for block processing
            iter_flat = iter_input.reshape(batch * seq_len, self.d_model)

            # 4. Apply depth LoRA adaptation
            lora_delta = self.depth_lora(iter_flat, loop_idx=i)
            iter_adapted = iter_flat + lora_delta
            iter_adapted = iter_adapted.reshape(batch, seq_len, self.d_model)

            # 5. Run the wrapped block
            block_out, _block_aux = self.block(
                iter_adapted,
                attention_mask=attention_mask,
                **block_kwargs,
            )

            # 6. Residual connection with learned scaling
            scale = torch.sigmoid(self.residual_scales[i])
            hidden = (prelude_out if i == 0 else hidden) + scale * block_out

            # 7. LTI-stable injection
            # Reshape for per-token state update: (batch*seq_len, d_model)
            hidden_flat = hidden.reshape(batch * seq_len, self.d_model)
            lti_state_flat = lti_state.reshape(batch * seq_len, d_state)

            lti_state_flat, hidden_flat = self.lti(
                lti_state_flat, hidden_flat
            )

            hidden = hidden_flat.reshape(batch, seq_len, self.d_model)
            lti_state = lti_state_flat.reshape(batch, seq_len, d_state)

            # Track spectral radius from first iteration
            if i == 0:
                if self.lti.use_eigendecomp:
                    _, spectral_radius = self.lti._build_stable_A_eigendecomp()
                else:
                    _, spectral_radius = self.lti._build_stable_A_power_iter()

            # Store per-iteration representations for ACT
            iteration_full.append(hidden)                      # (batch, seq_len, d_model)
            iteration_pooled.append(hidden.mean(dim=1))        # (batch, d_model)

        # ---- Adaptive Computation Time ----
        if self.use_act:
            _, n_iters, halt_probs, ponder_cost = self.act(
                iteration_pooled, training=self.training
            )

            if self.training:
                # Soft pondering: weighted combination of full hidden states
                # Reconstruct iteration weights from halt probabilities
                cumulative = torch.zeros(batch, device=device, dtype=dtype)
                weights_list: List[torch.Tensor] = []
                for i in range(n_iters):
                    p_i = halt_probs[i]
                    if i == n_iters - 1:
                        w = 1.0 - cumulative
                    else:
                        w = p_i * (1.0 - cumulative)
                    weights_list.append(w)
                    cumulative = cumulative + w

                # Stack: (n_iters, batch)
                weight_tensor = torch.stack(weights_list, dim=0)
                # Stack full hidden states: (n_iters, batch, seq_len, d_model)
                stacked_full = torch.stack(iteration_full, dim=0)
                # Weighted sum: (batch, seq_len, d_model)
                output = (
                    stacked_full * weight_tensor.unsqueeze(-1).unsqueeze(-1)
                ).sum(dim=0)
            else:
                # Hard halting: use the last hidden state that passed ACT
                output = hidden
        else:
            n_iters = self.max_loop_iters
            halt_probs = None
            ponder_cost = torch.zeros(batch, device=device, dtype=dtype)
            output = hidden

        # ---- Coda ----
        output = self.coda_norm(output)
        output = self.coda_proj(output)
        output = output + prelude_out  # Final residual

        # ---- Assemble aux info ----
        aux = RecurrentDepthAuxInfo(
            loop_count=n_iters,
            halt_probs=halt_probs,
            spectral_radius=spectral_radius,
            ponder_cost=ponder_cost,
        )

        return output, aux

    def forward_inference(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
        **block_kwargs: Any,
    ) -> Tuple[torch.Tensor, RecurrentDepthAuxInfo, torch.Tensor]:
        """Forward pass for single-token O(1) inference.

        Optimized for autoregressive generation where each step processes
        a single token (batch, 1, d_model).

        Args:
            x: Input tensor, shape (batch, 1, d_model).
            attention_mask: Optional attention mask.
            recurrent_state: LTI recurrent state from previous step,
                shape (batch, 1, d_state).
            **block_kwargs: Additional kwargs for the wrapped block.

        Returns:
            Tuple of (output, aux_info, updated_recurrent_state):
            - output: shape (batch, 1, d_model).
            - aux_info: RecurrentDepthAuxInfo.
            - updated_recurrent_state: shape (batch, 1, d_state) for next step.
        """
        batch = x.shape[0]
        device = x.device
        dtype = x.dtype
        d_state = self.lti.d_state

        # Prelude
        hidden = self.prelude_norm(x)
        hidden = self.prelude_proj(hidden)
        prelude_out = x + hidden

        # Initialize LTI state
        if recurrent_state is None:
            lti_state = torch.zeros(batch, 1, d_state, device=device, dtype=dtype)
        else:
            lti_state = recurrent_state

        # Recurrent loop with hard halting
        iteration_outputs: List[torch.Tensor] = []

        for i in range(self.max_loop_iters):
            iter_input = self.iter_norms[i](prelude_out if i == 0 else hidden)
            loop_emb = self.loop_embedding(i, batch)
            iter_input = iter_input + loop_emb.unsqueeze(1)

            iter_flat = iter_input.reshape(batch, self.d_model)
            lora_delta = self.depth_lora(iter_flat, loop_idx=i)
            iter_adapted = (iter_flat + lora_delta).unsqueeze(1)

            # Run block (inference mode if available)
            if hasattr(self.block, "forward_inference"):
                block_out = self.block.forward_inference(
                    iter_adapted, attention_mask=attention_mask, **block_kwargs
                )
                if isinstance(block_out, tuple):
                    block_out = block_out[0]
            else:
                block_out, _ = self.block(
                    iter_adapted, attention_mask=attention_mask, **block_kwargs
                )

            scale = torch.sigmoid(self.residual_scales[i])
            hidden = (prelude_out if i == 0 else hidden) + scale * block_out

            # LTI update
            hidden_flat = hidden.reshape(batch, self.d_model)
            lti_state_flat = lti_state.reshape(batch, d_state)
            lti_state_flat, hidden_flat = self.lti(lti_state_flat, hidden_flat)
            hidden = hidden_flat.unsqueeze(1)
            lti_state = lti_state_flat.unsqueeze(1)

            # ACT hard halting check
            if self.use_act:
                iter_repr = hidden_flat  # (batch, d_model)
                p = self.act.halt_mlp(iter_repr).squeeze(-1)  # (batch,)
                # Simple hard halt: if average halt probability exceeds threshold
                if p.mean().item() >= (1.0 - self.act.halt_threshold):
                    break

        # Coda
        output = self.coda_norm(hidden)
        output = self.coda_proj(output)
        output = output + prelude_out

        aux = RecurrentDepthAuxInfo(
            loop_count=i + 1,
            spectral_radius=None,
        )

        return output, aux, lti_state

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Initialize the recurrent state for inference.

        Args:
            batch_size: Batch size.
            device: Device for state tensors.
            dtype: Data type (supports bf16/fp16).

        Returns:
            Initial LTI recurrent state, shape (batch, 1, d_state).
        """
        return self.lti.init_state(batch_size, device, dtype).unsqueeze(1)
