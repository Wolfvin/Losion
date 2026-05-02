"""
LLM-JEPA — Joint-Embedding Predictive Architecture for LLMs
============================================================

Implements the LLM-JEPA training objective which predicts future latent
states instead of next tokens, providing a principled auxiliary training
signal for hybrid SSM-Attention-MoE models like Losion.

Key insight: SSM components naturally predict future states via their
recurrent hidden dynamics, making JEPA a natural fit for the Tri-Jalur
architecture. The LatentPredictor learns to forecast latent representations
H steps ahead, while a TargetEncoder (EMA teacher) provides stable targets
without gradient flow.

Components:
1. JEPAConfig          — Configuration dataclass for all JEPA hyperparameters
2. LatentPredictor     — Predicts future latent states from current hidden states
3. TargetEncoder       — EMA-updated teacher encoder for target representations
4. VICRegLoss          — Variance-Invariance-Covariance regularization loss
5. LLMJEPA             — Main module wrapping a Losion model with JEPA training

References:
- LLM-JEPA (2025, arXiv, 19 citations)
- LeCun, "A Path Towards Autonomous Machine Intelligence" (JEPA, 2022)
- Assran et al., "Self-Supervised Learning from Images with I-JEPA" (2023)
- Bardes et al., "Revisiting Feature Prediction for Self-Supervised Learning"
  (V-JEPA, 2024)

Hardware: Pure PyTorch, compatible with CUDA, ROCm, and CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losion.config import LosionConfig
from losion.models.losion_decoder import LosionForCausalLM, LosionCausalLMOutput

logger = logging.getLogger(__name__)


# ============================================================================
# JEPAConfig
# ============================================================================


@dataclass
class JEPAConfig:
    """Configuration for LLM-JEPA training objective.

    Controls the prediction horizon, latent dimensionality, loss type,
    and EMA teacher update schedule for the JEPA auxiliary training
    signal.

    Attributes:
        d_model: Model hidden dimension (must match LosionConfig.d_model).
        prediction_horizon: Number of future latent states to predict (H).
            Higher values encourage the model to learn more about long-range
            structure but increase prediction difficulty.
        latent_dim: Dimension of predicted latent states. Acts as a
            bottleneck that prevents trivial identity solutions.
        predictor_depth: Number of transformer layers in the LatentPredictor.
        loss_type: JEPA loss variant — "vicreg" (default, prevents collapse),
            "cosine" (simple directional alignment), or "mse" (direct regression).
        vicreg_cov_weight: Weight for the covariance term in VICReg loss.
            Encourages decorrelation across latent dimensions.
        vicreg_var_weight: Weight for the variance term in VICReg loss.
            Prevents representation collapse by enforcing unit variance.
        vicreg_inv_weight: Weight for the invariance term in VICReg loss.
            Encourages predicted and target latents to be similar.
        teacher_ema_decay: Exponential moving average decay for the target
            encoder. Higher values = slower teacher updates = more stable
            targets. Typical range: 0.99–0.9996.
        prediction_weight: Weight for JEPA loss relative to the standard
            LM loss. Total loss = LM_loss + prediction_weight * JEPA_loss.
    """

    d_model: int = 768
    prediction_horizon: int = 4
    latent_dim: int = 256
    predictor_depth: int = 3
    loss_type: str = "vicreg"
    vicreg_cov_weight: float = 1.0
    vicreg_var_weight: float = 1.0
    vicreg_inv_weight: float = 1.0
    teacher_ema_decay: float = 0.996
    prediction_weight: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.prediction_horizon < 1:
            raise ValueError(
                f"prediction_horizon must be >= 1, got {self.prediction_horizon}"
            )
        if self.latent_dim < 1:
            raise ValueError(
                f"latent_dim must be >= 1, got {self.latent_dim}"
            )
        if self.predictor_depth < 1:
            raise ValueError(
                f"predictor_depth must be >= 1, got {self.predictor_depth}"
            )
        if self.loss_type not in ("vicreg", "cosine", "mse"):
            raise ValueError(
                f"loss_type must be 'vicreg', 'cosine', or 'mse', "
                f"got '{self.loss_type}'"
            )
        if not 0.0 <= self.teacher_ema_decay <= 1.0:
            raise ValueError(
                f"teacher_ema_decay must be in [0, 1], got {self.teacher_ema_decay}"
            )
        if self.prediction_weight < 0.0:
            raise ValueError(
                f"prediction_weight must be >= 0, got {self.prediction_weight}"
            )


# ============================================================================
# LatentPredictor
# ============================================================================


class LatentPredictor(nn.Module):
    """Predicts future latent states from current latent representations.

    Uses a small transformer-based architecture that takes the current
    latent representation (output of the online encoder) at each position
    and predicts the next H latent states. The prediction is made
    autoregressively within the predictor: each predicted latent step
    conditions on the previous prediction.

    Architecture:
        1. Project latent input to predictor working dimension
        2. Learnable horizon tokens as initial queries
        3. predictor_depth transformer layers with causal masking
        4. Project each horizon token to latent_dim

    Args:
        config: JEPAConfig with prediction parameters.
    """

    def __init__(self, config: JEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.prediction_horizon = config.prediction_horizon
        self.predictor_depth = config.predictor_depth

        # Working dimension for the predictor transformer
        self.d_pred = config.d_model

        # Project from latent_dim to predictor working dimension
        self.input_proj = nn.Linear(config.latent_dim, self.d_pred, bias=False)

        # Learnable queries for each prediction horizon step
        # These act as "slots" that get filled with predicted latents
        self.horizon_queries = nn.Parameter(
            torch.randn(1, config.prediction_horizon, self.d_pred) * 0.02
        )

        # Small transformer predictor
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=self.d_pred,
            nhead=max(1, self.d_pred // 64),
            dim_feedforward=self.d_pred * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(
            predictor_layer,
            num_layers=config.predictor_depth,
            enable_nested_tensor=False,
        )

        # Project from predictor dimension back to latent space
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_pred, self.d_pred, bias=False),
            nn.GELU(),
            nn.Linear(self.d_pred, config.latent_dim, bias=False),
        )

        # Layer norm for stable predictions
        self.norm = nn.LayerNorm(self.d_pred)

    def forward(
        self, current_latent: torch.Tensor
    ) -> torch.Tensor:
        """Predict future latent states from current latent representations.

        Args:
            current_latent: Latent representations from the online encoder,
                shape (batch, seq_len, latent_dim).

        Returns:
            Predicted future latent states,
            shape (batch, seq_len, prediction_horizon, latent_dim).
        """
        batch, seq_len, _ = current_latent.shape

        # Project latent input to predictor working dimension
        projected = self.input_proj(current_latent)  # (B, S, d_pred)

        # Expand horizon queries for batch and sequence
        # Each position gets its own set of horizon queries
        queries = self.horizon_queries.expand(batch, -1, -1)  # (B, H, d_pred)
        queries = queries.unsqueeze(1).expand(batch, seq_len, -1, -1)  # (B, S, H, d_pred)
        queries = queries.reshape(batch * seq_len, self.prediction_horizon, self.d_pred)

        # Expand projected latent to match
        context = projected.reshape(batch * seq_len, 1, self.d_pred)
        # Concatenate context with queries: [context_token, horizon_queries]
        predictor_input = torch.cat([context, queries], dim=1)  # (B*S, 1+H, d_pred)

        # Causal mask: context can attend to itself, each query can
        # attend to context and previous queries
        seq_len_pred = 1 + self.prediction_horizon
        causal_mask = torch.triu(
            torch.ones(
                seq_len_pred, seq_len_pred,
                device=predictor_input.device, dtype=torch.bool,
            ),
            diagonal=1,
        )

        # Run through transformer predictor
        predicted = self.predictor(predictor_input, mask=causal_mask)
        predicted = self.norm(predicted)

        # Extract only the horizon predictions (skip context token)
        predicted_latents = predicted[:, 1:, :]  # (B*S, H, d_pred)

        # Project to latent space
        predicted_latents = self.output_proj(predicted_latents)

        # Reshape to (B, S, H, latent_dim)
        predicted_latents = predicted_latents.view(
            batch, seq_len, self.prediction_horizon, self.latent_dim
        )

        return predicted_latents


# ============================================================================
# TargetEncoder
# ============================================================================


class TargetEncoder(nn.Module):
    """EMA-updated teacher encoder that provides target latent representations.

    Maintains a copy of a projection head that maps hidden states to the
    latent space. The weights are updated via exponential moving average
    (EMA) from the online encoder, providing stable targets without
    gradient flow.

    The target encoder serves the same function as the "teacher" in
    BYOL/DINO/EMA-based self-supervised methods: it provides slowly-
    evolving targets that prevent the representation from collapsing
    to a trivial solution.

    Args:
        d_model: Model hidden dimension.
        latent_dim: Dimension of the target latent space.
    """

    def __init__(self, d_model: int, latent_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Simple projection head (matches the online encoder structure)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, latent_dim, bias=False),
        )

        # Initialize to small values
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to target latent representations.

        Runs in no_grad context — the target encoder should never
        receive gradient flow. Targets are detached for safety.

        Args:
            x: Hidden states from the main model,
                shape (batch, seq_len, d_model).

        Returns:
            Target latent representations, detached,
            shape (batch, seq_len, latent_dim).
        """
        with torch.no_grad():
            target_latents = self.encoder(x)
            # Ensure no gradient leaks
            target_latents = target_latents.detach()
        return target_latents

    def update_from_encoder(
        self,
        online_encoder: nn.Module,
        ema_decay: float,
    ) -> None:
        """Update target encoder weights via EMA from online encoder.

        Performs: θ_target ← ema_decay * θ_target + (1 - ema_decay) * θ_online

        This method should be called after each training step to keep
        the teacher encoder synchronized with the evolving online encoder.

        The target encoder's parameters are nested under ``encoder.*``
        (due to ``nn.Sequential`` wrapping), so the ``encoder.`` prefix
        is stripped when matching against the online encoder's flat
        parameter names.

        Args:
            online_encoder: The online encoder module whose weights
                are used as the source for the EMA update.
            ema_decay: EMA decay rate (higher = slower update).
        """
        with torch.no_grad():
            online_params = dict(online_encoder.named_parameters())
            for name, param in self.named_parameters():
                # Strip the 'encoder.' prefix to match online param names
                online_name = name.removeprefix("encoder.")
                if online_name in online_params:
                    online_param = online_params[online_name]
                    param.data.mul_(ema_decay).add_(
                        online_param.data, alpha=1.0 - ema_decay
                    )


# ============================================================================
# VICRegLoss
# ============================================================================


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance regularization loss.

    Prevents representation collapse in JEPA training by combining
    three objectives:

    1. **Variance**: Maintains unit variance in each dimension of
       the predicted and target representations, preventing collapse
       to a constant vector.
         L_var = (1/d) * Σ max(0, γ - std(z_i))

    2. **Invariance**: Encourages predicted and target representations
       to be similar (mean squared error between them).
         L_inv = (1/d) * Σ (z_pred_i - z_target_i)²

    3. **Covariance**: Decorrelates the dimensions of each representation
       by minimizing off-diagonal elements of the covariance matrix.
         L_cov = (1/d) * Σ_{i≠j} cov(z)_ij²

    Total: L = λ_inv * L_inv + λ_var * L_var + λ_cov * L_cov

    References:
    - Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
      for Self-Supervised Learning" (ICLR 2022)

    Args:
        var_weight: Weight for the variance term (λ_var).
        inv_weight: Weight for the invariance term (λ_inv).
        cov_weight: Weight for the covariance term (λ_cov).
        variance_target: Target standard deviation for variance term (γ).
            Defaults to 1.0.
        epsilon: Small constant for numerical stability in std computation.
    """

    def __init__(
        self,
        var_weight: float = 1.0,
        inv_weight: float = 1.0,
        cov_weight: float = 1.0,
        variance_target: float = 1.0,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        self.var_weight = var_weight
        self.inv_weight = inv_weight
        self.cov_weight = cov_weight
        self.variance_target = variance_target
        self.epsilon = epsilon

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute variance regularization loss for one representation.

        Encourages each dimension to have standard deviation ≥ variance_target.

        Args:
            z: Representation tensor, (batch, dim) or (batch, seq, dim).

        Returns:
            Scalar variance loss.
        """
        # Compute std per dimension
        std_z = torch.sqrt(z.var(dim=0) + self.epsilon)
        # Hinge loss: penalize dimensions below target
        loss = torch.mean(F.relu(self.variance_target - std_z))
        return loss

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute covariance regularization loss for one representation.

        Penalizes off-diagonal elements of the covariance matrix to
        decorrelate representation dimensions.

        Args:
            z: Representation tensor, (batch, dim) or (batch, seq, dim).

        Returns:
            Scalar covariance loss.
        """
        # Center the representation
        z_centered = z - z.mean(dim=0)

        # Compute covariance matrix
        n = z.shape[0]
        cov = (z_centered.T @ z_centered) / (n - 1)

        # Sum of squared off-diagonal elements
        diag_mask = ~torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
        off_diag_sq = (cov ** 2) * diag_mask.float()
        loss = off_diag_sq.sum() / cov.shape[0]
        return loss

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute VICReg loss between predicted and target representations.

        Both tensors should have the same shape. The loss is computed
        across all dimensions, treating the first axis as the batch
        dimension (after flattening any spatial/temporal dims).

        Args:
            predicted: Predicted latent representations,
                shape (batch, seq_len, prediction_horizon, latent_dim) or
                any shape where the last dim is the feature dimension.
            target: Target latent representations (same shape as predicted).

        Returns:
            Tuple (total_loss, loss_components):
                - total_loss: Weighted sum of variance, invariance, and
                  covariance losses.
                - loss_components: Dictionary with individual loss values
                  for logging.
        """
        # Flatten batch and sequence dimensions for loss computation
        orig_shape = predicted.shape
        pred_flat = predicted.reshape(-1, orig_shape[-1])  # (N, latent_dim)
        targ_flat = target.reshape(-1, orig_shape[-1])      # (N, latent_dim)

        # --- Invariance loss: MSE between predicted and target ---
        inv_loss = F.mse_loss(pred_flat, targ_flat)

        # --- Variance loss: prevent collapse in both representations ---
        var_loss_pred = self._variance_loss(pred_flat)
        var_loss_targ = self._variance_loss(targ_flat)
        var_loss = var_loss_pred + var_loss_targ

        # --- Covariance loss: decorrelate both representations ---
        cov_loss_pred = self._covariance_loss(pred_flat)
        cov_loss_targ = self._covariance_loss(targ_flat)
        cov_loss = cov_loss_pred + cov_loss_targ

        # --- Weighted total ---
        total_loss = (
            self.inv_weight * inv_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
        )

        components = {
            "vicreg_total": total_loss.item(),
            "vicreg_invariance": inv_loss.item(),
            "vicreg_variance": var_loss.item(),
            "vicreg_covariance": cov_loss.item(),
        }

        return total_loss, components


# ============================================================================
# LLMJEPA
# ============================================================================


class LLMJEPA(nn.Module):
    """Main module wrapping a Losion model with JEPA training.

    Combines standard autoregressive language modeling with the
    Joint-Embedding Predictive Architecture (JEPA) objective. During
    training, the model simultaneously:

    1. Processes input through the Losion model for standard LM loss
    2. Uses LatentPredictor to predict future latent states from
       current hidden states
    3. Uses TargetEncoder (EMA teacher) to provide stable target
       latent representations
    4. Computes JEPA loss between predicted and target latents
    5. Combines: total_loss = LM_loss + prediction_weight * JEPA_loss

    The JEPA loss encourages the model's hidden states to contain
    predictive information about future states, which is especially
    beneficial for the SSM pathway that naturally operates on
    state transitions.

    Compatible with LosionTrainer's training loop — forward returns
    (logits, total_loss, loss_dict) that can be directly consumed
    by the trainer.

    Args:
        losion_config: LosionConfig for the base model.
        jepa_config: JEPAConfig for the JEPA training objective.
    """

    def __init__(
        self,
        losion_config: LosionConfig,
        jepa_config: Optional[JEPAConfig] = None,
    ) -> None:
        super().__init__()

        # Ensure d_model consistency
        self.jepa_config = jepa_config or JEPAConfig()
        if self.jepa_config.d_model != losion_config.d_model:
            logger.warning(
                f"JEPAConfig.d_model ({self.jepa_config.d_model}) does not match "
                f"LosionConfig.d_model ({losion_config.d_model}). "
                f"Overriding JEPAConfig.d_model to {losion_config.d_model}."
            )
            self.jepa_config.d_model = losion_config.d_model

        self.losion_config = losion_config

        # ---- Main Losion model ----
        self.model = LosionForCausalLM(losion_config)

        # ---- LatentPredictor (online) ----
        self.latent_predictor = LatentPredictor(self.jepa_config)

        # ---- Online encoder (projects hidden states to latent space) ----
        # This is the "student" encoder that receives gradients
        self.online_encoder = nn.Sequential(
            nn.Linear(self.jepa_config.d_model, self.jepa_config.d_model, bias=False),
            nn.GELU(),
            nn.Linear(self.jepa_config.d_model, self.jepa_config.latent_dim, bias=False),
        )

        # ---- Target encoder (EMA teacher, no gradients) ----
        self.target_encoder = TargetEncoder(
            d_model=self.jepa_config.d_model,
            latent_dim=self.jepa_config.latent_dim,
        )

        # Initialize target encoder as a copy of the online encoder
        self._init_target_encoder()

        # ---- JEPA loss function ----
        if self.jepa_config.loss_type == "vicreg":
            self.jepa_loss_fn = VICRegLoss(
                var_weight=self.jepa_config.vicreg_var_weight,
                inv_weight=self.jepa_config.vicreg_inv_weight,
                cov_weight=self.jepa_config.vicreg_cov_weight,
            )
        elif self.jepa_config.loss_type == "cosine":
            self.jepa_loss_fn = None  # Use F.cosine_similarity directly
        elif self.jepa_config.loss_type == "mse":
            self.jepa_loss_fn = None  # Use F.mse_loss directly

        # ---- Training state ----
        self._step_count: int = 0

        # Log parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        jepa_params = sum(
            p.numel()
            for n, p in self.named_parameters()
            if "latent_predictor" in n or "online_encoder" in n
        )
        logger.info(
            f"LLMJEPA: {total_params:,} total parameters, "
            f"{trainable_params:,} trainable, "
            f"{jepa_params:,} JEPA-specific"
        )

    def _init_target_encoder(self) -> None:
        """Initialize target encoder weights as a copy of online encoder.

        Ensures the teacher starts with identical weights to the student,
        providing a warm start for the EMA updates.
        """
        online_state = self.online_encoder.state_dict()
        target_state = self.target_encoder.state_dict()

        # Map parameter names (both use 'encoder.X' internally for target,
        # so we need to adjust)
        new_target_state = {}
        for name, param in online_state.items():
            target_name = f"encoder.{name}"
            if target_name in target_state:
                new_target_state[target_name] = param.clone()

        self.target_encoder.load_state_dict(new_target_state, strict=False)

        # Freeze target encoder — it should never receive gradients
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def update_teacher(self) -> None:
        """Update target encoder weights via EMA from online encoder.

        Should be called after each training step (after optimizer.step())
        to maintain the slowly-evolving teacher network. Uses the
        teacher_ema_decay from JEPAConfig.

        The EMA update rule:
            θ_target ← α * θ_target + (1 - α) * θ_online
        where α = teacher_ema_decay.
        """
        self.target_encoder.update_from_encoder(
            self.online_encoder,
            ema_decay=self.jepa_config.teacher_ema_decay,
        )

    def _compute_jepa_loss(
        self,
        predicted_latents: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute JEPA loss between predicted and target latent states.

        Dispatches to the appropriate loss function based on
        JEPAConfig.loss_type.

        Args:
            predicted_latents: Predicted future latents,
                shape (batch, seq_len, prediction_horizon, latent_dim).
            target_latents: Target future latents (same shape).

        Returns:
            Tuple (jepa_loss, loss_components_dict).
        """
        if self.jepa_config.loss_type == "vicreg":
            return self.jepa_loss_fn(predicted_latents, target_latents)

        elif self.jepa_config.loss_type == "cosine":
            # Cosine similarity loss: 1 - mean(cos_sim)
            cos_sim = F.cosine_similarity(
                predicted_latents, target_latents, dim=-1
            )
            loss = 1.0 - cos_sim.mean()
            components = {
                "vicreg_total": loss.item(),
                "cosine_loss": loss.item(),
                "mean_cos_sim": cos_sim.mean().item(),
            }
            return loss, components

        elif self.jepa_config.loss_type == "mse":
            loss = F.mse_loss(predicted_latents, target_latents)
            components = {
                "vicreg_total": loss.item(),
                "mse_loss": loss.item(),
            }
            return loss, components

        else:
            raise ValueError(f"Unknown loss_type: {self.jepa_config.loss_type}")

    def _construct_future_targets(
        self,
        target_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Construct target latent states for future positions.

        Given target latents of shape (B, S, latent_dim), creates
        targets for prediction_horizon future steps by shifting the
        sequence. For horizon step h, the target at position t is
        the latent at position t+h.

        For positions where t+h >= S, we repeat the last valid latent
        (boundary handling).

        Args:
            target_latents: Target latent representations,
                shape (batch, seq_len, latent_dim).

        Returns:
            Future target latents,
            shape (batch, seq_len, prediction_horizon, latent_dim).
        """
        batch, seq_len, latent_dim = target_latents.shape
        H = self.jepa_config.prediction_horizon

        future_targets = []
        for h in range(1, H + 1):
            # Shift by h positions
            if h < seq_len:
                shifted = target_latents[:, h:, :]  # (B, S-h, latent_dim)
                # Pad the end by repeating the last position
                pad = target_latents[:, -1:, :].expand(batch, h, latent_dim)
                shifted = torch.cat([shifted, pad], dim=1)
            else:
                # If shift >= seq_len, just repeat the last position
                shifted = target_latents[:, -1:, :].expand(batch, seq_len, latent_dim)
            future_targets.append(shifted)

        # Stack along new dimension: (B, S, H, latent_dim)
        future_targets = torch.stack(future_targets, dim=2)
        return future_targets

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        thinking_mode: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Forward pass combining standard LM and JEPA objectives.

        Performs:
        a) Main model processes input → standard LM loss + hidden states
        b) LatentPredictor predicts future hidden states from current states
        c) TargetEncoder provides target representations (EMA, no gradient)
        d) JEPA loss = loss_fn(predicted, target)
        e) Total loss = LM_loss + prediction_weight * JEPA_loss

        Args:
            input_ids: Token IDs, shape (batch, seq_len).
            labels: Target labels for LM loss, shape (batch, seq_len).
                Use -100 for positions to ignore.
            attention_mask: Optional attention mask.
            thinking_mode: If True, bias towards thinking pathways.

        Returns:
            Tuple (logits, total_loss, loss_dict):
                - logits: LM head output, (batch, seq_len, vocab_size).
                - total_loss: Combined LM + JEPA loss (None if no labels).
                - loss_dict: Dictionary with detailed loss components
                  for logging (lm_loss, jepa_loss, individual VICReg
                  terms, etc.).
        """
        # ---- Step 1: Standard LM forward ----
        lm_output: LosionCausalLMOutput = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            thinking_mode=thinking_mode,
        )

        logits = lm_output.logits
        lm_loss = lm_output.loss

        loss_dict: Dict[str, Any] = {
            "lm_loss": lm_loss.item() if lm_loss is not None else 0.0,
        }
        if lm_output.ar_loss is not None:
            loss_dict["ar_loss"] = lm_output.ar_loss.item()
        if lm_output.mtp_loss is not None:
            loss_dict["mtp_loss"] = lm_output.mtp_loss.item()

        # ---- Step 2: JEPA forward (only during training) ----
        jepa_loss = torch.tensor(0.0, device=input_ids.device)

        if self.training and labels is not None:
            hidden_states = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                thinking_mode=thinking_mode,
            ).hidden_states  # (B, S, d_model)

            # 2a. Encode current hidden states with online encoder
            #     (receives gradients — this is the "student" path)
            online_latents = self.online_encoder(hidden_states)
            # Shape: (B, S, latent_dim)

            # 2b. Predict future latent states from current online latents
            predicted_latents = self.latent_predictor(online_latents)
            # Shape: (B, S, H, latent_dim)

            # 2c. Encode with target encoder (EMA teacher, no gradients)
            #     Provides stable target representations
            with torch.no_grad():
                target_latents = self.target_encoder(hidden_states)
                target_latents = target_latents.detach()
            # Shape: (B, S, latent_dim)

            # 2d. Construct future targets by shifting target latents
            future_targets = self._construct_future_targets(target_latents)
            # Shape: (B, S, H, latent_dim)

            # 2e. Compute JEPA loss between predicted and target latents
            jepa_loss, jepa_components = self._compute_jepa_loss(
                predicted_latents, future_targets
            )

            # Merge JEPA components into loss_dict
            loss_dict.update(jepa_components)

            # Update teacher via EMA after loss computation
            self.update_teacher()

        # ---- Step 3: Combine losses ----
        total_loss: Optional[torch.Tensor] = None
        if lm_loss is not None:
            total_loss = lm_loss + self.jepa_config.prediction_weight * jepa_loss
            loss_dict["jepa_loss"] = jepa_loss.item()
            loss_dict["total_loss"] = total_loss.item()
        else:
            loss_dict["jepa_loss"] = 0.0
            loss_dict["total_loss"] = 0.0

        self._step_count += 1

        return logits, total_loss, loss_dict

    def get_model(self) -> LosionForCausalLM:
        """Get the underlying LosionForCausalLM model.

        Useful for checkpoint saving, evaluation, or inference
        where the JEPA components are not needed.

        Returns:
            The base LosionForCausalLM model.
        """
        return self.model

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by category.

        Returns:
            Dictionary with parameter counts for the base model,
            JEPA-specific components, and total.
        """
        total = sum(p.numel() for p in self.parameters())
        base_params = sum(p.numel() for p in self.model.parameters())
        predictor_params = sum(p.numel() for p in self.latent_predictor.parameters())
        online_enc_params = sum(p.numel() for p in self.online_encoder.parameters())
        target_enc_params = sum(p.numel() for p in self.target_encoder.parameters())

        jepa_params = predictor_params + online_enc_params + target_enc_params

        return {
            "total": total,
            "base_model": base_params,
            "latent_predictor": predictor_params,
            "online_encoder": online_enc_params,
            "target_encoder": target_enc_params,
            "jepa_total": jepa_params,
        }
