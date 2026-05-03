"""
Losion Causal LM — Language modeling head for the Losion Framework.

Implements LosionForCausalLM and LosionCausalLMOutput for autoregressive
language modeling with:
  - Cross-entropy loss computation (with -100 ignore index)
  - Multi-Token Prediction (MTP) loss
  - save_pretrained / from_pretrained for checkpoint management
  - Thinking mode support
  - Evoformer recycling (optional)

Hardware: Pure PyTorch, compatible with CUDA / ROCm / CPU.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losion.config import LosionConfig
from losion.models.losion_model import LosionModel


# ============================================================================
# LosionCausalLMOutput
# ============================================================================


@dataclass
class LosionCausalLMOutput:
    """Output from LosionForCausalLM.

    Attributes:
        logits: Predicted logits (batch, seq_len, vocab_size).
        loss: Total loss (if labels provided).
        ar_loss: Autoregressive cross-entropy loss.
        mtp_loss: Multi-Token Prediction loss (if MTP enabled).
        routing_info: Routing information from backbone.
        recycled_logits: Logits from Evoformer recycling steps (optional).
    """
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    ar_loss: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    routing_info: Optional[Any] = None
    recycled_logits: Optional[List[torch.Tensor]] = None


# ============================================================================
# MTP Head — Multi-Token Prediction
# ============================================================================


class MTPHead(nn.Module):
    """Multi-Token Prediction head.

    Predicts multiple future tokens in parallel from the current hidden state.
    Used for training efficiency and speculative decoding.

    Args:
        d_model: Model dimension.
        vocab_size: Vocabulary size.
        num_tokens: Number of future tokens to predict.
    """

    def __init__(self, d_model: int, vocab_size: int, num_tokens: int = 2) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.projections = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(num_tokens)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through MTP head.

        Args:
            hidden_states: Hidden states (batch, seq_len, d_model).
            labels: Target labels (batch, seq_len), with -100 for ignored positions.

        Returns:
            Tuple (mtp_loss, mtp_logits_list):
                - mtp_loss: Average MTP loss (None if labels not provided).
                - mtp_logits_list: List of logits for each future token position.
        """
        mtp_logits_list = []
        mtp_losses = []

        for i, proj in enumerate(self.projections):
            # Predict token at position (i+1) steps ahead
            logits = proj(hidden_states)  # (batch, seq_len, vocab_size)
            mtp_logits_list.append(logits)

            if labels is not None:
                # Shift: predict future token
                shift = i + 1
                if shift < labels.shape[1]:
                    shift_logits = logits[:, :-shift, :].contiguous()
                    shift_labels = labels[:, shift:].contiguous()

                    # Compute cross-entropy with ignore_index=-100
                    loss = F.cross_entropy(
                        shift_logits.view(-1, logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    mtp_losses.append(loss)

        mtp_loss = None
        if mtp_losses:
            mtp_loss = torch.stack(mtp_losses).mean()

        return mtp_loss, mtp_logits_list


# ============================================================================
# LosionForCausalLM
# ============================================================================


class LosionForCausalLM(nn.Module):
    """Losion model with a causal language modeling head.

    Wraps LosionModel backbone and adds:
    - LM head for next-token prediction
    - MTP head for multi-token prediction (optional)
    - Loss computation with label smoothing
    - save_pretrained / from_pretrained for checkpoint management

    Args:
        config: LosionConfig with model parameters.
    """

    def __init__(self, config: LosionConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        # ---- Backbone ----
        self.model = LosionModel(config)

        # ---- LM head ----
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ---- MTP head (optional) ----
        self.mtp_head: Optional[MTPHead] = None
        if config.output.use_mtp:
            self.mtp_head = MTPHead(
                d_model=config.d_model,
                vocab_size=config.vocab_size,
                num_tokens=config.output.mtp_num_tokens,
            )

        # ---- Initialize weights ----
        self.lm_head.apply(self._init_weights)
        if self.mtp_head is not None:
            self.mtp_head.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Standard weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_model(self) -> LosionModel:
        """Get the backbone LosionModel."""
        return self.model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        thinking_mode: Optional[bool] = None,
        return_routing_info: bool = False,
        use_evo_recycling: bool = False,
    ) -> LosionCausalLMOutput:
        """Forward pass for causal language modeling.

        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Optional attention mask.
            labels: Target labels for loss computation (batch, seq_len).
                Use -100 for positions to ignore.
            thinking_mode: If True, bias towards thinking pathways.
            return_routing_info: If True, return routing info.
            use_evo_recycling: If True, use Evoformer recycling (iterative refinement).

        Returns:
            LosionCausalLMOutput with logits, loss, and optional aux info.
        """
        # ---- Backbone forward ----
        backbone_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            thinking_mode=thinking_mode,
            return_routing_info=return_routing_info,
        )
        hidden_states = backbone_output.hidden_states  # (batch, seq_len, d_model)

        # ---- LM head ----
        logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)

        # ---- Evoformer recycling (optional, training only) ----
        recycled_logits: Optional[List[torch.Tensor]] = None
        if use_evo_recycling and self.training:
            recycled_logits = [logits]
            # Simplified: one recycling step
            with torch.no_grad():
                # Use current logits to create soft targets, then re-run
                # (In practice, this would be more sophisticated)
                pass
            recycled_logits.append(logits)  # Same logits for stub

        # ---- Loss computation ----
        loss: Optional[torch.Tensor] = None
        ar_loss: Optional[torch.Tensor] = None
        mtp_loss: Optional[torch.Tensor] = None

        if labels is not None:
            # Autoregressive (next-token) loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ar_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = ar_loss

            # MTP loss
            if self.mtp_head is not None:
                mtp_loss, _ = self.mtp_head(hidden_states, labels=labels)
                if mtp_loss is not None:
                    loss = loss + 0.1 * mtp_loss  # MTP with 0.1 weight

        return LosionCausalLMOutput(
            logits=logits,
            loss=loss,
            ar_loss=ar_loss,
            mtp_loss=mtp_loss,
            routing_info=backbone_output.routing_info,
            recycled_logits=recycled_logits,
        )

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by category.

        Returns:
            Dictionary with parameter counts including 'lm_head' key.
        """
        total = 0
        lm_head = 0

        for name, param in self.named_parameters():
            n = param.numel()
            total += n
            if "lm_head" in name:
                lm_head += n

        # Include backbone params
        backbone_counts = self.model.count_parameters()

        return {
            "total": total,
            "lm_head": lm_head,
            **{f"backbone_{k}": v for k, v in backbone_counts.items()},
        }

    def save_pretrained(self, save_directory: str) -> None:
        """Save model and configuration to a directory.

        Saves:
        - config.json: Model configuration
        - model.pt: Model state dict

        Args:
            save_directory: Directory to save to (created if it doesn't exist).
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_dict = self.config.to_dict()
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        # Save model state dict
        model_path = os.path.join(save_directory, "model.pt")
        torch.save(self.state_dict(), model_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: str = "cpu",
    ) -> "LosionForCausalLM":
        """Load model from a saved checkpoint.

        Args:
            pretrained_path: Path to the directory containing config.json and model.pt.
            device: Device to load the model to (default "cpu").

        Returns:
            LosionForCausalLM instance with loaded weights.
        """
        # Load config
        config_path = os.path.join(pretrained_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Reconstruct config from dict
        config = LosionConfig._from_dict(config_dict)

        # Create model
        model = cls(config)

        # Load state dict
        model_path = os.path.join(pretrained_path, "model.pt")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        if device != "cpu":
            model = model.to(device)

        return model
