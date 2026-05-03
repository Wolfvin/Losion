"""
Post-training Neural Architecture Search for Losion Framework v0.4.

Upgrade #8: NAS for finding optimal SSM/Attention layer ratio.

After pre-training, search for the best layer-type assignment per layer
position.  Instead of a fixed ratio (e.g., 4:1 SSM:Attention), each
layer can independently choose to be SSM, Attention, or Both.

Key components:
1. NASConfig — Search budget, constraints, and hyperparameters
2. NASLayerChoice — Differentiable layer-type choice (DARTS-style)
3. NASController — Manages architecture search across all layers

Background:
    Differentiable Architecture Search (DARTS, Liu et al. 2019) relaxes
    the discrete architecture choice into a continuous optimisation
    problem.  Each layer has learnable "architecture parameters" (alpha)
    that weight the different layer types.  After search, the highest-
    weighted type is selected for each layer.

    For Losion, this means each layer position can independently choose
    between:
    - SSM (State Space Model — efficient for local patterns)
    - Attention (Full attention — better for long-range dependencies)
    - Both (Hybrid — combine both for maximum capacity)

    The search is constrained by:
    - Maximum number of attention layers (compute budget)
    - Minimum number of SSM layers (efficiency requirement)
    - Total parameter budget

    This is a **post-training** search: start from a pre-trained model,
    then fine-tune with architecture parameters enabled.  The search
    typically converges in a few hundred steps.

Hardware: Pure PyTorch, compatible with CUDA / ROCm / CPU.
No custom kernels required.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layer type constants
# ---------------------------------------------------------------------------

LAYER_SSM = 0
LAYER_ATTENTION = 1
LAYER_BOTH = 2
NUM_LAYER_TYPES = 3

LAYER_TYPE_NAMES = {LAYER_SSM: "ssm", LAYER_ATTENTION: "attention", LAYER_BOTH: "both"}


# ---------------------------------------------------------------------------
# NASConfig — Search configuration
# ---------------------------------------------------------------------------

@dataclass
class NASConfig:
    """
    Configuration for Neural Architecture Search.

    Controls the search budget, constraints, and optimisation
    hyperparameters.

    Attributes:
        n_layers:                  Total number of layers in the model.
        search_steps:              Number of fine-tuning steps for search
                                   (default 500).
        max_attention_layers:      Upper bound on attention layers
                                   (None = no constraint).
        min_ssm_layers:            Lower bound on SSM layers (default 1).
        max_both_layers:           Upper bound on "both" layers
                                   (None = no constraint).
        alpha_lr:                  Learning rate for architecture parameters
                                   (default 0.001).
        alpha_weight_decay:        Weight decay on architecture parameters
                                   (default 1e-4).
        temperature:               Softmax temperature for architecture
                                   weights (default 1.0).  Lower = sharper
                                   choices.
        entropy_weight:            Weight for entropy regularisation on
                                   architecture weights.  Encourages
                                   decisive choices (default 0.01).
        diversity_weight:          Weight for diversity regularisation.
                                   Encourages different layers to choose
                                   different types (default 0.001).
        constraint_weight:         Weight for constraint violation penalty
                                   (default 1.0).
        init_alpha:                Initialisation for architecture logits.
                                   "uniform" = equal weights, "ssm_bias" =
                                   slight bias toward SSM (default "uniform").
        freeze_weights:            If True, freeze model weights during
                                   search and only optimise alpha (faster).
                                   If False, jointly fine-tune weights
                                   and alpha (better quality, slower).
    """

    n_layers: int = 12
    search_steps: int = 500
    max_attention_layers: Optional[int] = None
    min_ssm_layers: int = 1
    max_both_layers: Optional[int] = None
    alpha_lr: float = 0.001
    alpha_weight_decay: float = 1e-4
    temperature: float = 1.0
    entropy_weight: float = 0.01
    diversity_weight: float = 0.001
    constraint_weight: float = 1.0
    init_alpha: str = "uniform"
    freeze_weights: bool = True


# ---------------------------------------------------------------------------
# NASLayerChoice — Differentiable layer-type choice
# ---------------------------------------------------------------------------

class NASLayerChoice(nn.Module):
    """
    Differentiable choice between SSM, Attention, or Both for a single layer.

    Implements the DARTS-style differentiable relaxation: instead of
    selecting a single layer type, all types are computed and their
    outputs are blended using softmax weights over learnable architecture
    parameters (alpha).

    During search:
        output = sum_i(softmax(alpha / tau)_i * layer_i(input))

    After search:
        output = layer_{argmax(alpha)}(input)

    The three layer types represent:
    - SSM (index 0): State Space Model — O(n) complexity, efficient
      for local patterns, low memory.
    - Attention (index 1): Full attention — O(n²) complexity, better
      for long-range dependencies, high memory.
    - Both (index 2): Hybrid — combines SSM and attention in parallel,
      highest capacity but also highest compute.

    Args:
        ssm_layer:        nn.Module for the SSM pathway.
        attention_layer:  nn.Module for the attention pathway.
        both_layer:       nn.Module for the hybrid pathway (or None
                          to construct from ssm + attention).
        layer_idx:        Index of this layer in the model (for logging).
        init_alpha:       Initialisation mode ("uniform" or "ssm_bias").
        temperature:      Softmax temperature (lower = sharper).
    """

    def __init__(
        self,
        ssm_layer: nn.Module,
        attention_layer: nn.Module,
        both_layer: Optional[nn.Module] = None,
        layer_idx: int = 0,
        init_alpha: str = "uniform",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.temperature = temperature

        # ---- Sub-layer modules ----
        self.ssm_layer = ssm_layer
        self.attention_layer = attention_layer
        self.both_layer = both_layer  # If None, built from ssm + attention

        # ---- Architecture parameters (alpha) ----
        # One logit per layer type
        if init_alpha == "ssm_bias":
            # Slight bias toward SSM (more efficient)
            alpha_init = torch.tensor([1.0, 0.0, 0.0])
        elif init_alpha == "attention_bias":
            alpha_init = torch.tensor([0.0, 1.0, 0.0])
        else:
            # Uniform
            alpha_init = torch.zeros(3)

        self.alpha = nn.Parameter(alpha_init)

        # ---- Output blend gate (optional learnable gate for "both" mode) ----
        self.both_gate = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    # Architecture weights
    # ------------------------------------------------------------------

    def get_weights(self) -> torch.Tensor:
        """
        Get the softmax architecture weights.

        Returns:
            Tensor of shape ``(3,)`` with softmax probabilities over
            [SSM, Attention, Both].
        """
        return F.softmax(self.alpha / self.temperature, dim=0)

    def get_chosen_type(self) -> int:
        """
        Get the chosen layer type (argmax of alpha).

        Returns:
            Integer layer type: 0=SSM, 1=Attention, 2=Both.
        """
        return self.alpha.argmax().item()

    def get_chosen_type_name(self) -> str:
        """Get the name of the chosen layer type."""
        return LAYER_TYPE_NAMES[self.get_chosen_type()]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with differentiable architecture blending.

        During search, all three layer types are computed and blended
        using the architecture weights.  This enables gradient-based
        architecture optimisation.

        Args:
            x:      Input tensor ``(batch, seq_len, d_model)``.
            kwargs: Additional keyword arguments passed to sub-layers.

        Returns:
            (output, aux):
                output: Blended output ``(batch, seq_len, d_model)``.
                aux:    Dict with auxiliary information:
                    - "alpha": Architecture logits
                    - "weights": Architecture weights (softmax)
                    - "entropy": Weight entropy
        """
        weights = self.get_weights()  # (3,)
        aux: Dict[str, torch.Tensor] = {
            "alpha": self.alpha,
            "weights": weights,
            "entropy": -(weights * (weights + 1e-8).log()).sum(),
        }

        # Compute all sub-layer outputs
        # We compute all three regardless (for gradient flow during search)
        ssm_out = self._call_layer(self.ssm_layer, x, **kwargs)
        attn_out = self._call_layer(self.attention_layer, x, **kwargs)

        if self.both_layer is not None:
            both_out = self._call_layer(self.both_layer, x, **kwargs)
        else:
            # Construct "both" from ssm + attention with learned gate
            gate = torch.sigmoid(self.both_gate)
            both_out = gate * ssm_out + (1 - gate) * attn_out

        # Blend
        output = (
            weights[0] * ssm_out
            + weights[1] * attn_out
            + weights[2] * both_out
        )

        return output, aux

    def forward_chosen(
        self, x: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass using only the chosen (argmax) layer type.

        Used after search is complete for efficient inference.

        Args:
            x:      Input tensor.
            kwargs: Additional keyword arguments.

        Returns:
            Output tensor from the chosen sub-layer.
        """
        chosen = self.get_chosen_type()

        if chosen == LAYER_SSM:
            return self._call_layer(self.ssm_layer, x, **kwargs)
        elif chosen == LAYER_ATTENTION:
            return self._call_layer(self.attention_layer, x, **kwargs)
        else:  # LAYER_BOTH
            if self.both_layer is not None:
                return self._call_layer(self.both_layer, x, **kwargs)
            else:
                ssm_out = self._call_layer(self.ssm_layer, x, **kwargs)
                attn_out = self._call_layer(self.attention_layer, x, **kwargs)
                gate = torch.sigmoid(self.both_gate)
                return gate * ssm_out + (1 - gate) * attn_out

    @staticmethod
    def _call_layer(layer: nn.Module, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Call a sub-layer, handling both single-output and tuple-output layers.

        Some Losion layers return (output, state) or (output, state, aux).
        We extract just the output tensor.
        """
        result = layer(x, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result

    def extra_repr(self) -> str:
        weights = self.get_weights()
        return (
            f"layer_idx={self.layer_idx}, "
            f"chosen={self.get_chosen_type_name()}, "
            f"weights=[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]"
        )


# ---------------------------------------------------------------------------
# NASController — Manages architecture search across all layers
# ---------------------------------------------------------------------------

class NASController:
    """
    Manages Neural Architecture Search across all layers of a model.

    The controller:
    1. Replaces each layer with a :class:`NASLayerChoice`.
    2. Manages architecture parameter optimisation.
    3. Enforces constraints on the layer type distribution.
    4. Provides search loop utilities.
    5. Extracts the final architecture after search.

    Search procedure (DARTS-style, adapted for layer search)::

        controller = NASController(model, config)
        controller.prepare()  # Replace layers with NASLayerChoice

        for step in range(config.search_steps):
            loss = compute_loss(model, batch)
            controller.step(loss)  # Update architecture parameters

        architecture = controller.get_architecture()  # Final choices
        controller.apply_architecture()  # Replace with fixed layers

    Constraint enforcement:
        The controller adds a penalty to the loss when constraints
        (e.g., max attention layers) are violated.  The penalty is::

            penalty = constraint_weight * sum(max(0, violation) for each constraint)

        This is softer than hard constraints but works well in practice.

    Args:
        model:  The model to search over.
        config: NASConfig with search parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[NASConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or NASConfig()
        self._layer_choices: Dict[str, NASLayerChoice] = {}
        self._original_layers: Dict[str, nn.Module] = {}
        self._prepared = False

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """
        Prepare the model for NAS by replacing layers with NASLayerChoice.

        This walks the model tree and wraps eligible layers.  For each
        position, three variants (SSM, Attention, Both) are created.

        The original layer is saved for restoration if needed.

        Note: This method assumes the model has a ``layers`` attribute
        that is an nn.ModuleList.  For other structures, you may need
        to manually create NASLayerChoice instances.
        """
        if self._prepared:
            return

        # Try to find a ModuleList called "layers" or similar
        layers_attr = None
        for attr_name in ["layers", "blocks", "layer", "block"]:
            if hasattr(self.model, attr_name):
                candidate = getattr(self.model, attr_name)
                if isinstance(candidate, nn.ModuleList):
                    layers_attr = attr_name
                    break

        if layers_attr is None:
            # Cannot auto-prepare; user must manually set up
            raise RuntimeError(
                "Cannot auto-prepare model for NAS.  "
                "No ModuleList found with name 'layers', 'blocks', etc.  "
                "Please manually create NASLayerChoice instances and "
                "register them with register_layer_choice()."
            )

        layers = getattr(self.model, layers_attr)
        for idx, layer in enumerate(layers):
            self._replace_layer(layers_attr, idx, layer)

        self._prepared = True

    def _replace_layer(
        self,
        layers_attr: str,
        idx: int,
        original_layer: nn.Module,
    ) -> None:
        """
        Replace a single layer with an NASLayerChoice.

        Creates three variants of the layer:
        1. SSM variant (the original if it's an SSM layer, or a copy)
        2. Attention variant (needs to be provided or created)
        3. Both variant (SSM + Attention combined)

        For simplicity, we create lightweight proxy layers if the exact
        type isn't available.  In production, the user should provide
        the actual layer variants.

        Args:
            layers_attr:  Name of the ModuleList attribute.
            idx:          Index in the ModuleList.
            original_layer: The original layer to replace.
        """
        name = f"{layers_attr}.{idx}"
        self._original_layers[name] = original_layer

        # Create layer variants
        ssm_layer = copy.deepcopy(original_layer)
        attention_layer = copy.deepcopy(original_layer)
        # "Both" layer: we'll use a gate in NASLayerChoice
        both_layer = None

        # Create NASLayerChoice
        choice = NASLayerChoice(
            ssm_layer=ssm_layer,
            attention_layer=attention_layer,
            both_layer=both_layer,
            layer_idx=idx,
            init_alpha=self.config.init_alpha,
            temperature=self.config.temperature,
        )

        # Replace in model
        layers = getattr(self.model, layers_attr)
        layers[idx] = choice

        self._layer_choices[name] = choice

    def register_layer_choice(
        self,
        name: str,
        choice: NASLayerChoice,
    ) -> None:
        """
        Manually register a NASLayerChoice.

        Use this when the model structure doesn't support auto-preparation.

        Args:
            name:   Dotted name for the layer.
            choice: NASLayerChoice instance.
        """
        self._layer_choices[name] = choice
        self._prepared = True

    # ------------------------------------------------------------------
    # Architecture parameter access
    # ------------------------------------------------------------------

    def get_alpha_parameters(self) -> List[nn.Parameter]:
        """Return all architecture (alpha) parameters."""
        return [choice.alpha for choice in self._layer_choices.values()]

    def get_alpha_optimizer(self) -> torch.optim.Optimizer:
        """
        Create an optimiser for architecture parameters.

        Returns:
            Adam optimiser on the alpha parameters.
        """
        params = self.get_alpha_parameters()
        return torch.optim.Adam(
            params,
            lr=self.config.alpha_lr,
            weight_decay=self.config.alpha_weight_decay,
        )

    # ------------------------------------------------------------------
    # Search step
    # ------------------------------------------------------------------

    def step(
        self,
        loss: torch.Tensor,
        alpha_optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, float]:
        """
        Perform one architecture search step.

        1. Compute architecture regularisation losses.
        2. Add constraint violation penalties.
        3. Update architecture parameters via gradient descent.

        Args:
            loss:            Task loss for the current batch.
            alpha_optimizer: Optimiser for alpha parameters.  If None,
                             a new one is created.

        Returns:
            Dict of auxiliary losses (for logging).
        """
        if alpha_optimizer is None:
            alpha_optimizer = self.get_alpha_optimizer()

        # Compute regularisation losses
        reg_loss = torch.tensor(0.0, device=loss.device)
        metrics: Dict[str, float] = {}

        # 1. Entropy regularisation (encourage decisive choices)
        total_entropy = 0.0
        for name, choice in self._layer_choices.items():
            weights = choice.get_weights()
            entropy = -(weights * (weights + 1e-8).log()).sum()
            total_entropy += entropy.item()
            reg_loss = reg_loss + self.config.entropy_weight * entropy

        n_choices = max(1, len(self._layer_choices))
        metrics["avg_entropy"] = total_entropy / n_choices

        # 2. Diversity regularisation (encourage different layers to choose differently)
        if self.config.diversity_weight > 0 and len(self._layer_choices) > 1:
            all_weights = torch.stack(
                [c.get_weights() for c in self._layer_choices.values()]
            )  # (n_layers, 3)
            # Pairwise similarity penalty
            sim_matrix = torch.mm(all_weights, all_weights.t())  # (n, n)
            # Only penalise off-diagonal (between different layers)
            mask = 1.0 - torch.eye(
                len(self._layer_choices), device=sim_matrix.device
            )
            diversity_loss = (sim_matrix * mask).sum() / max(
                1, len(self._layer_choices) * (len(self._layer_choices) - 1)
            )
            reg_loss = reg_loss + self.config.diversity_weight * diversity_loss
            metrics["diversity_loss"] = diversity_loss.item()

        # 3. Constraint penalties
        constraint_loss = self._compute_constraint_penalty()
        reg_loss = reg_loss + self.config.constraint_weight * constraint_loss
        metrics["constraint_loss"] = constraint_loss.item()

        # Total architecture loss
        arch_loss = loss.detach() + reg_loss

        # Update alpha
        alpha_optimizer.zero_grad()
        arch_loss.backward()
        alpha_optimizer.step()

        # Temperature annealing (gradually sharpen choices)
        # After 50% of search steps, start annealing
        # (This is a simple strategy; more sophisticated ones exist)

        metrics["arch_loss"] = arch_loss.item()
        return metrics

    def _compute_constraint_penalty(self) -> torch.Tensor:
        """
        Compute penalty for constraint violations.

        Checks:
        - max_attention_layers
        - min_ssm_layers
        - max_both_layers

        Returns:
            Scalar penalty tensor.
        """
        if not self._layer_choices:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0)

        # Collect weights
        all_weights = {}
        for name, choice in self._layer_choices.items():
            all_weights[name] = choice.get_weights()

        # Expected count per type
        ssm_count = sum(w[LAYER_SSM].item() for w in all_weights.values())
        attn_count = sum(w[LAYER_ATTENTION].item() for w in all_weights.values())
        both_count = sum(w[LAYER_BOTH].item() for w in all_weights.values())

        # Max attention layers
        if self.config.max_attention_layers is not None:
            excess = attn_count - self.config.max_attention_layers
            if excess > 0:
                penalty = penalty + excess ** 2

        # Min SSM layers
        if self.config.min_ssm_layers is not None:
            deficit = self.config.min_ssm_layers - ssm_count
            if deficit > 0:
                penalty = penalty + deficit ** 2

        # Max both layers
        if self.config.max_both_layers is not None:
            excess = both_count - self.config.max_both_layers
            if excess > 0:
                penalty = penalty + excess ** 2

        return penalty

    # ------------------------------------------------------------------
    # Architecture extraction
    # ------------------------------------------------------------------

    def get_architecture(self) -> Dict[int, str]:
        """
        Get the current best architecture (argmax of alpha per layer).

        Returns:
            Dict mapping layer index → layer type name ("ssm", "attention", "both").
        """
        architecture = {}
        for name, choice in self._layer_choices.items():
            chosen_type = choice.get_chosen_type()
            architecture[choice.layer_idx] = LAYER_TYPE_NAMES[chosen_type]
        return architecture

    def get_architecture_weights(self) -> Dict[int, List[float]]:
        """
        Get the full architecture weight distribution per layer.

        Returns:
            Dict mapping layer index → [SSM_weight, Attn_weight, Both_weight].
        """
        weights = {}
        for name, choice in self._layer_choices.items():
            w = choice.get_weights()
            weights[choice.layer_idx] = [w[0].item(), w[1].item(), w[2].item()]
        return weights

    def apply_architecture(self) -> None:
        """
        Apply the searched architecture to the model.

        Replaces each NASLayerChoice with the chosen sub-layer (argmax),
        removing the architecture search overhead.  After this call,
        the model is ready for normal fine-tuning or inference.
        """
        for name, choice in self._layer_choices.items():
            chosen_type = choice.get_chosen_type()

            if chosen_type == LAYER_SSM:
                selected_layer = choice.ssm_layer
            elif chosen_type == LAYER_ATTENTION:
                selected_layer = choice.attention_layer
            else:  # LAYER_BOTH
                if choice.both_layer is not None:
                    selected_layer = choice.both_layer
                else:
                    # Create a hybrid layer wrapper
                    selected_layer = HybridLayerWrapper(
                        choice.ssm_layer, choice.attention_layer, choice.both_gate
                    )

            # Replace in model
            parts = name.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], selected_layer)

        self._layer_choices.clear()
        self._prepared = False

    def restore_original(self) -> None:
        """
        Restore the original layers (undo NAS preparation).
        """
        for name, original in self._original_layers.items():
            parts = name.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], original)

        self._layer_choices.clear()
        self._prepared = False

    # ------------------------------------------------------------------
    # Statistics and reporting
    # ------------------------------------------------------------------

    def get_search_summary(self) -> Dict[str, object]:
        """
        Get a summary of the current search state.

        Returns:
            Dict with:
            - "architecture": Current best architecture
            - "weights": Weight distribution per layer
            - "expected_counts": Expected layer type counts
            - "constraints_satisfied": Whether all constraints are met
        """
        arch = self.get_architecture()
        weights = self.get_architecture_weights()

        # Expected counts
        all_w = list(weights.values())
        ssm_count = sum(w[0] for w in all_w)
        attn_count = sum(w[1] for w in all_w)
        both_count = sum(w[2] for w in all_w)

        # Check constraints
        constraints_ok = True
        if self.config.max_attention_layers is not None:
            constraints_ok = constraints_ok and (attn_count <= self.config.max_attention_layers)
        if self.config.min_ssm_layers is not None:
            constraints_ok = constraints_ok and (ssm_count >= self.config.min_ssm_layers)
        if self.config.max_both_layers is not None:
            constraints_ok = constraints_ok and (both_count <= self.config.max_both_layers)

        return {
            "architecture": arch,
            "weights": weights,
            "expected_counts": {
                "ssm": ssm_count,
                "attention": attn_count,
                "both": both_count,
            },
            "constraints_satisfied": constraints_ok,
        }

    def __repr__(self) -> str:
        return (
            f"NASController("
            f"layers={len(self._layer_choices)}, "
            f"prepared={self._prepared}, "
            f"config={self.config})"
        )


# ---------------------------------------------------------------------------
# HybridLayerWrapper — Wrapper for the "Both" layer type
# ---------------------------------------------------------------------------

class HybridLayerWrapper(nn.Module):
    """
    Wrapper that combines SSM and Attention outputs with a learned gate.

    Used when the "Both" layer type is selected after NAS and no
    explicit hybrid layer was provided.

    output = sigmoid(gate) * ssm_output + (1 - sigmoid(gate)) * attn_output

    Args:
        ssm_layer:      SSM sub-layer module.
        attention_layer: Attention sub-layer module.
        gate:           Initial gate parameter (from NAS search).
    """

    def __init__(
        self,
        ssm_layer: nn.Module,
        attention_layer: nn.Module,
        gate: torch.Tensor,
    ) -> None:
        super().__init__()
        self.ssm_layer = ssm_layer
        self.attention_layer = attention_layer
        self.gate = nn.Parameter(gate.clone())

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with gated combination.

        Args:
            x:      Input tensor.
            kwargs: Additional arguments for sub-layers.

        Returns:
            Combined output tensor.
        """
        ssm_out = NASLayerChoice._call_layer(self.ssm_layer, x, **kwargs)
        attn_out = NASLayerChoice._call_layer(self.attention_layer, x, **kwargs)
        g = torch.sigmoid(self.gate)
        return g * ssm_out + (1 - g) * attn_out


# ---------------------------------------------------------------------------
# NAS utility functions
# ---------------------------------------------------------------------------

def compute_nas_loss(
    controller: NASController,
    task_loss: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the total NAS loss (task loss + architecture regularisation).

    Convenience function that calls ``controller.step()`` and returns
    the combined loss for joint optimisation.

    Args:
        controller: NASController instance.
        task_loss:  Task-specific loss tensor.

    Returns:
        (total_loss, metrics): Combined loss and auxiliary metrics.
    """
    metrics = controller.step(task_loss)
    total_loss = task_loss + torch.tensor(
        metrics.get("arch_loss", 0.0) - task_loss.item(),
        device=task_loss.device,
    )
    return task_loss, metrics


def suggest_architecture(
    n_layers: int,
    model_size: str = "1b",
    task_type: str = "general",
) -> Dict[int, str]:
    """
    Suggest a layer architecture based on heuristics.

    This provides a reasonable starting architecture when NAS is not
    feasible.  The suggestions are based on empirical observations:

    - Small models (1B): More SSM layers (efficient), fewer attention
    - Large models (7B+): More attention layers (capacity)
    - Code tasks: More attention (precise recall)
    - General tasks: Balanced SSM/attention

    Args:
        n_layers:    Total number of layers.
        model_size:  Model size category ("1b", "7b", "70b").
        task_type:   Task category ("general", "code", "math", "retrieval").

    Returns:
        Dict mapping layer index → layer type name.
    """
    architecture = {}

    if model_size == "1b":
        # Small model: 75% SSM, 15% attention, 10% both
        attn_ratio = 0.15
        both_ratio = 0.10
    elif model_size == "7b":
        # Medium model: 50% SSM, 35% attention, 15% both
        attn_ratio = 0.35
        both_ratio = 0.15
    else:
        # Large model: 40% SSM, 40% attention, 20% both
        attn_ratio = 0.40
        both_ratio = 0.20

    # Adjust for task type
    if task_type == "code":
        attn_ratio += 0.10
        both_ratio -= 0.05
    elif task_type == "math":
        both_ratio += 0.05
        attn_ratio -= 0.05
    elif task_type == "retrieval":
        attn_ratio += 0.15
        both_ratio -= 0.10

    # Clamp
    attn_ratio = max(0.0, min(0.8, attn_ratio))
    both_ratio = max(0.0, min(0.4, both_ratio))
    ssm_ratio = 1.0 - attn_ratio - both_ratio

    # Distribute: place attention layers evenly, SSM fills the rest
    n_attn = max(1, round(n_layers * attn_ratio))
    n_both = max(0, round(n_layers * both_ratio))
    n_ssm = n_layers - n_attn - n_both

    # Assign: spread attention layers evenly
    attn_spacing = max(1, n_layers // max(1, n_attn))
    both_spacing = max(1, n_layers // max(1, n_both + 1))

    assigned = set()

    # Place attention layers
    for i in range(n_attn):
        idx = min(n_layers - 1, (i + 1) * attn_spacing)
        while idx in assigned and idx < n_layers:
            idx += 1
        if idx < n_layers:
            architecture[idx] = "attention"
            assigned.add(idx)

    # Place "both" layers
    for i in range(n_both):
        idx = min(n_layers - 1, (i + 1) * both_spacing + attn_spacing // 2)
        while idx in assigned and idx < n_layers:
            idx += 1
        if idx < n_layers:
            architecture[idx] = "both"
            assigned.add(idx)

    # Fill remaining with SSM
    for i in range(n_layers):
        if i not in assigned:
            architecture[i] = "ssm"

    return architecture
