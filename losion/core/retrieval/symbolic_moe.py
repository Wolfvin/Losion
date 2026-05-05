"""
Symbolic-MoE — Skill-Based Discrete Routing for Mixture-of-Experts (2025).

Standard MoE routing is *learned* at the token level: a gating network
produces per-token scores over experts, and the top-K are selected.  While
effective, this approach is opaque — it is hard to predict or control which
pathway a given input will follow, and the router must rediscover routing
structure from data.

Symbolic-MoE takes a different approach inspired by cognitive architecture
principles: a **skill classifier** first assigns each token to a high-level
skill type (REASONING, NARRATIVE, KNOWLEDGE, …), and then a set of
**symbolic routing rules** maps each skill type to a fixed pathway
allocation.  For example:

    REASONING → {attention: 0.7, moe: 0.3, ssm: 0.0}
    NARRATIVE → {ssm: 0.8, attention: 0.1, moe: 0.1}

This is *discrete routing*: the skill type determines the pathway mix, not a
learned gate.  The advantage is interpretability and controllability —
domain experts can inspect and override the routing rules without retraining.
Symbolic-MoE can be combined with Losion's existing ``BiasRouter`` as a
macro-level controller that constrains the token-level router.

Architecture
------------
1. **SkillType** — Enum of recognised skill categories.

2. **SkillClassifier** — A small MLP that maps token representations to
   skill-type probability distributions.  Can be overridden manually for
   domain-specific routing.

3. **SymbolicRoutingRule** — A dataclass mapping each skill type to pathway
   allocation weights (attention, MoE, SSM).  Rules are customisable and
   can be loaded from configuration.

4. **SymbolicMoERouter** — Combines learned skill classification with
   symbolic routing rules.  Supports both **soft** (blended) and **hard**
   (discrete) routing.  Can be combined with ``BiasRouter`` as a
   macro-level controller.

References
----------
- Symbolic-MoE (2025) — Skill-based discrete routing instead of learned routing.
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models"
  (2022) — sparse expert routing.
- Cao et al., "Mixture of Experts survey" (2024) — comprehensive MoE taxonomy.
- Losion BiasRouter — Existing bias-based routing in ``core/router/bias_router.py``.

Hardware: Pure PyTorch.  No custom CUDA kernels required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SkillType Enum
# ============================================================================

class SkillType(Enum):
    """High-level skill categories for symbolic routing.

    Each skill type corresponds to a different cognitive processing
    pattern and maps to a preferred pathway allocation.

    Values:
        REASONING:  Logical deduction, multi-step inference, math proofs.
        NARRATIVE:  Story generation, dialogue, creative writing, flow.
        KNOWLEDGE:  Fact retrieval, encyclopedic QA, knowledge lookup.
        CODING:     Code generation, debugging, algorithm design.
        CREATIVE:   Brainstorming, poetry, metaphor, lateral thinking.
        MATHEMATICAL: Formal calculation, symbolic math, numerical tasks.
    """

    REASONING = "reasoning"
    NARRATIVE = "narrative"
    KNOWLEDGE = "knowledge"
    CODING = "coding"
    CREATIVE = "creative"
    MATHEMATICAL = "mathematical"


# Mapping from SkillType enum index to enum member (for tensor → enum)
_SKILL_TYPE_LIST = list(SkillType)
_NUM_SKILL_TYPES = len(_SKILL_TYPE_LIST)


# ============================================================================
# Default Routing Rules
# ============================================================================

def _default_routing_rules() -> Dict[SkillType, Dict[str, float]]:
    """Return the default symbolic routing rules.

    These are based on cognitive architecture principles:
      - REASONING benefits from focused attention (step-by-step).
      - NARRATIVE benefits from SSM pathways (long-range flow).
      - KNOWLEDGE benefits from MoE (diverse expert retrieval).
      - CODING benefits from attention + MoE (structured + diverse).
      - CREATIVE benefits from SSM + attention (flow + focus).
      - MATHEMATICAL benefits from attention + MoE (structured + retrieval).
    """
    return {
        SkillType.REASONING: {
            "attention": 0.7,
            "moe": 0.3,
            "ssm": 0.0,
        },
        SkillType.NARRATIVE: {
            "attention": 0.1,
            "moe": 0.1,
            "ssm": 0.8,
        },
        SkillType.KNOWLEDGE: {
            "attention": 0.2,
            "moe": 0.7,
            "ssm": 0.1,
        },
        SkillType.CODING: {
            "attention": 0.5,
            "moe": 0.4,
            "ssm": 0.1,
        },
        SkillType.CREATIVE: {
            "attention": 0.3,
            "moe": 0.1,
            "ssm": 0.6,
        },
        SkillType.MATHEMATICAL: {
            "attention": 0.6,
            "moe": 0.3,
            "ssm": 0.1,
        },
    }


# ============================================================================
# SymbolicRoutingRule — Skill → Pathway mapping
# ============================================================================

@dataclass
class SymbolicRoutingRule:
    """Maps skill types to pathway allocation weights.

    Each rule assigns a weight to three pathways — ``attention``, ``moe``,
    and ``ssm`` — for a given skill type.  Weights are normalised to sum
    to 1 during routing.

    The routing rules are customisable: domain experts can adjust them
    without retraining the model, providing interpretability and control.

    Attributes:
        rules: Dictionary mapping ``SkillType`` → pathway weight dict.
            Each inner dict has keys ``"attention"``, ``"moe"``, ``"ssm"``
            with float values.  Values are normalised at runtime.
        pathway_names: Ordered list of pathway names (default:
            ``["attention", "moe", "ssm"]``).
    """

    rules: Dict[SkillType, Dict[str, float]] = field(
        default_factory=_default_routing_rules
    )
    pathway_names: List[str] = field(
        default_factory=lambda: ["attention", "moe", "ssm"]
    )

    def get_pathway_weights(
        self, skill_type: SkillType
    ) -> Dict[str, float]:
        """Get normalised pathway weights for a skill type.

        Args:
            skill_type: The skill category.

        Returns:
            Dictionary of pathway → weight, normalised to sum to 1.
        """
        raw = self.rules.get(skill_type, self._uniform_weights())
        total = sum(raw.values()) + 1e-8
        return {k: v / total for k, v in raw.items()}

    def get_pathway_tensor(
        self, skill_type: SkillType, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Get pathway weights as a tensor for a skill type.

        Args:
            skill_type: The skill category.
            device: Torch device for the output tensor.

        Returns:
            Tensor of shape ``(num_pathways,)`` with normalised weights.
        """
        weights = self.get_pathway_weights(skill_type)
        return torch.tensor(
            [weights.get(p, 0.0) for p in self.pathway_names],
            dtype=torch.float32,
            device=device,
        )

    def get_routing_matrix(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Build the full routing matrix ``[num_skills, num_pathways]``.

        Returns:
            Tensor of shape ``(_NUM_SKILL_TYPES, num_pathways)``.
        """
        rows = []
        for skill in _SKILL_TYPE_LIST:
            rows.append(self.get_pathway_tensor(skill, device))
        return torch.stack(rows, dim=0)  # (S, P)

    def set_rule(
        self, skill_type: SkillType, pathway_weights: Dict[str, float]
    ) -> None:
        """Set or update the routing rule for a skill type.

        Args:
            skill_type: The skill category to update.
            pathway_weights: New pathway weight dict.
        """
        self.rules[skill_type] = pathway_weights

    def _uniform_weights(self) -> Dict[str, float]:
        """Return uniform pathway weights as a fallback."""
        n = len(self.pathway_names)
        return {p: 1.0 / n for p in self.pathway_names}

    def validate(self) -> List[str]:
        """Validate all routing rules and return any warnings.

        Returns:
            List of warning messages (empty if all rules are valid).
        """
        warnings: List[str] = []
        for skill, weights in self.rules.items():
            for pathway in self.pathway_names:
                if pathway not in weights:
                    warnings.append(
                        f"Skill {skill.value} is missing pathway '{pathway}'"
                    )
            for key in weights:
                if key not in self.pathway_names:
                    warnings.append(
                        f"Skill {skill.value} has unknown pathway '{key}'"
                    )
            total = sum(weights.values())
            if abs(total - 1.0) > 0.1:
                warnings.append(
                    f"Skill {skill.value} weights sum to {total:.3f}, "
                    f"expected ~1.0 (will be normalised)"
                )
            if any(v < 0 for v in weights.values()):
                warnings.append(
                    f"Skill {skill.value} has negative pathway weight"
                )
        return warnings


# ============================================================================
# SkillClassifier — Learned skill-type prediction
# ============================================================================

class SkillClassifier(nn.Module):
    """Classifies input tokens into skill types using a small MLP.

    The classifier produces per-token skill-type probabilities.
    It can be trained end-to-end with the rest of the model, or
    frozen after pre-training.  Manual overrides are supported for
    domain-specific routing — the caller can supply skill labels
    directly, bypassing the learned classifier.

    Architecture::

        logits = MLP(x)            # (B, S, num_skill_types)
        probs  = softmax(logits)   # (B, S, num_skill_types)

    Args:
        d_model: Model hidden dimension.
        bottleneck: MLP bottleneck width (default 128).
        num_skill_types: Number of skill categories (default 6).
        dropout: Dropout rate for the MLP (default 0.1).
    """

    def __init__(
        self,
        d_model: int,
        bottleneck: int = 128,
        num_skill_types: int = _NUM_SKILL_TYPES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_skill_types = num_skill_types

        self.mlp = nn.Sequential(
            nn.Linear(d_model, bottleneck, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, bottleneck, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, num_skill_types, bias=True),
        )

        # Initialise bias toward uniform distribution
        with torch.no_grad():
            self.mlp[-1].bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        override_skills: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify tokens into skill types.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            override_skills: Optional tensor ``[batch, seq_len]`` with
                integer skill-type indices.  When provided, the classifier
                output is replaced with one-hot vectors at these indices,
                bypassing the learned MLP.

        Returns:
            skill_logits: ``[batch, seq_len, num_skill_types]`` — raw logits.
            skill_probs: ``[batch, seq_len, num_skill_types]`` — softmax probs.
        """
        if override_skills is not None:
            # Use manual skill labels (discrete override)
            skill_probs = F.one_hot(
                override_skills.clamp(0, self.num_skill_types - 1),
                self.num_skill_types,
            ).float()
            skill_logits = torch.log(skill_probs + 1e-8)
        else:
            skill_logits = self.mlp(x)  # (B, S, num_skills)
            skill_probs = F.softmax(skill_logits, dim=-1)

        return skill_logits, skill_probs

    def predict_skill(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, SkillType]:
        """Predict the dominant skill type for the entire input.

        Aggregates per-token skill probabilities by mean and returns
        the dominant skill type.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.

        Returns:
            dominant_index: Scalar tensor with the dominant skill index.
            dominant_skill: The corresponding ``SkillType`` enum member.
        """
        with torch.no_grad():
            _, probs = self.forward(x)
            mean_probs = probs.mean(dim=(0, 1))  # (num_skills,)
            dominant_index = mean_probs.argmax()
            dominant_skill = _SKILL_TYPE_LIST[dominant_index.item()]
            return dominant_index, dominant_skill


# ============================================================================
# Routing Info
# ============================================================================

@dataclass
class SymbolicRoutingInfo:
    """Routing information returned by SymbolicMoERouter.

    Attributes:
        pathway_weights: ``[batch, seq_len, num_pathways]`` — final blended
            pathway allocation weights per token.
        skill_probs: ``[batch, seq_len, num_skill_types]`` — skill-type
            probabilities per token.
        dominant_skills: ``[batch, seq_len]`` — integer index of the
            dominant skill per token.
        skill_logits: ``[batch, seq_len, num_skill_types]`` — raw logits
            from the skill classifier.
        routing_rule_matrix: ``[num_skills, num_pathways]`` — the symbolic
            routing matrix used for this forward pass.
        mode: ``"soft"`` or ``"hard"`` — the routing mode used.
    """

    pathway_weights: torch.Tensor
    skill_probs: torch.Tensor
    dominant_skills: torch.Tensor
    skill_logits: torch.Tensor
    routing_rule_matrix: torch.Tensor
    mode: str


# ============================================================================
# SymbolicMoERouter — Combines skill classification + symbolic rules
# ============================================================================

class SymbolicMoERouter(nn.Module):
    """Symbolic MoE Router: skill classification → symbolic pathway routing.

    Unlike a learned token-level router, SymbolicMoERouter operates in
    two stages:

    1. **Skill Classification** — a learned MLP (or manual override)
       maps each token to a distribution over skill types.
    2. **Symbolic Routing** — a fixed (but customisable) routing matrix
       maps each skill type to pathway allocation weights.

    The final pathway weights for each token are a **soft blend** of the
    symbolic rules, weighted by the skill probabilities::

        pathway_weights[t] = Σ_s  skill_prob[t, s] × rule[s]

    In **hard** mode, the dominant skill type is selected and its rule
    is applied directly (no blending).

    This router can be combined with Losion's ``BiasRouter`` as a
    macro-level controller: symbolic rules constrain the pathway mix at
    a high level, while ``BiasRouter`` handles fine-grained load balancing
    within each pathway.

    Example
    -------
    >>> router = SymbolicMoERouter(d_model=768)
    >>> x = torch.randn(2, 16, 768)
    >>> pathway_weights, skill_probs, info = router(x)
    >>> pathway_weights.shape
    torch.Size([2, 16, 3])

    Args:
        d_model: Model hidden dimension.
        routing_rule: A :class:`SymbolicRoutingRule` instance.  If ``None``,
            the default rules are used.
        skill_classifier: An optional pre-built :class:`SkillClassifier`.
            If ``None``, a default one is created.
        routing_mode: ``"soft"`` (blended) or ``"hard"`` (discrete).
            Default ``"soft"``.
        temperature: Temperature for skill softmax (default ``1.0``).
            Lower values make skill classification sharper.
        classifier_bottleneck: Bottleneck width for the default classifier
            (default 128).
    """

    def __init__(
        self,
        d_model: int,
        routing_rule: Optional[SymbolicRoutingRule] = None,
        skill_classifier: Optional[SkillClassifier] = None,
        routing_mode: str = "soft",
        temperature: float = 1.0,
        classifier_bottleneck: int = 128,
    ) -> None:
        super().__init__()

        if routing_mode not in ("soft", "hard"):
            raise ValueError(
                f"routing_mode must be 'soft' or 'hard', got '{routing_mode}'"
            )

        self.d_model = d_model
        self.routing_mode = routing_mode
        self.temperature = temperature
        self.pathway_names = ["attention", "moe", "ssm"]
        self.num_pathways = len(self.pathway_names)

        # ---- Symbolic Routing Rule ----
        self.routing_rule = routing_rule or SymbolicRoutingRule()

        # ---- Skill Classifier ----
        self.skill_classifier = skill_classifier or SkillClassifier(
            d_model=d_model,
            bottleneck=classifier_bottleneck,
        )

        # ---- Routing Matrix (buffer, rebuilt when rules change) ----
        self.register_buffer(
            "routing_matrix",
            self.routing_rule.get_routing_matrix(),
        )  # (num_skills, num_pathways)

        # Temperature buffer
        self.register_buffer(
            "skill_temperature", torch.tensor(temperature)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        override_skills: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, SymbolicRoutingInfo]:
        """Compute pathway weights via skill classification + symbolic rules.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            override_skills: Optional ``[batch, seq_len]`` tensor of integer
                skill indices.  When provided, bypasses the learned
                classifier and uses these skill labels directly.

        Returns:
            pathway_weights: ``[batch, seq_len, num_pathways]`` — final
                pathway allocation weights per token.
            skill_probs: ``[batch, seq_len, num_skill_types]`` — skill-type
                probabilities per token.
            routing_info: :class:`SymbolicRoutingInfo` for monitoring.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input must be 3D [batch, seq, d_model], got {x.dim()}D"
            )

        batch_size, seq_len, _ = x.shape

        # ---- Stage 1: Skill Classification ----
        skill_logits, skill_probs = self.skill_classifier(x, override_skills)
        # skill_probs: (B, S, num_skills)

        # Apply temperature
        if self.skill_temperature.item() != 1.0:
            skill_probs = F.softmax(
                skill_logits / self.skill_temperature.clamp(min=0.1),
                dim=-1,
            )

        # ---- Stage 2: Symbolic Routing ----
        # routing_matrix: (num_skills, num_pathways)
        # Blend: pathway_weights[t] = Σ_s skill_prob[t,s] × rule[s, p]
        if self.routing_mode == "soft":
            pathway_weights = torch.matmul(
                skill_probs, self.routing_matrix
            )  # (B, S, P)
        else:
            # Hard: use dominant skill's rule directly
            dominant_skills = skill_probs.argmax(dim=-1)  # (B, S)
            pathway_weights = F.one_hot(
                dominant_skills, _NUM_SKILL_TYPES
            ).float() @ self.routing_matrix  # (B, S, P)

        # Ensure pathway weights are normalised (numerical stability)
        pathway_weights = pathway_weights / (
            pathway_weights.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Dominant skill indices
        dominant_skills = skill_probs.argmax(dim=-1)  # (B, S)

        # ---- Routing Info ----
        routing_info = SymbolicRoutingInfo(
            pathway_weights=pathway_weights,
            skill_probs=skill_probs,
            dominant_skills=dominant_skills,
            skill_logits=skill_logits,
            routing_rule_matrix=self.routing_matrix.clone(),
            mode=self.routing_mode,
        )

        return pathway_weights, skill_probs, routing_info

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_routing_mode(self, mode: str) -> None:
        """Set the routing mode.

        Args:
            mode: ``"soft"`` for blended routing, ``"hard"`` for discrete.
        """
        if mode not in ("soft", "hard"):
            raise ValueError(f"mode must be 'soft' or 'hard', got '{mode}'")
        self.routing_mode = mode

    def set_temperature(self, temperature: float) -> None:
        """Set the skill classification temperature.

        Lower values make the classification sharper (more confident).

        Args:
            temperature: New temperature value (must be > 0).
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be > 0, got {temperature}")
        self.skill_temperature.fill_(temperature)

    def update_routing_rule(
        self, skill_type: SkillType, pathway_weights: Dict[str, float]
    ) -> None:
        """Update a single routing rule and rebuild the routing matrix.

        Args:
            skill_type: The skill category to update.
            pathway_weights: New pathway weight dict.
        """
        self.routing_rule.set_rule(skill_type, pathway_weights)
        self.routing_matrix = self.routing_rule.get_routing_matrix(
            device=self.routing_matrix.device
        )

    def set_routing_rules(
        self, routing_rule: SymbolicRoutingRule
    ) -> None:
        """Replace the entire routing rule set.

        Args:
            routing_rule: New :class:`SymbolicRoutingRule` instance.
        """
        self.routing_rule = routing_rule
        self.routing_matrix = routing_rule.get_routing_matrix(
            device=self.routing_matrix.device
        )

    # ------------------------------------------------------------------
    # BiasRouter integration
    # ------------------------------------------------------------------

    def combine_with_bias_router(
        self,
        pathway_weights: torch.Tensor,
        bias_router_weights: torch.Tensor,
        alpha: float = 0.7,
    ) -> torch.Tensor:
        """Combine symbolic pathway weights with BiasRouter weights.

        This enables a two-level routing architecture:
          - **Macro level**: Symbolic-MoE determines the high-level
            pathway mix (attention vs. MoE vs. SSM).
          - **Micro level**: BiasRouter handles fine-grained load
            balancing within each pathway.

        The combination is a weighted average::

            final = α × symbolic + (1 - α) × bias_router

        Args:
            pathway_weights: Symbolic pathway weights
                ``[batch, seq_len, num_pathways]``.
            bias_router_weights: BiasRouter pathway weights
                ``[batch, seq_len, num_pathways]``.
            alpha: Weight for symbolic routing (default 0.7).
                ``alpha=1.0`` → pure symbolic, ``alpha=0.0`` → pure
                BiasRouter.

        Returns:
            Combined pathway weights ``[batch, seq_len, num_pathways]``,
            normalised to sum to 1.
        """
        combined = alpha * pathway_weights + (1 - alpha) * bias_router_weights
        # Re-normalise
        combined = combined / (combined.sum(dim=-1, keepdim=True) + 1e-8)
        return combined

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def get_skill_distribution(
        self, x: torch.Tensor
    ) -> Dict[str, float]:
        """Get the aggregate skill-type distribution for an input.

        Useful for monitoring and debugging routing behaviour.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.

        Returns:
            Dictionary mapping skill name → mean probability.
        """
        with torch.no_grad():
            _, skill_probs, _ = self.forward(x)
            mean_probs = skill_probs.mean(dim=(0, 1))  # (num_skills,)
            return {
                skill.value: mean_probs[i].item()
                for i, skill in enumerate(_SKILL_TYPE_LIST)
            }

    def get_pathway_distribution(
        self, x: torch.Tensor
    ) -> Dict[str, float]:
        """Get the aggregate pathway distribution for an input.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.

        Returns:
            Dictionary mapping pathway name → mean weight.
        """
        with torch.no_grad():
            pathway_weights, _, _ = self.forward(x)
            mean_weights = pathway_weights.mean(dim=(0, 1))  # (P,)
            return {
                self.pathway_names[i]: mean_weights[i].item()
                for i in range(self.num_pathways)
            }

    def get_routing_entropy(self, skill_probs: torch.Tensor) -> torch.Tensor:
        """Compute the entropy of the skill-type distribution.

        Low entropy → confident skill classification (one dominant skill).
        High entropy → uncertain classification (uniform skill probs).

        Args:
            skill_probs: ``[batch, seq_len, num_skill_types]``.

        Returns:
            Mean normalised entropy scalar in ``[0, 1]``.
        """
        with torch.no_grad():
            clamped = skill_probs.clamp(min=1e-8)
            entropy = -(clamped * clamped.log()).sum(dim=-1)  # (B, S)
            max_entropy = math.log(_NUM_SKILL_TYPES)
            return (entropy / max_entropy).mean()
