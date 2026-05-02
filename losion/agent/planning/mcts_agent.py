"""
LATS-style MCTS Agent Loop — Tree-structured action exploration with backtracking.

Inspired by:
- LATS (Zhou et al., ICML 2024): Language Agent Tree Search — unifies reasoning,
  acting, and planning in a single MCTS framework. Each node is a language state;
  each edge is an action. Uses LM self-evaluation as the value function.
- DFSDT (Qin et al., 2023): Depth-First Search Decision Tree from ToolLLM —
  when a tool path fails, backtrack and try alternatives instead of continuing.
- ExACT (2024): Reflective MCTS — combines reflection with tree search for
  improved agent decision-making.

This replaces the linear agent loop (iteration 1→2→3...) with a tree-structured
search where:
1. SELECT: Use UCB1 to select the most promising node to expand
2. EXPAND: Generate possible actions at the selected node
3. EVALUATE: Estimate the value of each action (using model confidence)
4. SIMULATE: Execute the best action and observe the result
5. BACKPROPAGATE: Update values up the tree
6. BACKTRACK: If an action fails, return to parent and try siblings

Key design:
    AgentState → ActionNode tree → MCTS search → Best action path

    Root: (query, empty_context, initial_confidence)
    Children: (action_taken, context_after, confidence_after)
    Value: confidence_after * tool_trust * experience_bonus
    Policy: SignalExtractor's priority scores

Integration with Losion:
    - Uses existing MCTSReasoner from losion.core.reasoning when available
    - SignalExtractor provides the expansion policy
    - CalibrationEngine provides tool trust for value estimation
    - EpisodicMemory provides experience-based priors
"""

from __future__ import annotations

import math
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of an action node in the MCTS tree."""

    UNEXPLORED = "unexplored"
    EXPANDING = "expanding"
    EXPLORED = "explored"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class ActionEdge:
    """An edge in the MCTS tree representing an agent action.

    Attributes:
        action: The agent action taken (e.g., "web_search").
        action_result: Result of the action.
        context_delta: Context added by this action.
        confidence_before: Confidence before the action.
        confidence_after: Confidence after the action.
        tool_trust: Trust score for the tool used.
        reward: Computed reward for this action.
    """

    action: str = ""
    action_result: Any = None
    context_delta: str = ""
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    tool_trust: float = 0.5
    reward: float = 0.0

    @property
    def confidence_delta(self) -> float:
        """Change in confidence from this action."""
        return self.confidence_after - self.confidence_before


@dataclass
class ActionNode:
    """A node in the MCTS agent tree.

    Each node represents a state in the agent's decision process.
    The tree structure enables backtracking: if a path fails,
    the agent can return to a parent and try alternative actions.

    Attributes:
        state: The agent state at this node.
        parent: Parent node (None for root).
        children: Child nodes explored from this state.
        edge_from_parent: The action edge that led to this node.
        visits: Number of times this node has been visited.
        total_value: Cumulative value from backpropagation.
        status: Current status of this node.
        depth: Depth in the tree (0 for root).
    """

    state: Any = None  # AgentState
    parent: Optional["ActionNode"] = None
    children: List["ActionNode"] = field(default_factory=list)
    edge_from_parent: Optional[ActionEdge] = None
    visits: int = 0
    total_value: float = 0.0
    status: NodeStatus = NodeStatus.UNEXPLORED
    depth: int = 0

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def ucb1(self) -> float:
        """UCB1 score for selection (exploration-exploitation balance).

        UCB1 = Q(s,a) + C * sqrt(ln(N(parent)) / N(s,a))

        Where C is the exploration constant. Higher C = more exploration.
        """
        if self.visits == 0:
            return float("inf")  # Unvisited nodes get highest priority

        if self.parent is None or self.parent.visits == 0:
            return self.value

        exploration_constant = 1.414  # sqrt(2), standard UCB1
        exploitation = self.value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    @property
    def is_leaf(self) -> bool:
        """Whether this node has no children."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Whether this is the root node."""
        return self.parent is None

    @property
    def path_from_root(self) -> List["ActionNode"]:
        """Get the path from root to this node."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    @property
    def action_history(self) -> List[str]:
        """Get the sequence of actions from root to this node."""
        path = self.path_from_root
        actions = []
        for node in path:
            if node.edge_from_parent is not None:
                actions.append(node.edge_from_parent.action)
        return actions


@dataclass
class AgentState:
    """State of the agent at a given point in the MCTS tree.

    This captures everything the agent knows at a decision point:
    the query, accumulated context, current confidence, and actions taken.

    Attributes:
        query: The original user query.
        context: Accumulated context from previous actions.
        confidence: Current model confidence.
        actions_taken: List of actions taken so far.
        domain: Domain classification.
        thinking_mode: Whether model is in thinking mode.
        routing_weights: Tri-Jalur routing weights.
        metadata: Additional state metadata.
    """

    query: str = ""
    context: List[str] = field(default_factory=list)
    confidence: float = 0.5
    actions_taken: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    thinking_mode: Optional[str] = None
    routing_weights: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def context_str(self) -> str:
        """Concatenated context string."""
        return "\n".join(self.context)

    def clone(self) -> "AgentState":
        """Create a copy of this state for branching."""
        return AgentState(
            query=self.query,
            context=list(self.context),
            confidence=self.confidence,
            actions_taken=list(self.actions_taken),
            domain=self.domain,
            thinking_mode=self.thinking_mode,
            routing_weights=list(self.routing_weights) if self.routing_weights else None,
            metadata=dict(self.metadata),
        )


@dataclass
class MCTSResult:
    """Result of an MCTS agent loop execution.

    Attributes:
        best_path: The best action path found by MCTS.
        final_state: The agent state after the best path.
        total_simulations: Number of MCTS simulations run.
        tree_size: Total number of nodes in the tree.
        backtracks: Number of backtracks performed.
        final_confidence: Confidence at the end of the best path.
        all_paths_explored: All paths that were explored (for debugging).
        total_time: Total execution time.
    """

    best_path: List[str] = field(default_factory=list)
    final_state: Optional[AgentState] = None
    total_simulations: int = 0
    tree_size: int = 0
    backtracks: int = 0
    final_confidence: float = 0.0
    all_paths_explored: List[List[str]] = field(default_factory=list)
    total_time: float = 0.0


class MCTSAgentLoop:
    """LATS-style MCTS agent loop for tree-structured action exploration.

    This replaces the linear agent loop with a tree search that:
    1. Explores multiple possible action sequences
    2. Backtracks when actions fail (DFSDT-style)
    3. Uses UCB1 for balanced exploration-exploitation
    4. Propagates confidence changes as reward signals

    The key insight from LATS: "By unifying reasoning, acting, and planning
    in a single MCTS framework, the agent can deliberate about WHEN to use
    tools, WHICH tools to use, and WHAT to do when tools fail."

    Usage:
        loop = MCTSAgentLoop(
            action_executor=my_execute_fn,
            signal_extractor=my_extractor,
            max_simulations=10,
        )
        result = loop.run(
            query="What is the capital of France?",
            initial_confidence=0.2,
        )

    Args:
        action_executor: Function(action_name, query, context) → (result, new_confidence).
        signal_extractor: SignalExtractor for generating action candidates.
        max_simulations: Maximum MCTS simulations per query.
        max_depth: Maximum tree depth.
        confidence_threshold: Stop when confidence exceeds this.
        backtrack_threshold: Backtrack if confidence drops by this much.
        exploration_constant: UCB1 exploration parameter.
    """

    def __init__(
        self,
        action_executor: Optional[Callable] = None,
        signal_extractor: Optional[Any] = None,
        max_simulations: int = 8,
        max_depth: int = 5,
        confidence_threshold: float = 0.7,
        backtrack_threshold: float = 0.15,
        exploration_constant: float = 1.414,
    ) -> None:
        self.action_executor = action_executor or self._default_executor
        self.signal_extractor = signal_extractor
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold
        self.backtrack_threshold = backtrack_threshold
        self.exploration_constant = exploration_constant

    def run(
        self,
        query: str,
        initial_confidence: float = 0.5,
        initial_context: Optional[List[str]] = None,
        domain: Optional[str] = None,
        thinking_mode: Optional[str] = None,
        routing_weights: Optional[List[float]] = None,
    ) -> MCTSResult:
        """Run the MCTS agent loop.

        Executes the full MCTS cycle: select → expand → evaluate →
        simulate → backpropagate, with DFSDT-style backtracking.

        Args:
            query: The user's query.
            initial_confidence: Starting confidence from the model.
            initial_context: Pre-existing context.
            domain: Domain classification.
            thinking_mode: Model's thinking mode.
            routing_weights: Tri-Jalur routing weights.

        Returns:
            MCTSResult with the best action path found.
        """
        start_time = time.time()

        # Initialize root state
        root_state = AgentState(
            query=query,
            context=list(initial_context) if initial_context else [],
            confidence=initial_confidence,
            domain=domain,
            thinking_mode=thinking_mode,
            routing_weights=routing_weights,
        )
        root = ActionNode(state=root_state, status=NodeStatus.EXPLORED)
        backtracks = 0

        # Check if we even need MCTS
        if initial_confidence >= self.confidence_threshold:
            return MCTSResult(
                best_path=[],
                final_state=root_state,
                total_simulations=0,
                tree_size=1,
                backtracks=0,
                final_confidence=initial_confidence,
                total_time=time.time() - start_time,
            )

        # MCTS simulations
        for sim in range(self.max_simulations):
            # === SELECT: Find the most promising leaf to expand ===
            leaf = self._select(root)

            if leaf is None or leaf.depth >= self.max_depth:
                continue

            # === EXPAND: Generate possible actions at the leaf ===
            children = self._expand(leaf)

            if not children:
                leaf.status = NodeStatus.FAILED
                continue

            # === EVALUATE & SIMULATE: Execute the most promising child ===
            for child in children:
                if child.status == NodeStatus.EXPLORED:
                    # This child was already simulated during expand
                    continue

                # Simulate: execute the action
                reward = self._simulate(child)

                # === BACKTRACK check (DFSDT-style) ===
                if (child.edge_from_parent is not None and
                        child.edge_from_parent.confidence_delta < -self.backtrack_threshold):
                    # Action hurt confidence → mark as failed, backtrack
                    child.status = NodeStatus.FAILED
                    backtracks += 1
                    logger.debug(
                        f"DFSDT backtrack: {child.edge_from_parent.action} "
                        f"reduced confidence by {abs(child.edge_from_parent.confidence_delta):.3f}"
                    )
                    continue

                # === BACKPROPAGATE: Update values up the tree ===
                self._backpropagate(child, reward)

                # Check if we found a good enough path
                if (child.state is not None and
                        child.state.confidence >= self.confidence_threshold):
                    # Found a path that achieves sufficient confidence
                    best_path = child.action_history
                    return MCTSResult(
                        best_path=best_path,
                        final_state=child.state,
                        total_simulations=sim + 1,
                        tree_size=self._count_nodes(root),
                        backtracks=backtracks,
                        final_confidence=child.state.confidence,
                        total_time=time.time() - start_time,
                    )

        # === Select best path from the tree ===
        best_leaf = self._find_best_leaf(root)
        best_path = best_leaf.action_history if best_leaf else []

        return MCTSResult(
            best_path=best_path,
            final_state=best_leaf.state if best_leaf else root_state,
            total_simulations=self.max_simulations,
            tree_size=self._count_nodes(root),
            backtracks=backtracks,
            final_confidence=best_leaf.state.confidence if best_leaf and best_leaf.state else initial_confidence,
            total_time=time.time() - start_time,
        )

    def _select(self, node: ActionNode) -> Optional[ActionNode]:
        """SELECT phase: Use UCB1 to find the most promising leaf.

        Traverses the tree from the given node, selecting the child
        with the highest UCB1 score at each level, until reaching
        a leaf node.
        """
        current = node

        while not current.is_leaf:
            # Filter out failed and pruned children
            valid_children = [
                c for c in current.children
                if c.status not in (NodeStatus.FAILED, NodeStatus.PRUNED)
            ]

            if not valid_children:
                # All children failed — this node is a dead end
                current.status = NodeStatus.FAILED
                return current.parent  # Backtrack to parent

            # Select child with highest UCB1
            best_child = max(valid_children, key=lambda c: c.ucb1)
            current = best_child

        return current

    def _expand(self, node: ActionNode) -> List[ActionNode]:
        """EXPAND phase: Generate possible actions at a leaf node.

        Uses SignalExtractor to determine candidate actions, then
        creates child nodes for each candidate.
        """
        if node.state is None:
            return []

        # Get candidate actions from signal extractor
        candidate_actions = self._get_candidate_actions(node.state)

        if not candidate_actions:
            return []

        children = []
        for action_info in candidate_actions:
            action_name = action_info.get("action", "")
            priority = action_info.get("priority", 0.0)

            # Create child state
            child_state = node.state.clone()
            child_state.actions_taken.append(action_name)

            # Create edge
            edge = ActionEdge(
                action=action_name,
                confidence_before=node.state.confidence,
                tool_trust=action_info.get("tool_trust", 0.5),
            )

            # Create child node
            child = ActionNode(
                state=child_state,
                parent=node,
                edge_from_parent=edge,
                depth=node.depth + 1,
                status=NodeStatus.UNEXPLORED,
            )

            node.children.append(child)
            children.append(child)

        node.status = NodeStatus.EXPANDING
        return children

    def _simulate(self, node: ActionNode) -> float:
        """SIMULATE phase: Execute the action and compute reward.

        Executes the action via the action_executor, updates the
        node's state with the result, and computes a reward signal.
        """
        if node.state is None or node.edge_from_parent is None:
            return 0.0

        action = node.edge_from_parent.action
        state = node.state

        try:
            # Execute action
            result, new_confidence = self.action_executor(
                action_name=action,
                query=state.query,
                context=state.context,
                domain=state.domain,
            )

            # Update state
            node.state.confidence = new_confidence

            # Update edge
            node.edge_from_parent.confidence_after = new_confidence
            node.edge_from_parent.action_result = result

            if result is not None:
                if isinstance(result, str):
                    node.state.context.append(result)
                    node.edge_from_parent.context_delta = result
                elif isinstance(result, list):
                    for item in result:
                        node.state.context.append(str(item))
                elif isinstance(result, dict):
                    context_str = str(result)
                    node.state.context.append(context_str)
                    node.edge_from_parent.context_delta = context_str

            # Compute reward
            reward = self._compute_reward(node.edge_from_parent)
            node.edge_from_parent.reward = reward

            # Mark as explored
            node.status = NodeStatus.EXPLORED

            return reward

        except Exception as e:
            logger.warning(f"Action simulation failed for {action}: {e}")
            node.status = NodeStatus.FAILED
            node.edge_from_parent.confidence_after = max(0.0, state.confidence - 0.1)
            return -0.1  # Penalty for failed action

    def _backpropagate(self, node: ActionNode, reward: float) -> None:
        """BACKPROPAGATE phase: Update values from leaf to root.

        Propagates the reward up the tree, updating visit counts
        and total values. Applies discount factor for deeper nodes.
        """
        current = node
        discount = 1.0
        discount_factor = 0.9  # Gamma for MCTS discount

        while current is not None:
            current.visits += 1
            current.total_value += reward * discount
            discount *= discount_factor
            current = current.parent

    def _compute_reward(self, edge: ActionEdge) -> float:
        """Compute reward for an action edge.

        Reward = confidence_delta * tool_trust * experience_bonus

        The reward encourages:
        - Actions that increase confidence (positive delta)
        - Trusted tools (high tool trust score)
        - Novel actions (not repeating the same action)
        """
        confidence_delta = edge.confidence_delta

        # Base reward: confidence improvement
        reward = confidence_delta

        # Tool trust bonus: trusted tools get higher reward
        if edge.tool_trust > 0.5:
            reward *= 1.0 + (edge.tool_trust - 0.5) * 0.5
        elif edge.tool_trust < 0.5:
            reward *= 0.5 + edge.tool_trust

        # Novelty bonus: penalize repeated actions
        # (checked by the caller via state.actions_taken)

        return reward

    def _get_candidate_actions(self, state: AgentState) -> List[Dict[str, Any]]:
        """Get candidate actions for a state.

        Uses SignalExtractor if available, otherwise falls back to
        a simple confidence-based heuristic.
        """
        if self.signal_extractor is not None:
            try:
                signal = self.signal_extractor.extract(
                    model_output=None,
                    confidence=state.confidence,
                    query_text=state.query,
                )
                if signal.needs_intervention:
                    return [{
                        "action": signal.action.value,
                        "priority": signal.priority,
                        "tool_trust": signal.tool_trust,
                    }]

                # Also generate alternative actions based on domain
                alternatives = self._generate_alternative_actions(state, signal.action.value)
                return alternatives
            except Exception:
                pass

        # Fallback: simple heuristic
        return self._heuristic_actions(state)

    def _generate_alternative_actions(
        self, state: AgentState, primary_action: str
    ) -> List[Dict[str, Any]]:
        """Generate alternative actions in case the primary fails.

        This is the DFSDT component: always have backup actions ready
        so that backtracking can try alternatives.
        """
        actions = [{"action": primary_action, "priority": 1.0, "tool_trust": 0.5}]

        # Add alternatives based on domain and confidence
        all_possible = ["web_search", "skill_lookup", "tool_search", "terminal_execute", "verify_output"]

        for alt in all_possible:
            if alt != primary_action and alt not in state.actions_taken:
                actions.append({
                    "action": alt,
                    "priority": 0.5,  # Lower priority than primary
                    "tool_trust": 0.3,  # Lower trust for untried alternatives
                })

        return actions

    def _heuristic_actions(self, state: AgentState) -> List[Dict[str, Any]]:
        """Simple heuristic action generation when no signal extractor is available."""
        actions = []
        confidence = state.confidence

        if confidence < 0.3:
            actions.append({"action": "web_search", "priority": 1.0, "tool_trust": 0.5})
        if confidence < 0.4 and state.domain:
            actions.append({"action": "skill_lookup", "priority": 0.8, "tool_trust": 0.5})
        if confidence < 0.35:
            actions.append({"action": "tool_search", "priority": 0.7, "tool_trust": 0.5})
        if confidence < 0.25:
            actions.append({"action": "terminal_execute", "priority": 0.4, "tool_trust": 0.3})

        return actions

    def _find_best_leaf(self, root: ActionNode) -> Optional[ActionNode]:
        """Find the leaf with the highest value (best action path)."""
        best = None
        best_value = float("-inf")

        stack = [root]
        while stack:
            node = stack.pop()
            if node.state is not None and node.visits > 0:
                node_value = node.value
                # Prefer deeper nodes (more actions taken) with high value
                depth_bonus = node.depth * 0.05
                adjusted_value = node_value + depth_bonus

                if adjusted_value > best_value and node.status != NodeStatus.FAILED:
                    best_value = adjusted_value
                    best = node

            stack.extend(node.children)

        return best

    def _count_nodes(self, root: ActionNode) -> int:
        """Count total nodes in the tree."""
        count = 0
        stack = [root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)
        return count

    @staticmethod
    def _default_executor(
        action_name: str, query: str, context: List[str], **kwargs
    ) -> Tuple[Any, float]:
        """Default action executor (no-op, returns unchanged confidence)."""
        return None, 0.5
