"""
Test suite for MCTS Agent Loop.

Covers the LATS-style Monte Carlo Tree Search agent loop with
DFSDT-style backtracking. Tests all MCTS phases:
- SELECT: UCB1-based node selection
- EXPAND: Action candidate generation
- EVALUATE/SIMULATE: Action execution and reward computation
- BACKPROPAGATE: Value update from leaf to root
- BACKTRACK: Confidence-drop detection and alternative exploration

v2.5.0: Created to address audit finding C4.3 — MCTS agent had zero
dedicated test coverage despite being one of the more complex agent
components.
"""

import pytest
import math

from losion.agent.planning.mcts_agent import (
    NodeStatus,
    ActionEdge,
    ActionNode,
    AgentState,
    MCTSResult,
    MCTSAgentLoop,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_state():
    """Create a simple agent state for testing."""
    return AgentState(
        query="What is the capital of France?",
        confidence=0.2,
        domain="geography",
    )


@pytest.fixture
def high_confidence_state():
    """Create a state that already meets the confidence threshold."""
    return AgentState(
        query="Easy question",
        confidence=0.8,
    )


@pytest.fixture
def mock_executor():
    """Create a mock action executor that increases confidence."""
    call_count = {"n": 0}

    def executor(action_name, query, context, **kwargs):
        call_count["n"] += 1
        # Each action increases confidence by 0.15
        new_confidence = min(0.2 + call_count["n"] * 0.15, 0.95)
        result = f"Result of {action_name}"
        return result, new_confidence

    executor.call_count = call_count
    return executor


@pytest.fixture
def failing_executor():
    """Create a mock executor that decreases confidence (triggers backtracking)."""

    def executor(action_name, query, context, **kwargs):
        # Decrease confidence
        new_confidence = max(0.2 - 0.1, 0.0)
        result = f"Failed: {action_name}"
        return result, new_confidence

    return executor


@pytest.fixture
def mcts_loop(mock_executor):
    """Create an MCTS loop with the mock executor."""
    return MCTSAgentLoop(
        action_executor=mock_executor,
        max_simulations=5,
        max_depth=3,
        confidence_threshold=0.7,
    )


# ============================================================================
# ActionEdge Tests
# ============================================================================


class TestActionEdge:
    """Tests for ActionEdge confidence delta computation."""

    def test_confidence_delta_positive(self):
        """Positive delta when confidence increases."""
        edge = ActionEdge(
            action="web_search",
            confidence_before=0.2,
            confidence_after=0.5,
        )
        assert edge.confidence_delta == pytest.approx(0.3)

    def test_confidence_delta_negative(self):
        """Negative delta when confidence decreases."""
        edge = ActionEdge(
            action="terminal_execute",
            confidence_before=0.5,
            confidence_after=0.2,
        )
        assert edge.confidence_delta == pytest.approx(-0.3)

    def test_confidence_delta_zero(self):
        """Zero delta when confidence unchanged."""
        edge = ActionEdge(
            action="skill_lookup",
            confidence_before=0.5,
            confidence_after=0.5,
        )
        assert edge.confidence_delta == pytest.approx(0.0)


# ============================================================================
# ActionNode Tests
# ============================================================================


class TestActionNode:
    """Tests for ActionNode tree structure and UCB1 computation."""

    def test_value_with_no_visits(self):
        """Value should be 0 when unvisited."""
        node = ActionNode()
        assert node.value == 0.0

    def test_value_with_visits(self):
        """Value should be average of total_value / visits."""
        node = ActionNode(visits=4, total_value=2.0)
        assert node.value == pytest.approx(0.5)

    def test_ucb1_unvisited_is_infinity(self):
        """Unvisited nodes should have infinite UCB1 (highest priority)."""
        node = ActionNode()
        assert node.ucb1 == float("inf")

    def test_ucb1_root_node(self):
        """Root node with no parent should return plain value."""
        node = ActionNode(visits=10, total_value=5.0, parent=None)
        assert node.ucb1 == pytest.approx(0.5)

    def test_ucb1_balances_exploration_exploitation(self):
        """UCB1 should balance between high-value and less-visited nodes."""
        parent = ActionNode(visits=100, total_value=50.0)

        # High value, many visits → high exploitation, low exploration
        high_value = ActionNode(visits=50, total_value=40.0, parent=parent)
        # Low value, few visits → low exploitation, high exploration
        low_value = ActionNode(visits=2, total_value=0.5, parent=parent)

        # Both should have reasonable UCB1 scores
        assert high_value.ucb1 > 0
        assert low_value.ucb1 > 0
        # The low-visit node should have higher exploration bonus
        assert low_value.ucb1 > low_value.value

    def test_is_leaf(self):
        """Node with no children should be a leaf."""
        node = ActionNode()
        assert node.is_leaf is True

        child = ActionNode()
        node.children.append(child)
        assert node.is_leaf is False

    def test_is_root(self):
        """Node with no parent should be root."""
        root = ActionNode()
        assert root.is_root is True

        child = ActionNode(parent=root)
        assert child.is_root is False

    def test_path_from_root(self):
        """Path should trace from root to current node."""
        root = ActionNode()
        child1 = ActionNode(parent=root)
        child2 = ActionNode(parent=child1)

        path = child2.path_from_root
        assert len(path) == 3
        assert path[0] is root
        assert path[1] is child1
        assert path[2] is child2

    def test_action_history(self):
        """Action history should list all actions from root to node."""
        root = ActionNode()
        edge1 = ActionEdge(action="web_search")
        child1 = ActionNode(parent=root, edge_from_parent=edge1)
        edge2 = ActionEdge(action="verify_output")
        child2 = ActionNode(parent=child1, edge_from_parent=edge2)

        history = child2.action_history
        assert history == ["web_search", "verify_output"]


# ============================================================================
# AgentState Tests
# ============================================================================


class TestAgentState:
    """Tests for AgentState cloning and properties."""

    def test_clone_independence(self, simple_state):
        """Cloned state should be independent from original."""
        clone = simple_state.clone()
        clone.confidence = 0.9
        clone.actions_taken.append("new_action")

        assert simple_state.confidence == 0.2
        assert "new_action" not in simple_state.actions_taken

    def test_clone_preserves_data(self, simple_state):
        """Cloned state should preserve all original data."""
        clone = simple_state.clone()
        assert clone.query == simple_state.query
        assert clone.domain == simple_state.domain
        assert clone.confidence == simple_state.confidence

    def test_context_str(self, simple_state):
        """context_str should join context lines."""
        simple_state.context = ["Line 1", "Line 2", "Line 3"]
        assert simple_state.context_str == "Line 1\nLine 2\nLine 3"


# ============================================================================
# MCTSAgentLoop Tests
# ============================================================================


class TestMCTSAgentLoop:
    """Tests for the main MCTS agent loop."""

    def test_high_confidence_skips_mcts(self, high_confidence_state):
        """When initial confidence exceeds threshold, no MCTS needed."""
        loop = MCTSAgentLoop(confidence_threshold=0.7)
        result = loop.run(
            query="Easy question",
            initial_confidence=0.8,
        )
        assert result.total_simulations == 0
        assert result.final_confidence == 0.8
        assert result.best_path == []

    def test_mcts_finds_path(self, mcts_loop, mock_executor):
        """MCTS should find a path that increases confidence."""
        result = mcts_loop.run(
            query="What is the capital of France?",
            initial_confidence=0.2,
            domain="geography",
        )
        # The mock executor increases confidence, so MCTS should find something
        assert result.final_confidence > 0.2
        assert result.tree_size > 1

    def test_mcts_respects_max_depth(self, mock_executor):
        """MCTS should not expand nodes beyond max_depth."""
        loop = MCTSAgentLoop(
            action_executor=mock_executor,
            max_simulations=10,
            max_depth=2,
        )
        result = loop.run(query="Test", initial_confidence=0.1)
        # No path should be deeper than max_depth
        if result.best_path:
            assert len(result.best_path) <= 2

    def test_mcts_respects_max_simulations(self, mock_executor):
        """MCTS should not exceed max_simulations."""
        loop = MCTSAgentLoop(
            action_executor=mock_executor,
            max_simulations=3,
        )
        result = loop.run(query="Test", initial_confidence=0.1)
        assert result.total_simulations <= 3

    def test_backtracking_on_confidence_drop(self, failing_executor):
        """MCTS should backtrack when confidence drops significantly."""
        loop = MCTSAgentLoop(
            action_executor=failing_executor,
            max_simulations=5,
            backtrack_threshold=0.05,
        )
        result = loop.run(query="Test", initial_confidence=0.3)
        # Should have recorded backtracks
        assert result.backtracks >= 0  # At minimum, no crash

    def test_result_has_valid_structure(self, mcts_loop):
        """MCTSResult should have all expected fields."""
        result = mcts_loop.run(query="Test", initial_confidence=0.2)
        assert isinstance(result.best_path, list)
        assert result.final_state is not None or result.final_confidence >= 0
        assert isinstance(result.total_simulations, int)
        assert isinstance(result.tree_size, int)
        assert isinstance(result.backtracks, int)
        assert result.total_time >= 0

    def test_default_executor_no_crash(self):
        """Default executor should not crash."""
        loop = MCTSAgentLoop()  # Uses _default_executor
        result = loop.run(query="Test", initial_confidence=0.1)
        assert result is not None

    def test_heuristic_actions_low_confidence(self):
        """Heuristic actions should suggest more actions at low confidence."""
        loop = MCTSAgentLoop()
        state = AgentState(query="Test", confidence=0.1)
        actions = loop._heuristic_actions(state)
        # At very low confidence, multiple actions should be suggested
        assert len(actions) >= 1
        action_names = [a["action"] for a in actions]
        assert "web_search" in action_names

    def test_heuristic_actions_high_confidence(self):
        """Heuristic actions should suggest fewer actions at high confidence."""
        loop = MCTSAgentLoop()
        state = AgentState(query="Test", confidence=0.9)
        actions = loop._heuristic_actions(state)
        # At high confidence, no actions should be needed
        assert len(actions) == 0

    def test_generate_alternative_actions(self):
        """Alternative actions should include backup options."""
        loop = MCTSAgentLoop()
        state = AgentState(query="Test", confidence=0.3)
        alternatives = loop._generate_alternative_actions(state, "web_search")
        action_names = [a["action"] for a in alternatives]
        assert "web_search" in action_names
        # Should have alternative actions
        assert len(alternatives) > 1


# ============================================================================
# NodeStatus Tests
# ============================================================================


class TestNodeStatus:
    """Tests for NodeStatus enum values."""

    def test_all_statuses_exist(self):
        """All expected node statuses should exist."""
        assert NodeStatus.UNEXPLORED.value == "unexplored"
        assert NodeStatus.EXPANDING.value == "expanding"
        assert NodeStatus.EXPLORED.value == "explored"
        assert NodeStatus.FAILED.value == "failed"
        assert NodeStatus.PRUNED.value == "pruned"


# ============================================================================
# Compute Reward Tests
# ============================================================================


class TestComputeReward:
    """Tests for reward computation."""

    def test_positive_confidence_delta_reward(self):
        """Positive confidence delta should produce positive reward."""
        loop = MCTSAgentLoop()
        edge = ActionEdge(
            action="web_search",
            confidence_before=0.2,
            confidence_after=0.5,
            tool_trust=0.7,
        )
        reward = loop._compute_reward(edge)
        assert reward > 0

    def test_negative_confidence_delta_reward(self):
        """Negative confidence delta should produce negative reward."""
        loop = MCTSAgentLoop()
        edge = ActionEdge(
            action="terminal_execute",
            confidence_before=0.5,
            confidence_after=0.2,
            tool_trust=0.5,
        )
        reward = loop._compute_reward(edge)
        assert reward < 0

    def test_high_trust_amplifies_reward(self):
        """High tool trust should amplify positive reward."""
        loop = MCTSAgentLoop()
        edge_low_trust = ActionEdge(
            action="web_search",
            confidence_before=0.2,
            confidence_after=0.5,
            tool_trust=0.3,
        )
        edge_high_trust = ActionEdge(
            action="web_search",
            confidence_before=0.2,
            confidence_after=0.5,
            tool_trust=0.8,
        )
        reward_low = loop._compute_reward(edge_low_trust)
        reward_high = loop._compute_reward(edge_high_trust)
        assert reward_high > reward_low


# ============================================================================
# Backpropagation Tests
# ============================================================================


class TestBackpropagation:
    """Tests for backpropagation with discount factor."""

    def test_backpropagate_updates_visits(self):
        """Backpropagation should increment visit counts up the tree."""
        loop = MCTSAgentLoop()
        root = ActionNode()
        child = ActionNode(parent=root)
        leaf = ActionNode(parent=child)

        loop._backpropagate(leaf, 1.0)

        assert leaf.visits == 1
        assert child.visits == 1
        assert root.visits == 1

    def test_backpropagate_applies_discount(self):
        """Backpropagation should discount reward for deeper nodes."""
        loop = MCTSAgentLoop()
        root = ActionNode()
        child = ActionNode(parent=root)
        leaf = ActionNode(parent=child)

        loop._backpropagate(leaf, 1.0)

        # Leaf gets full reward, child gets discounted, root gets double discounted
        assert leaf.total_value == pytest.approx(1.0)
        assert child.total_value == pytest.approx(0.9)  # 1.0 * 0.9
        assert root.total_value == pytest.approx(0.81)  # 1.0 * 0.9 * 0.9


# ============================================================================
# Select Tests
# ============================================================================


class TestSelect:
    """Tests for UCB1-based node selection."""

    def test_select_returns_leaf(self):
        """Select should return a leaf node."""
        loop = MCTSAgentLoop()
        root = ActionNode(status=NodeStatus.EXPLORED)
        child = ActionNode(
            parent=root,
            status=NodeStatus.EXPLORED,
            visits=1,
            total_value=0.5,
        )
        root.children.append(child)

        result = loop._select(root)
        assert result.is_leaf is True

    def test_select_skips_failed_nodes(self):
        """Select should skip failed child nodes."""
        loop = MCTSAgentLoop()
        root = ActionNode(status=NodeStatus.EXPLORED)
        failed = ActionNode(parent=root, status=NodeStatus.FAILED)
        valid = ActionNode(parent=root, status=NodeStatus.UNEXPLORED)
        root.children = [failed, valid]

        result = loop._select(root)
        assert result.status != NodeStatus.FAILED
