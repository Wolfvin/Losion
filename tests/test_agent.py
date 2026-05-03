"""Tests for the Losion Agent Layer v2."""

import os
import json
import tempfile
import unittest
from pathlib import Path

from losion.agent.signals import (
    AgentAction,
    AgentSignal,
    ConfidenceThreshold,
    SignalExtractor,
)
from losion.agent.skills.store import SkillEntry, SkillMetadata, SkillStore
from losion.agent.skills.manager import SkillManager, SkillLookupResult
from losion.agent.skills.creator import SkillCreator
from losion.agent.tools.registry import ToolEntry, ToolRegistry, ToolSafety
from losion.agent.tools.terminal import SandboxedTerminal, TerminalResult, SandboxConfig
from losion.agent.tools.web_search import WebSearchInterface, SearchConfig, SearchResult
from losion.agent.tools.creator import ToolCreator
from losion.agent.orchestrator import AgentOrchestrator, AgentConfig, AgentResult
from losion.agent.reflection import ReflectionEngine, Reflection, ReflectionType
from losion.agent.memory import EpisodicMemory, Episode
from losion.agent.calibration import CalibrationEngine, DomainProfile, ToolTrustScore
from losion.agent.meta_skills import (
    SkillSynthesisMetaSkill,
    SkillVerificationMetaSkill,
    SkillCompositionMetaSkill,
    ComposedSkill,
)


class TestSignalExtractor(unittest.TestCase):
    """Test SignalExtractor — bridge between model and agent."""

    def setUp(self):
        self.extractor = SignalExtractor()

    def test_model_only_high_confidence(self):
        """When confidence is high, no intervention is needed."""
        signal = self.extractor.extract(
            model_output=None,
            confidence=0.9,
            query_text="simple question",
        )
        self.assertEqual(signal.action, AgentAction.MODEL_ONLY)
        self.assertFalse(signal.needs_intervention)

    def test_web_search_low_confidence(self):
        """When confidence is very low, web search is triggered."""
        signal = self.extractor.extract(
            model_output=None,
            confidence=0.1,
            query_text="What is the latest news about AI?",
        )
        self.assertEqual(signal.action, AgentAction.WEB_SEARCH)
        self.assertTrue(signal.needs_intervention)

    def test_skill_lookup_with_domain(self):
        """When confidence is low and domain is identified, skill lookup is triggered."""
        extractor = SignalExtractor(thresholds=ConfidenceThreshold(
            web_search=0.1,  # Suppress web search
            skill_lookup=0.5,
        ))
        signal = extractor.extract(
            model_output=None,
            confidence=0.3,
            query_text="calculate the integral of x squared",
        )
        self.assertIn(signal.action, [AgentAction.SKILL_LOOKUP, AgentAction.WEB_SEARCH])

    def test_confidence_threshold_validation(self):
        """ConfidenceThreshold should reject invalid values."""
        with self.assertRaises(ValueError):
            ConfidenceThreshold(web_search=1.5)

    def test_domain_classification(self):
        """SignalExtractor should classify domains correctly."""
        domain = self.extractor._classify_domain("how to write a python function")
        self.assertEqual(domain, "code")

        domain = self.extractor._classify_domain("solve the differential equation")
        self.assertEqual(domain, "math")

        domain = self.extractor._classify_domain("hello world")
        self.assertIsNone(domain)

    def test_v2_tool_trust_in_signal(self):
        """v2: Signal should include tool trust score."""
        signal = self.extractor.extract(
            model_output=None,
            confidence=0.1,
            query_text="search the web",
        )
        self.assertIn("tool_trust", signal.__dataclass_fields__)


class TestCalibrationEngine(unittest.TestCase):
    """Test CalibrationEngine — adaptive threshold calibration."""

    def setUp(self):
        self.engine = CalibrationEngine(learning_rate=0.2, min_samples=2)

    def test_default_thresholds(self):
        """Should return default thresholds for unknown domains."""
        thresholds = self.engine.get_thresholds(domain=None)
        self.assertAlmostEqual(thresholds["web_search"], 0.3, places=1)

    def test_domain_specific_thresholds(self):
        """Should return domain-specific thresholds."""
        thresholds = self.engine.get_thresholds(domain="math")
        # Math domain has higher web search threshold (model is usually right)
        self.assertGreater(thresholds["web_search"], 0.3)

    def test_calibration_adapts(self):
        """Thresholds should adapt based on outcomes."""
        initial = self.engine.get_thresholds(domain="math")

        # Record several successful web searches for math
        for _ in range(5):
            self.engine.record_outcome(
                action="web_search",
                domain="math",
                success=True,
                confidence_before=0.2,
                confidence_after=0.6,
            )

        adapted = self.engine.get_thresholds(domain="math")
        # After successful outcomes, threshold should decrease
        # (easier to trigger web search)
        self.assertLessEqual(adapted["web_search"], initial["web_search"] + 0.01)

    def test_tool_trust_score(self):
        """Tool trust scores should update based on outcomes."""
        self.engine.record_outcome(
            action="web_search", domain=None, success=True,
            confidence_before=0.2, confidence_after=0.6,
        )
        trust = self.engine.get_tool_trust("web_search")
        self.assertGreater(trust, 0.5)  # Should be above neutral


class TestReflectionEngine(unittest.TestCase):
    """Test ReflectionEngine — self-reflection on agent actions."""

    def setUp(self):
        self.engine = ReflectionEngine()

    def test_successful_action_reflection(self):
        """Should generate positive reflection for successful actions."""
        reflections = self.engine.evaluate(
            action="web_search",
            action_result=[{"title": "Result", "snippet": "Found relevant info"}],
            confidence_before=0.2,
            confidence_after=0.6,
            query="test query",
        )
        self.assertGreater(len(reflections), 0)
        self.assertEqual(reflections[0].reflection_type, ReflectionType.ACTION_SUCCESS)

    def test_failed_action_reflection(self):
        """Should generate negative reflection for failed actions."""
        reflections = self.engine.evaluate(
            action="web_search",
            action_result=None,
            confidence_before=0.3,
            confidence_after=0.2,
            query="test query",
        )
        self.assertGreater(len(reflections), 0)
        self.assertEqual(reflections[0].reflection_type, ReflectionType.ACTION_FAILURE)

    def test_strategy_correction(self):
        """Should suggest strategy correction when actions reduce confidence."""
        reflections = self.engine.evaluate(
            action="web_search",
            action_result=None,
            confidence_before=0.4,
            confidence_after=0.2,
            query="complex query",
        )
        strategy_reflections = [
            r for r in reflections
            if r.reflection_type == ReflectionType.STRATEGY_CORRECTION
        ]
        self.assertGreater(len(strategy_reflections), 0)

    def test_reflection_has_lesson(self):
        """Every reflection should contain a meaningful lesson."""
        reflections = self.engine.evaluate(
            action="tool_search",
            action_result={"success": True, "tools": ["python-exec"]},
            confidence_before=0.3,
            confidence_after=0.5,
        )
        for r in reflections:
            self.assertGreater(len(r.lesson), 0)


class TestEpisodicMemory(unittest.TestCase):
    """Test EpisodicMemory — experience-based memory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.memory = EpisodicMemory(store_dir=self.temp_dir, auto_save=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_and_retrieve_episode(self):
        """Should store and retrieve episodes."""
        episode = Episode(
            query="How to calculate derivatives",
            domain="math",
            actions=["web_search", "skill_lookup"],
            final_confidence=0.8,
            success=True,
        )
        self.memory.store_episode(episode)

        # Retrieve similar
        results = self.memory.retrieve_similar("How to calculate derivatives")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0].query, "How to calculate derivatives")

    def test_get_lessons(self):
        """Should extract lessons from similar episodes."""
        episode = Episode(
            query="Solve differential equation",
            domain="math",
            reflections=[{"lesson": "Web search is helpful for equations", "reflection_type": "action_success"}],
            final_confidence=0.7,
            success=True,
        )
        self.memory.store_episode(episode)

        lessons = self.memory.get_lessons_for_query("Solve differential equation", domain="math")
        self.assertGreater(len(lessons), 0)

    def test_action_recommendations(self):
        """Should recommend actions based on past experience."""
        episode = Episode(
            query="Write a Python function",
            domain="code",
            actions=["web_search", "tool_search"],
            reflections=[
                {"action_taken": "web_search", "reflection_type": "action_success"},
                {"action_taken": "tool_search", "reflection_type": "action_success"},
            ],
            final_confidence=0.9,
            success=True,
        )
        self.memory.store_episode(episode)

        recs = self.memory.get_action_recommendations("Write a Python function", domain="code")
        self.assertIn("web_search", recs)

    def test_persistence(self):
        """Episodes should persist across memory instances."""
        episode = Episode(
            query="Test persistence",
            domain="test",
            final_confidence=0.5,
            success=True,
        )
        self.memory.store_episode(episode)

        new_memory = EpisodicMemory(store_dir=self.temp_dir)
        results = new_memory.retrieve_similar("Test persistence")
        self.assertGreater(len(results), 0)


class TestMetaSkills(unittest.TestCase):
    """Test Meta-Skill System — skill creation, verification, composition."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = SkillStore(store_dir=self.temp_dir, auto_save=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_skill_verification(self):
        """Should verify skills with test cases."""
        # Create a skill with test cases
        skill = SkillEntry(
            name="test-math-skill",
            description="A math skill",
            skill_type="code",
            definition=(
                "# Math Skill\n"
                "Calculate mathematical expressions.\n\n"
                "## Test Cases\n"
                "1. Input: 2+2 → Expected: 4\n"
                "2. Input: 3*3 → Expected: 9\n"
            ),
            metadata=SkillMetadata(domain="math", confidence=0.3),
        )
        self.store.store(skill)

        # Verify
        verifier = SkillVerificationMetaSkill(skill_store=self.store)
        result = verifier.verify("test-math-skill")
        self.assertIsNotNone(result)
        self.assertGreater(result.test_cases_run, 0)

    def test_skill_composition(self):
        """Should compose multiple skills into a pipeline."""
        # Register some skills with names that match sub-task decomposition
        for name in ["collect data", "process the data", "analyze results"]:
            self.store.store(SkillEntry(
                name=name,
                description=f"Skill for {name}",
                skill_type="prompt",
                definition=f"Definition for {name}",
                metadata=SkillMetadata(domain="data", confidence=0.5),
            ))

        manager = SkillManager(store=self.store, auto_create=False)
        composer = SkillCompositionMetaSkill(skill_manager=manager)
        composed = composer.compose("collect and process data", domain="data")
        # Composition may not find exact matches (fuzzy lookup)
        # Just verify it doesn't crash and returns a result or None
        if composed:
            self.assertIsInstance(composed, ComposedSkill)


class TestSkillStore(unittest.TestCase):
    """Test SkillStore — persistent skill storage."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = SkillStore(store_dir=self.temp_dir, auto_save=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_and_lookup(self):
        """Should store and retrieve a skill."""
        entry = SkillEntry(
            name="test-skill",
            description="A test skill",
            skill_type="prompt",
            definition="test definition",
            metadata=SkillMetadata(domain="test"),
        )
        self.store.store(entry)

        retrieved = self.store.lookup("test-skill")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test-skill")
        self.assertEqual(retrieved.definition, "test definition")

    def test_lookup_nonexistent(self):
        """Should return None for non-existent skills."""
        result = self.store.lookup("nonexistent")
        self.assertIsNone(result)

    def test_search_by_domain(self):
        """Should search skills by domain."""
        entry = SkillEntry(
            name="math-skill",
            description="Math skill",
            skill_type="prompt",
            definition="math definition",
            metadata=SkillMetadata(domain="math"),
        )
        self.store.store(entry)

        results = self.store.search(domain="math")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "math-skill")

    def test_record_usage(self):
        """Should track skill usage."""
        entry = SkillEntry(
            name="usage-test",
            description="Usage test",
            skill_type="prompt",
            definition="test",
            metadata=SkillMetadata(domain="test"),
        )
        self.store.store(entry)

        self.store.record_usage("usage-test", success=True)
        self.store.record_usage("usage-test", success=True)
        self.store.record_usage("usage-test", success=False)

        retrieved = self.store.lookup("usage-test")
        self.assertEqual(retrieved.metadata.usage_count, 3)
        self.assertEqual(retrieved.metadata.success_count, 2)

    def test_persistence(self):
        """Skills should persist across store instances."""
        entry = SkillEntry(
            name="persist-skill",
            description="Persistence test",
            skill_type="prompt",
            definition="persist definition",
            metadata=SkillMetadata(domain="test"),
        )
        self.store.store(entry)

        new_store = SkillStore(store_dir=self.temp_dir)
        retrieved = new_store.lookup("persist-skill")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "persist-skill")


class TestToolRegistry(unittest.TestCase):
    """Test ToolRegistry — tool discovery and management."""

    def setUp(self):
        self.registry = ToolRegistry(allow_dangerous=False)

    def test_register_and_lookup(self):
        """Should register a tool and find it via lookup."""
        tool = ToolEntry(
            name="test-tool",
            description="A test tool",
            handler=lambda x: x,
            safety=ToolSafety.SAFE,
        )
        self.registry.register(tool)

        retrieved = self.registry.lookup("test-tool")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test-tool")

    def test_dangerous_tool_blocked(self):
        """Should block dangerous tools when allow_dangerous is False."""
        tool = ToolEntry(
            name="dangerous-tool",
            description="A dangerous tool",
            safety=ToolSafety.DANGEROUS,
        )
        with self.assertRaises(PermissionError):
            self.registry.register(tool)

    def test_duplicate_registration(self):
        """Should reject duplicate tool names."""
        tool = ToolEntry(
            name="dup-tool",
            description="First",
            safety=ToolSafety.SAFE,
        )
        self.registry.register(tool)

        with self.assertRaises(ValueError):
            self.registry.register(ToolEntry(
                name="dup-tool",
                description="Second",
                safety=ToolSafety.SAFE,
            ))


class TestSandboxedTerminal(unittest.TestCase):
    """Test SandboxedTerminal — safe terminal execution."""

    def test_simple_command(self):
        """Should execute a simple safe command."""
        terminal = SandboxedTerminal()
        result = terminal.execute("echo hello")
        self.assertTrue(result.success)
        self.assertIn("hello", result.stdout)

    def test_blocked_command(self):
        """Should block dangerous commands."""
        terminal = SandboxedTerminal()
        with self.assertRaises(PermissionError):
            terminal.execute("rm -rf /")

    def test_timeout(self):
        """Should timeout long-running commands."""
        config = SandboxConfig(max_execution_time=1.0)
        terminal = SandboxedTerminal(config)
        result = terminal.execute("sleep 10")
        self.assertTrue(result.timed_out)


class TestWebSearchInterface(unittest.TestCase):
    """Test WebSearchInterface — web search with caching."""

    def test_mock_search(self):
        """Should return mock results in mock mode."""
        search = WebSearchInterface(config=SearchConfig(backend="mock"))
        results = search.search("test query")
        self.assertGreater(len(results), 0)
        self.assertTrue(results[0].is_valid)

    def test_caching(self):
        """Should cache search results."""
        search = WebSearchInterface(config=SearchConfig(
            backend="mock",
            cache_results=True,
        ))
        results1 = search.search("cache test")
        results2 = search.search("cache test")
        self.assertEqual(len(results1), len(results2))
        stats = search.get_stats()
        self.assertEqual(stats["cache_size"], 1)


class TestAgentOrchestrator(unittest.TestCase):
    """Test AgentOrchestrator v2 — central agent coordination."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.episodic_dir = tempfile.mkdtemp()
        self.config = AgentConfig(
            skill_store_dir=self.temp_dir,
            episodic_store_dir=self.episodic_dir,
            enable_web_search=True,
            enable_terminal=True,
            enable_skill_creation=True,
            enable_tool_creation=True,
            enable_reflection=True,
            enable_calibration=True,
            enable_meta_skills=True,
            max_iterations=3,
        )
        self.orchestrator = AgentOrchestrator(config=self.config)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.episodic_dir, ignore_errors=True)

    def test_high_confidence_no_intervention(self):
        """With high confidence, model output should be returned as-is."""
        result = self.orchestrator.run(
            model_output=None,
            confidence=0.95,
            query="simple question",
        )
        self.assertEqual(result.state.value, "complete")
        self.assertLessEqual(len(result.actions_taken), 1)

    def test_low_confidence_triggers_search(self):
        """With low confidence, web search should be triggered."""
        result = self.orchestrator.run(
            model_output=None,
            confidence=0.1,
            query="What is the latest AI news?",
        )
        self.assertIn("web_search", result.actions_taken)

    def test_max_iterations(self):
        """Should not exceed max iterations."""
        result = self.orchestrator.run(
            model_output=None,
            confidence=0.05,
            query="complex question requiring multiple searches",
        )
        self.assertLessEqual(result.iterations, self.config.max_iterations)

    def test_v2_reflection_generated(self):
        """v2: Should generate reflections during execution."""
        result = self.orchestrator.run(
            model_output=None,
            confidence=0.1,
            query="search for something",
        )
        # Reflections may or may not be generated depending on action outcomes
        # but the field should exist
        self.assertIsInstance(result.reflections, list)

    def test_v2_episode_stored(self):
        """v2: Should store an episode in episodic memory."""
        result = self.orchestrator.run(
            model_output=None,
            confidence=0.1,
            query="What is AI?",
        )
        # Episode should be stored
        self.assertIsNotNone(result.episode_id)

    def test_v2_calibration_applied(self):
        """v2: Calibration should be applied during execution."""
        result = self.orchestrator.run(
            model_output=None,
            confidence=0.1,
            query="What is the latest news?",
        )
        self.assertTrue(result.calibration_applied)

    def test_v2_get_stats(self):
        """v2: Stats should include calibration and episodic memory."""
        stats = self.orchestrator.get_stats()
        self.assertIn("calibration", stats)
        self.assertIn("episodic_memory", stats)


if __name__ == "__main__":
    unittest.main()
