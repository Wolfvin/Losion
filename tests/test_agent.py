"""Tests for the Losion Agent Layer."""

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

        # Create new store instance with same directory
        new_store = SkillStore(store_dir=self.temp_dir)
        retrieved = new_store.lookup("persist-skill")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "persist-skill")


class TestSkillManager(unittest.TestCase):
    """Test SkillManager — high-level skill management."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = SkillStore(store_dir=self.temp_dir, auto_save=True)
        self.manager = SkillManager(store=self.store, auto_create=False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_and_lookup(self):
        """Should register a skill and find it via lookup."""
        self.manager.register_skill(
            name="test-skill",
            description="A test skill",
            skill_type="prompt",
            definition="test definition",
            domain="test",
        )

        result = self.manager.lookup("test-skill")
        self.assertTrue(result.found)
        self.assertEqual(result.skill.name, "test-skill")

    def test_lookup_not_found(self):
        """Should return not-found for missing skills."""
        result = self.manager.lookup("nonexistent")
        self.assertFalse(result.found)


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

    def test_search_tools(self):
        """Should search tools by query and domain."""
        self.registry.register(ToolEntry(
            name="python-exec",
            description="Execute Python code",
            handler=lambda code="": code,
            safety=ToolSafety.REQUIRES_APPROVAL,
            domain="code",
        ))

        results = self.registry.search(query="execute", domain="code")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "python-exec")


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

    def test_stats(self):
        """Should track execution statistics."""
        terminal = SandboxedTerminal()
        terminal.execute("echo test1")
        terminal.execute("echo test2")
        stats = terminal.get_stats()
        self.assertEqual(stats["total_executions"], 2)


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
        # First search
        results1 = search.search("cache test")
        # Second search (should be cached)
        results2 = search.search("cache test")
        self.assertEqual(len(results1), len(results2))
        stats = search.get_stats()
        self.assertEqual(stats["cache_size"], 1)


class TestAgentOrchestrator(unittest.TestCase):
    """Test AgentOrchestrator — central agent coordination."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = AgentConfig(
            skill_store_dir=self.temp_dir,
            enable_web_search=True,
            enable_terminal=True,
            enable_skill_creation=True,
            enable_tool_creation=True,
            max_iterations=3,
        )
        self.orchestrator = AgentOrchestrator(config=self.config)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_high_confidence_no_intervention(self):
        """With high confidence, model output should be returned as-is."""
        result = self.orchestrator.run(
            model_output=None,
            confidence=0.95,
            query="simple question",
        )
        self.assertEqual(result.state.value, "complete")
        # Should have minimal actions taken
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
            confidence=0.05,  # Very low
            query="complex question requiring multiple searches",
        )
        self.assertLessEqual(result.iterations, self.config.max_iterations)


if __name__ == "__main__":
    unittest.main()
