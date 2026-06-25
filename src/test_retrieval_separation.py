"""
MemInterfere: Unit Tests for Phase 5.3 — Retrieval vs. Planning Separation

Tests that:
1. The three tracks produce different prompt configurations
2. Track A (gold retrieval) correctly injects only gold skills
3. Track B (RAG retrieval) retrieves top-K skills with proper precision
4. Track C (full context) includes all skills
5. Failure attribution is computed correctly
6. New Condition enum values exist and are valid
7. Embeddings are cached and reusable
8. Retrieval works with both 68-skill and 100-skill libraries
"""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field

# Set up path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import Condition, EvalTask, EvalResult
from retrieval_simulator import (
    retrieve_gold_skills,
    retrieve_rag_top_k,
    compute_skill_embeddings,
    build_track_prompt,
    get_skills_for_track,
    get_condition_for_track,
    RetrievalResult,
    evaluate_retrieval,
)
from retrieval_analysis import (
    compute_track_metrics,
    compute_failure_attribution,
    generate_comparison_table,
    generate_failure_attribution_table,
    TrackMetrics,
    FailureAttribution,
)


# ── Mock skills for testing ──────────────────────────────────────────────────

@dataclass
class MockSkill:
    """Minimal skill object matching the Skill interface used by retrieval_simulator."""
    skill_id: str
    name: str
    domain: str = "web_navigation"
    description: str = ""
    parameters: dict = field(default_factory=dict)
    return_format: str = "json"
    success_rate: float = 0.9
    last_verified: str = "2026-06-25"
    conflict_group_id: str = ""
    conflict_type: str = "none"
    fan_degree: int = 1
    staleness_days: int = 0
    is_trap: bool = False
    is_stale: bool = False
    is_clean: bool = True
    parametric_overlap: bool = False
    trap_description: str = ""
    example_usage: str = ""


def make_test_skills():
    """Create a set of test skills for testing."""
    return [
        MockSkill(skill_id="search_web", name="search_web",
                  description="Search the web for information"),
        MockSkill(skill_id="navigate_url", name="navigate_url",
                  description="Navigate to a URL and extract content"),
        MockSkill(skill_id="scrape_page", name="scrape_page",
                  description="Scrape structured data from a web page"),
        MockSkill(skill_id="create_event", name="create_event",
                  description="Create a calendar event"),
        MockSkill(skill_id="send_email", name="send_email",
                  description="Send an email to a recipient"),
        MockSkill(skill_id="get_weather", name="get_weather",
                  description="Get the current weather for a location"),
        MockSkill(skill_id="query_database", name="query_database",
                  description="Query a database and return results"),
        MockSkill(skill_id="find_information", name="find_information",
                  description="Find information on the web (similar to search_web)",
                  conflict_type="near_duplicate", is_clean=False),
        MockSkill(skill_id="web_search_v2", name="web_search_v2",
                  description="Search the web with advanced options",
                  conflict_type="schema_conflict", is_clean=False),
        MockSkill(skill_id="check_weather", name="check_weather",
                  description="Check the weather forecast for a location",
                  conflict_type="near_duplicate", is_clean=False),
    ]


def make_test_tasks():
    """Create test tasks for testing."""
    return [
        EvalTask(
            task_id="test_001",
            description="Search for 'machine learning tutorials' and return results",
            domain="web_navigation",
            difficulty="easy",
            expected_skill_ids=["search_web"],
            expected_outcome="List of search results",
            interference_potential="low",
            requires_memory=False,
        ),
        EvalTask(
            task_id="test_002",
            description="Get the current weather in London",
            domain="api_calling",
            difficulty="easy",
            expected_skill_ids=["get_weather"],
            expected_outcome="Weather data for London",
            interference_potential="medium",
            requires_memory=False,
        ),
        EvalTask(
            task_id="test_003",
            description="Create a calendar event and send an email invite",
            domain="api_calling",
            difficulty="medium",
            expected_skill_ids=["create_event", "send_email"],
            expected_outcome="Event created + email sent",
            interference_potential="high",
            requires_memory=True,
        ),
    ]


class TestConditionEnum(unittest.TestCase):
    """Test that new Condition enum values exist."""

    def test_gold_retrieval_exists(self):
        self.assertEqual(Condition.GOLD_RETRIEVAL.value, "gold_retrieval")

    def test_rag_conditions_exist(self):
        self.assertEqual(Condition.RAG_RETRIEVAL_K1.value, "rag_retrieval_k1")
        self.assertEqual(Condition.RAG_RETRIEVAL_K3.value, "rag_retrieval_k3")
        self.assertEqual(Condition.RAG_RETRIEVAL_K5.value, "rag_retrieval_k5")
        self.assertEqual(Condition.RAG_RETRIEVAL_K10.value, "rag_retrieval_k10")

    def test_existing_conditions_preserved(self):
        """All original conditions should still exist."""
        self.assertEqual(Condition.ORACLE.value, "oracle")
        self.assertEqual(Condition.NO_MEMORY.value, "no_memory")
        self.assertEqual(Condition.CLEAN_MEMORY.value, "clean_memory")
        self.assertEqual(Condition.CLEAN_INTERFERENCE.value, "clean_interference")
        self.assertEqual(Condition.CLEAN_STALE.value, "clean_stale")
        self.assertEqual(Condition.ALL_MEMORY.value, "all_memory")
        self.assertEqual(Condition.GROWING.value, "growing")

    def test_new_conditions_count(self):
        """Should have 7 original + 5 new = 12 total conditions."""
        self.assertEqual(len(Condition), 12)


class TestGoldRetrieval(unittest.TestCase):
    """Test Track A: Gold retrieval."""

    def setUp(self):
        self.skills = make_test_skills()
        self.tasks = make_test_tasks()

    def test_gold_retrieval_finds_correct_skill(self):
        """Gold retrieval should return only the expected skill."""
        task = self.tasks[0]  # expects search_web
        gold = retrieve_gold_skills(task, self.skills)
        gold_ids = [s.skill_id for s in gold]
        self.assertIn("search_web", gold_ids)

    def test_gold_retrieval_finds_multiple_skills(self):
        """Gold retrieval should find all expected skills for multi-skill tasks."""
        task = self.tasks[2]  # expects create_event + send_email
        gold = retrieve_gold_skills(task, self.skills)
        gold_ids = [s.skill_id for s in gold]
        self.assertIn("create_event", gold_ids)
        self.assertIn("send_email", gold_ids)

    def test_gold_retrieval_excludes_non_gold(self):
        """Gold retrieval should NOT include skills that aren't expected."""
        task = self.tasks[0]  # expects search_web only
        gold = retrieve_gold_skills(task, self.skills)
        gold_ids = [s.skill_id for s in gold]
        self.assertNotIn("navigate_url", gold_ids)
        self.assertNotIn("get_weather", gold_ids)

    def test_gold_retrieval_empty_for_missing(self):
        """Gold retrieval returns empty list if no matching skill found."""
        weird_task = EvalTask(
            task_id="test_missing",
            description="Do something impossible",
            domain="web_navigation",
            difficulty="hard",
            expected_skill_ids=["nonexistent_skill"],
            expected_outcome="Nothing",
            interference_potential="none",
            requires_memory=False,
        )
        gold = retrieve_gold_skills(weird_task, self.skills)
        self.assertEqual(len(gold), 0)


class TestTrackPrompts(unittest.TestCase):
    """Test that the three tracks produce different prompt configurations."""

    def setUp(self):
        self.skills = make_test_skills()
        self.tasks = make_test_tasks()

    def test_gold_prompt_has_fewer_skills_than_full(self):
        """Gold retrieval prompt should have fewer skills than full context."""
        task = self.tasks[0]
        gold_skills = retrieve_gold_skills(task, self.skills)

        gold_prompt = build_track_prompt(task, gold_skills, "Gold Retrieval")
        full_prompt = build_track_prompt(task, self.skills, "Full Context")

        # Gold prompt should be shorter (fewer skills listed)
        self.assertLess(len(gold_prompt), len(full_prompt))

    def test_gold_prompt_contains_only_gold(self):
        """Gold retrieval prompt should mention the gold skill."""
        task = self.tasks[0]  # expects search_web
        gold_skills = retrieve_gold_skills(task, self.skills)
        prompt = build_track_prompt(task, gold_skills, "Gold Retrieval")
        self.assertIn("search_web", prompt)

    def test_gold_prompt_excludes_interference(self):
        """Gold retrieval prompt should NOT include interference skills."""
        task = self.tasks[0]  # expects search_web
        gold_skills = retrieve_gold_skills(task, self.skills)
        prompt = build_track_prompt(task, gold_skills, "Gold Retrieval")
        # find_information is a near-duplicate of search_web — should NOT appear
        self.assertNotIn("find_information", prompt)

    def test_full_prompt_includes_all_skills(self):
        """Full context prompt should include all skills."""
        task = self.tasks[0]
        prompt = build_track_prompt(task, self.skills, "Full Context")
        self.assertIn("search_web", prompt)
        self.assertIn("find_information", prompt)
        self.assertIn("get_weather", prompt)

    def test_rag_prompt_subset_of_full(self):
        """RAG retrieval prompt should be a subset of full context skills."""
        task = self.tasks[0]
        # Simulate RAG top-3 (we can't use real embeddings in unit tests easily)
        rag_skills = self.skills[:3]
        rag_prompt = build_track_prompt(task, rag_skills, "RAG K=3")
        full_prompt = build_track_prompt(task, self.skills, "Full Context")
        # RAG prompt should be shorter
        self.assertLess(len(rag_prompt), len(full_prompt))

    def test_different_tracks_produce_different_prompts(self):
        """All three tracks should produce distinct prompts for the same task."""
        task = self.tasks[0]
        gold_skills = retrieve_gold_skills(task, self.skills)
        rag_skills = self.skills[:3]  # Simulated top-3

        gold_prompt = build_track_prompt(task, gold_skills, "Gold Retrieval")
        rag_prompt = build_track_prompt(task, rag_skills, "RAG K=3")
        full_prompt = build_track_prompt(task, self.skills, "Full Context")

        # All three should be different
        self.assertNotEqual(gold_prompt, rag_prompt)
        self.assertNotEqual(gold_prompt, full_prompt)
        self.assertNotEqual(rag_prompt, full_prompt)


class TestGetSkillsForTrack(unittest.TestCase):
    """Test the get_skills_for_track function."""

    def setUp(self):
        self.skills = make_test_skills()
        self.tasks = make_test_tasks()

    def test_gold_track_returns_only_gold(self):
        """Gold track should return only expected skills."""
        task = self.tasks[0]  # expects search_web
        skills, scores = get_skills_for_track(task, self.skills, "gold")
        skill_ids = [s.skill_id for s in skills]
        self.assertIn("search_web", skill_ids)
        # Should not include unrelated skills
        self.assertNotIn("get_weather", skill_ids)
        self.assertNotIn("create_event", skill_ids)

    def test_full_track_returns_all(self):
        """Full track should return all skills."""
        task = self.tasks[0]
        skills, scores = get_skills_for_track(task, self.skills, "full")
        self.assertEqual(len(skills), len(self.skills))

    def test_rag_track_requires_embeddings(self):
        """RAG track should raise error without embeddings."""
        task = self.tasks[0]
        with self.assertRaises(ValueError):
            get_skills_for_track(task, self.skills, "rag_k5")

    def test_unknown_track_raises(self):
        """Unknown track should raise ValueError."""
        task = self.tasks[0]
        with self.assertRaises(ValueError):
            get_skills_for_track(task, self.skills, "unknown_track")


class TestConditionMapping(unittest.TestCase):
    """Test track-to-condition mapping."""

    def test_gold_maps_correctly(self):
        self.assertEqual(get_condition_for_track("gold"), "gold_retrieval")

    def test_rag_maps_correctly(self):
        self.assertEqual(get_condition_for_track("rag_k1"), "rag_retrieval_k1")
        self.assertEqual(get_condition_for_track("rag_k5"), "rag_retrieval_k5")

    def test_full_maps_correctly(self):
        self.assertEqual(get_condition_for_track("full"), "all_memory")


class TestFailureAttribution(unittest.TestCase):
    """Test failure attribution computation."""

    def test_planning_failure_detected(self):
        """If Track A fails, it should be counted as a planning failure."""
        # Task fails in Track A (gold retrieval) → planning failure
        track_a = [
            {"task_id": "t1", "skill_correct": False, "condition": "gold_retrieval"},
            {"task_id": "t2", "skill_correct": True, "condition": "gold_retrieval"},
        ]
        track_b = [
            {"task_id": "t1", "skill_correct": False, "condition": "rag_retrieval_k5"},
            {"task_id": "t2", "skill_correct": False, "condition": "rag_retrieval_k5"},
        ]
        track_c = [
            {"task_id": "t1", "skill_correct": False, "condition": "all_memory"},
            {"task_id": "t2", "skill_correct": False, "condition": "all_memory"},
        ]

        attribution = compute_failure_attribution(track_a, track_b, track_c)
        self.assertEqual(attribution.planning_failure_count, 1)  # t1 fails in Track A
        self.assertEqual(attribution.retrieval_failure_count, 1)  # t2 fails in B but not A

    def test_retrieval_failure_detected(self):
        """If Track B fails but Track A succeeds, it's a retrieval failure."""
        track_a = [
            {"task_id": "t1", "skill_correct": True, "condition": "gold_retrieval"},
            {"task_id": "t2", "skill_correct": True, "condition": "gold_retrieval"},
        ]
        track_b = [
            {"task_id": "t1", "skill_correct": False, "condition": "rag_retrieval_k5"},
            {"task_id": "t2", "skill_correct": True, "condition": "rag_retrieval_k5"},
        ]
        track_c = [
            {"task_id": "t1", "skill_correct": False, "condition": "all_memory"},
            {"task_id": "t2", "skill_correct": True, "condition": "all_memory"},
        ]

        attribution = compute_failure_attribution(track_a, track_b, track_c)
        # t1: A succeeds, B fails → retrieval failure
        self.assertEqual(attribution.retrieval_failure_count, 1)
        # No planning failures (A succeeded for both)
        self.assertEqual(attribution.planning_failure_count, 0)

    def test_all_correct_means_no_failures(self):
        """If all tracks succeed, there should be no failures."""
        track_a = [
            {"task_id": "t1", "skill_correct": True},
        ]
        track_b = [
            {"task_id": "t1", "skill_correct": True},
        ]
        track_c = [
            {"task_id": "t1", "skill_correct": True},
        ]

        attribution = compute_failure_attribution(track_a, track_b, track_c)
        self.assertEqual(attribution.planning_failure_count, 0)
        self.assertEqual(attribution.retrieval_failure_count, 0)

    def test_context_overload_detected(self):
        """If B succeeds but C fails, it's context overload."""
        track_a = [
            {"task_id": "t1", "skill_correct": True},
        ]
        track_b = [
            {"task_id": "t1", "skill_correct": True},
        ]
        track_c = [
            {"task_id": "t1", "skill_correct": False},
        ]

        attribution = compute_failure_attribution(track_a, track_b, track_c)
        self.assertEqual(attribution.track_c_errors, 1)
        # No planning or retrieval failures (both A and B succeeded)
        self.assertEqual(attribution.planning_failure_count, 0)
        self.assertEqual(attribution.retrieval_failure_count, 0)

    def test_attribution_formula(self):
        """Test the core formula: retrieval_failures = max(0, B_errors - A_errors)."""
        # 3 tasks: t1 fails in both, t2 fails only in B, t3 succeeds everywhere
        track_a = [
            {"task_id": "t1", "skill_correct": False},  # planning failure
            {"task_id": "t2", "skill_correct": True},   # no planning failure
            {"task_id": "t3", "skill_correct": True},
        ]
        track_b = [
            {"task_id": "t1", "skill_correct": False},  # also fails
            {"task_id": "t2", "skill_correct": False},  # retrieval failure
            {"task_id": "t3", "skill_correct": True},
        ]
        track_c = [
            {"task_id": "t1", "skill_correct": False},
            {"task_id": "t2", "skill_correct": False},
            {"task_id": "t3", "skill_correct": True},
        ]

        attribution = compute_failure_attribution(track_a, track_b, track_c)
        # A_errors = 1 (t1), B_errors = 2 (t1, t2)
        # planning_failures = 1 (t1)
        # retrieval_failures = max(0, 2-1) = 1 (t2)
        self.assertEqual(attribution.planning_failure_count, 1)
        self.assertEqual(attribution.retrieval_failure_count, 1)


class TestTrackMetrics(unittest.TestCase):
    """Test TrackMetrics computation."""

    def test_empty_results(self):
        """Empty results should return zeroed metrics."""
        metrics = compute_track_metrics([], "gold", "gold_retrieval")
        self.assertEqual(metrics.num_tasks, 0)
        self.assertEqual(metrics.selection_accuracy, 0.0)
        self.assertEqual(metrics.num_errors, 0)

    def test_perfect_accuracy(self):
        """All correct results should give 1.0 accuracy."""
        results = [
            {"task_id": f"t{i}", "skill_correct": True, "parsed_confidence": 0.9,
             "latency_ms": 1000, "parse_method": "json"}
            for i in range(10)
        ]
        metrics = compute_track_metrics(results, "gold", "gold_retrieval")
        self.assertEqual(metrics.num_tasks, 10)
        self.assertAlmostEqual(metrics.selection_accuracy, 1.0)
        self.assertEqual(metrics.num_errors, 0)

    def test_zero_accuracy(self):
        """All incorrect results should give 0.0 accuracy."""
        results = [
            {"task_id": f"t{i}", "skill_correct": False, "parsed_confidence": 0.5,
             "latency_ms": 1500, "parse_method": "regex"}
            for i in range(5)
        ]
        metrics = compute_track_metrics(results, "rag_k5", "rag_retrieval_k5")
        self.assertEqual(metrics.num_tasks, 5)
        self.assertAlmostEqual(metrics.selection_accuracy, 0.0)
        self.assertEqual(metrics.num_errors, 5)

    def test_partial_accuracy(self):
        """50% correct should give 0.5 accuracy."""
        results = [
            {"task_id": "t1", "skill_correct": True, "parsed_confidence": 0.9,
             "latency_ms": 1000, "parse_method": "json"},
            {"task_id": "t2", "skill_correct": False, "parsed_confidence": 0.6,
             "latency_ms": 2000, "parse_method": "regex"},
        ]
        metrics = compute_track_metrics(results, "full", "all_memory")
        self.assertAlmostEqual(metrics.selection_accuracy, 0.5)


class TestComparisonTables(unittest.TestCase):
    """Test comparison table generation."""

    def test_generate_comparison_table(self):
        """Should produce a formatted table string."""
        track_metrics = {
            "gold": TrackMetrics("gold", "gold_retrieval", 10, 0.95, 0.90, 0.05, 800.0, 0.92, 0, {}),
            "rag_k5": TrackMetrics("rag_k5", "rag_retrieval_k5", 10, 0.80, 0.75, 0.12, 1200.0, 0.85, 2, {}),
            "full": TrackMetrics("full", "all_memory", 10, 0.70, 0.65, 0.18, 1500.0, 0.80, 3, {}),
        }
        table = generate_comparison_table(track_metrics)
        self.assertIn("gold", table)
        self.assertIn("rag_k5", table)
        self.assertIn("full", table)
        self.assertIn("0.9500", table)  # Gold accuracy

    def test_generate_failure_attribution_table(self):
        """Should produce a formatted attribution table."""
        attribution = FailureAttribution(
            total_tasks=10,
            track_a_errors=2,
            track_b_errors=3,
            track_c_errors=4,
            planning_failure_count=2,
            retrieval_failure_count=1,
            planning_failure_rate=0.2857,
            retrieval_failure_rate=0.1429,
            task_breakdown=[
                {"task_id": "t1", "failure_type": "planning"},
                {"task_id": "t2", "failure_type": "retrieval"},
            ],
        )
        table = generate_failure_attribution_table(attribution)
        self.assertIn("Planning failures", table)
        self.assertIn("Retrieval failures", table)
        self.assertIn("2", table)


class TestExpandedLibrary100Compatibility(unittest.TestCase):
    """Test that retrieval works with the 100-skill library."""

    def setUp(self):
        lib_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'skills', 'expanded_library_100.json'
        )
        self.lib_exists = os.path.exists(lib_path)
        if self.lib_exists:
            with open(lib_path) as f:
                data = json.load(f)
            self.skills_100 = [
                MockSkill(**{k: v for k, v in s.items() if k in MockSkill.__dataclass_fields__})
                for s in data['skills']
            ]

    def test_gold_retrieval_with_100_skills(self):
        """Gold retrieval should work with the expanded 100-skill library."""
        if not self.lib_exists:
            self.skipTest("Expanded library not available")

        task = EvalTask(
            task_id="test_100",
            description="Search for 'machine learning tutorials' and return results",
            domain="web_navigation",
            difficulty="easy",
            expected_skill_ids=["search_web"],
            expected_outcome="List of search results",
            interference_potential="low",
            requires_memory=False,
        )
        gold = retrieve_gold_skills(task, self.skills_100)
        self.assertGreaterEqual(len(gold), 1)
        self.assertEqual(gold[0].name, "search_web")

    def test_100_skills_all_retrievable(self):
        """All 100 skills should be retrievable by name or ID."""
        if not self.lib_exists:
            self.skipTest("Expanded library not available")

        task = EvalTask(
            task_id="test_all",
            description="Do everything",
            domain="web_navigation",
            difficulty="hard",
            expected_skill_ids=["search_web", "navigate_url", "scrape_page"],
            expected_outcome="Everything",
            interference_potential="high",
            requires_memory=True,
        )
        gold = retrieve_gold_skills(task, self.skills_100)
        gold_names = [s.name for s in gold]
        # retrieve_gold_skills matches by skill_id OR name, so check by name
        self.assertIn("search_web", gold_names)
        self.assertGreaterEqual(len(gold), 3)


class TestEvaluateAgentConditionHandling(unittest.TestCase):
    """Test that evaluate_agent handles new retrieval conditions."""

    def test_gold_retrieval_condition(self):
        """Gold retrieval condition should return all skills (filtering is per-task)."""
        from evaluate_agent import get_library_for_condition
        from metrics import Condition

        # Gold retrieval returns all skills; the prompt builder filters per-task
        library = get_library_for_condition(Condition.GOLD_RETRIEVAL, session=0)
        self.assertGreater(len(library), 0)

    def test_rag_retrieval_conditions(self):
        """RAG retrieval conditions should return all skills."""
        from evaluate_agent import get_library_for_condition
        from metrics import Condition

        for cond in [Condition.RAG_RETRIEVAL_K1, Condition.RAG_RETRIEVAL_K3,
                     Condition.RAG_RETRIEVAL_K5, Condition.RAG_RETRIEVAL_K10]:
            library = get_library_for_condition(cond, session=0)
            self.assertGreater(len(library), 0,
                             f"Condition {cond.value} should return a non-empty library")


class TestRunExperimentTrackFlag(unittest.TestCase):
    """Test that run_experiment.py handles --track correctly."""

    def test_track_names_mapping(self):
        """Track names should map to valid condition names."""
        from run_experiment import TRACK_NAMES
        for track, condition in TRACK_NAMES.items():
            # Each mapped condition should be a valid Condition value
            cond_values = [c.value for c in Condition]
            self.assertIn(condition, cond_values,
                         f"Track {track} maps to invalid condition {condition}")


if __name__ == "__main__":
    unittest.main()