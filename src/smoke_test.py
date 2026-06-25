#!/usr/bin/env python3
"""
MemInterfere: Smoke Test for Phase 5.2 Multi-Model Evaluation Harness

Tests all components without requiring API keys:
1. Model config loading
2. Response parser cascade
3. Skill library loading
4. Task loading
5. Prompt construction
6. Mock end-to-end run
"""

import sys
import os
import json
import random

# Set up path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_config import MODELS, get_model, get_all_models, estimate_cost, print_model_summary
from response_parser import parse_response, ParsedResponse, ParseStats, batch_parse, PARSE_METHOD_ERROR
from metrics import EvalTask, Condition, EVAL_TASKS
from expanded_tasks import ALL_TASKS
from evaluate_agent import get_library_for_condition, get_skills_by_type, _load_skills
from multi_model_runner import MultiModelRunner, RunResult, build_system_prompt, build_user_prompt


def test_model_config():
    """Test model config module."""
    print("\n" + "="*60)
    print("TEST 1: Model Configuration")
    print("="*60)
    
    # Check all models loaded
    assert len(MODELS) == 4, f"Expected 4 models, got {len(MODELS)}"
    print(f"  ✓ Loaded {len(MODELS)} models: {', '.join(MODELS.keys())}")
    
    # Check each model has required fields
    for name, config in MODELS.items():
        assert config.api_base, f"Missing api_base for {name}"
        assert config.env_key, f"Missing env_key for {name}"
        assert config.model_id, f"Missing model_id for {name}"
        print(f"  ✓ {name}: {config.tier} tier, {config.provider}, JSON mode={'✓' if config.supports_json_mode else '✗'}")
    
    # Check get_model
    grok = get_model("grok-3-mini")
    assert grok.provider == "xai", f"Expected xai provider, got {grok.provider}"
    print(f"  ✓ get_model('grok-3-mini') works")
    
    # Check cost estimation
    cost = estimate_cost(grok, 100)
    assert cost > 0, "Cost should be > 0 for paid models"
    print(f"  ✓ Cost estimation: 100 runs of grok-3-mini = ${cost:.4f}")
    
    print_model_summary()
    return True


def test_response_parser():
    """Test response parser with various formats."""
    print("\n" + "="*60)
    print("TEST 2: Response Parser")
    print("="*60)
    
    sample_skills = [
        "search_web", "navigate_url", "scrape_page", "extract_links",
        "send_email", "create_event", "get_weather", "calculate",
        "query_database", "write_file",
    ]
    
    test_cases = [
        # JSON - direct
        ('{"tool": "search_web", "confidence": 0.9, "approach": "Search the web"}',
         "json", "search_web", 0.9),
        
        # JSON - markdown wrapped
        ('```json\n{"tool": "navigate_url", "confidence": 0.85, "approach": "Navigate directly"}\n```',
         "json", "navigate_url", 0.85),
        
        # Regex format
        ('TOOL: scrape_page\nCONFIDENCE: 0.75\nAPPROACH: Extract data using CSS selectors',
         "regex", "scrape_page", 0.75),
        
        # Freeform mention of skill
        ('I would use the send_email skill to send the email to the team.',
         "freeform", "send_email", None),  # confidence is default 0.5
        
        # Edge case: lowercase tool name in regex
        ('tool: get_weather\nconfidence: 0.6\napproach: Check the weather',
         "regex", "get_weather", 0.6),
        
        # Completely unparseable
        ('I am not sure what to do here.',
         "parse_error", "PARSE_ERROR", 0.0),
        
        # Empty response
        ('',
         "parse_error", "PARSE_ERROR", 0.0),
    ]
    
    all_passed = True
    for response, expected_method, expected_tool, expected_conf in test_cases:
        parsed = parse_response(response, sample_skills)
        method_ok = parsed.parse_method == expected_method
        tool_ok = parsed.tool_name == expected_tool
        conf_ok = expected_conf is None or abs(parsed.confidence - expected_conf) < 0.01
        
        status = "✓" if (method_ok and tool_ok and conf_ok) else "✗"
        print(f"  {status} Method={parsed.parse_method}, Tool={parsed.tool_name}, Conf={parsed.confidence:.2f} "
              f"(expected: {expected_method}/{expected_tool}/{expected_conf})")
        
        if not (method_ok and tool_ok and conf_ok):
            all_passed = False
    
    # Test batch parsing
    responses = [tc[0] for tc in test_cases]
    results, stats = batch_parse(responses, sample_skills)
    print(f"\n  Batch parse: {stats.total_count} total, {stats.parse_rate:.0%} parse rate")
    print(f"    JSON: {stats.json_count}, Regex: {stats.regex_count}, "
          f"Freeform: {stats.freeform_count}, Error: {stats.error_count}")
    
    assert stats.total_count == len(test_cases), f"Expected {len(test_cases)} results, got {stats.total_count}"
    # Parse rate is 5/7 = ~71% because we intentionally include 2 unparseable responses
    # On real model output, unparseable rate should be much lower
    assert stats.parse_rate >= 0.5, f"Parse rate too low: {stats.parse_rate:.0%}"
    
    return all_passed


def test_skill_library():
    """Test loading skill library and condition-based filtering."""
    print("\n" + "="*60)
    print("TEST 3: Skill Library & Conditions")
    print("="*60)
    
    # Load skills
    skills = _load_skills()
    assert len(skills) > 0, "Skill library should not be empty"
    print(f"  ✓ Loaded {len(skills)} skills")
    
    # Test condition libraries
    conditions = ["oracle", "no_memory", "clean_memory", "clean_interference", "clean_stale", "all_memory"]
    for cond_name in conditions:
        condition = Condition(cond_name)
        lib = get_library_for_condition(condition, session=0)
        print(f"  ✓ Condition '{cond_name}': {len(lib)} skills")
    
    # Verify library sizes make sense
    clean = get_skills_by_type("clean")
    interference = get_skills_by_type("interference")
    stale = get_skills_by_type("stale")
    trap = get_skills_by_type("trap")
    
    print(f"\n  Skill breakdown:")
    print(f"    Clean: {len(clean)}")
    print(f"    Interference: {len(interference)}")
    print(f"    Stale: {len(stale)}")
    print(f"    Trap: {len(trap)}")
    
    # clean_interference should have clean + interference + trap
    ci_lib = get_library_for_condition(Condition.CLEAN_INTERFERENCE, session=0)
    expected_size = len(clean) + len(interference) + len(trap)
    assert len(ci_lib) == expected_size, f"clean_interference: expected {expected_size}, got {len(ci_lib)}"
    print(f"  ✓ clean_interference library size matches: {len(ci_lib)}")
    
    return True


def test_task_loading():
    """Test loading evaluation tasks."""
    print("\n" + "="*60)
    print("TEST 4: Task Loading")
    print("="*60)
    
    print(f"  ✓ Base EVAL_TASKS: {len(EVAL_TASKS)}")
    print(f"  ✓ ALL_TASKS (with expanded): {len(ALL_TASKS)}")
    
    # Check task fields
    for task in ALL_TASKS[:3]:
        assert hasattr(task, 'task_id'), "Task should have task_id"
        assert hasattr(task, 'description'), "Task should have description"
        assert hasattr(task, 'expected_skill_ids'), "Task should have expected_skill_ids"
        print(f"  ✓ Task {task.task_id}: {task.difficulty} / {task.interference_potential}")
    
    return True


def test_prompt_construction():
    """Test building prompts from skill library and tasks."""
    print("\n" + "="*60)
    print("TEST 5: Prompt Construction")
    print("="*60)
    
    # Build a prompt for one task/condition combo
    skills = get_library_for_condition(Condition.CLEAN_INTERFERENCE, session=0)
    task = ALL_TASKS[0]
    
    system_prompt = build_system_prompt(skills, "clean_interference")
    user_prompt = build_user_prompt(task)
    
    assert "Skill Library" in system_prompt, "System prompt should contain skill library"
    assert task.description in user_prompt, "User prompt should contain task description"
    assert "TOOL:" in system_prompt, "System prompt should mention TOOL format"
    
    print(f"  ✓ System prompt: {len(system_prompt)} chars, {len(system_prompt.split(chr(10)))} lines")
    print(f"  ✓ User prompt: {len(user_prompt)} chars")
    
    # Show a snippet
    lines = system_prompt.split("\n")
    print(f"  First 5 lines of system prompt:")
    for line in lines[:5]:
        print(f"    {line[:80]}")
    
    return True


def test_mock_end_to_end():
    """Test a mock end-to-end run."""
    print("\n" + "="*60)
    print("TEST 6: Mock End-to-End Run")
    print("="*60)
    
    random.seed(42)
    
    # Get all skill names
    all_skills = _load_skills()
    valid_skill_names = list(set(s.name for s in all_skills))
    
    # Build libraries for 2 conditions
    conditions = ["clean_memory", "clean_interference"]
    skill_libraries = {}
    for cond_name in conditions:
        condition = Condition(cond_name)
        skill_libraries[cond_name] = get_library_for_condition(condition, session=0)
    
    # Use a small subset of tasks
    tasks = ALL_TASKS[:5]
    temperatures = [0.0]
    
    # Simulate mock results
    results = []
    parse_stats = ParseStats()
    
    for task in tasks:
        for cond_name in conditions:
            skill_lib = skill_libraries[cond_name]
            
            # Simulate different response formats
            for fmt_idx, fmt in enumerate(["json", "regex", "freeform"]):
                # Pick a random skill
                if skill_lib:
                    chosen_skill = random.choice(skill_lib)
                    mock_tool = chosen_skill.name
                else:
                    mock_tool = "none"
                
                if fmt == "json":
                    mock_response = json.dumps({
                        "tool": mock_tool,
                        "confidence": round(random.uniform(0.6, 1.0), 2),
                        "approach": f"Use {mock_tool} to accomplish the task"
                    })
                elif fmt == "regex":
                    mock_response = (
                        f"TOOL: {mock_tool}\n"
                        f"CONFIDENCE: {round(random.uniform(0.5, 1.0), 2)}\n"
                        f"APPROACH: I will use {mock_tool} for this task"
                    )
                else:
                    mock_response = f"I think we should use the {mock_tool} skill to handle this."
                
                parsed = parse_response(mock_response, valid_skill_names)
                parse_stats.record(parsed.parse_method)
                
                # Check skill correctness
                skill_correct = False
                for expected in task.expected_skill_ids:
                    for skill in skill_lib:
                        if (skill.skill_id == expected or skill.name == expected):
                            if parsed.tool_name == skill.name:
                                skill_correct = True
                                break
                
                result = RunResult(
                    run_id=f"mock_{task.task_id}_{cond_name}_fmt{fmt_idx}",
                    model="mock-test",
                    model_id="mock",
                    task_id=task.task_id,
                    condition=cond_name,
                    temperature=0.0,
                    latency_ms=random.randint(500, 2000),
                    timestamp="2026-06-25T00:00:00",
                    prompt_tokens=300,
                    completion_tokens=100,
                    total_tokens=400,
                    prompt_text="[mock]",
                    raw_response=mock_response,
                    parsed_tool=parsed.tool_name,
                    parsed_confidence=parsed.confidence,
                    parsed_approach=parsed.approach,
                    parse_method=parsed.parse_method,
                    all_matched_skills=parsed.all_matched_skills,
                    expected_skills=task.expected_skill_ids,
                    skill_correct=skill_correct,
                    error="",
                )
                results.append(result)
    
    # Print statistics
    print(f"  Total mock results: {len(results)}")
    print(f"  Parse stats: {parse_stats.summary()}")
    
    # Check parse rate
    assert parse_stats.parse_rate >= 0.9, f"Mock parse rate too low: {parse_stats.parse_rate:.0%}"
    
    # Accuracy
    correct = sum(1 for r in results if r.skill_correct)
    print(f"  Skill match rate: {correct}/{len(results)} ({correct/len(results):.1%})")
    
    # Parse methods breakdown
    method_counts = {}
    for r in results:
        method_counts[r.parse_method] = method_counts.get(r.parse_method, 0) + 1
    print(f"  Parse method breakdown: {method_counts}")
    
    # Save mock results
    from dataclasses import asdict as _asdict
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results", "phase5_smoke_test")
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        "metadata": {
            "timestamp": "2026-06-25T00:00:00",
            "mode": "smoke_test",
            "models": ["mock-test"],
            "conditions": conditions,
            "temperatures": [0.0],
            "num_tasks": len(tasks),
            "parse_stats": parse_stats.summary(),
        },
        "results": [_asdict(r) for r in results],
    }
    
    output_file = os.path.join(output_dir, "smoke_test_results.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Saved smoke test results to {output_file}")
    
    return True


def main():
    """Run all smoke tests."""
    print("="*60)
    print("MemInterfere Phase 5.2: Smoke Test Suite")
    print("="*60)
    
    # Remove unused asdict import reference - already imported at top
    tests = [
        ("Model Config", test_model_config),
        ("Response Parser", test_response_parser),
        ("Skill Library", test_skill_library),
        ("Task Loading", test_task_loading),
        ("Prompt Construction", test_prompt_construction),
        ("Mock End-to-End", test_mock_end_to_end),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                print(f"\n  ✅ {name}: PASSED")
                passed += 1
            else:
                print(f"\n  ❌ {name}: FAILED")
                failed += 1
        except Exception as e:
            print(f"\n  ❌ {name}: FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Smoke Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)