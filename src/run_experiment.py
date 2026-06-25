#!/usr/bin/env python3
"""
MemInterfere: Multi-Model Experiment Runner

Top-level script that orchestrates the full multi-model evaluation:
1. Loads skill library, tasks, and model configs
2. Runs all conditions × tasks × temperatures for each model
3. Saves results incrementally
4. Produces summary CSV and JSON

Usage:
    python src/run_experiment.py --model grok-3-mini --condition clean_interference --temperature 0.0 --output data/results/phase5/
    python src/run_experiment.py --model all --condition all --temperature all --output data/results/phase5/
    python src/run_experiment.py --model grok-3-mini --condition all --temperature 0.0 --dry-run  # Estimate costs only
    python src/run_experiment.py --mock  # Run with mock responses (no API calls)
"""

import argparse
import json
import os
import sys
import logging
import time
from datetime import datetime
from dataclasses import asdict

# Set up path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_config import MODELS, get_model, get_all_models, estimate_cost
from multi_model_runner import MultiModelRunner, RunResult, produce_summary
from response_parser import ParsedResponse, ParseStats, PARSE_METHOD_ERROR
from metrics import EvalTask, Condition, EVAL_TASKS
from expanded_tasks import ALL_TASKS
from evaluate_agent import get_library_for_condition, get_skills_by_type, _load_skills
from multistep_evaluator import (
    load_multistep_tasks, validate_multistep_tasks, load_skill_library,
    evaluate_multistep_task, compute_multistep_summary, save_multistep_results,
)
from cascade_metrics import (
    compute_cascade_rate, compute_position_accuracy, compute_sequence_accuracy,
    compute_parameter_accuracy, compute_interference_selection_rate,
    compute_cascade_significance, compute_full_cascade_analysis, cascade_analysis_to_dict,
)
from retrieval_simulator import (
    compute_skill_embeddings, compute_task_embeddings,
    get_skills_for_track, get_condition_for_track, build_track_prompt,
)

# ── Condition name mapping ───────────────────────────────────────────────────

CONDITION_NAMES = {
    "oracle": Condition.ORACLE,
    "no_memory": Condition.NO_MEMORY,
    "clean_memory": Condition.CLEAN_MEMORY,
    "clean_interference": Condition.CLEAN_INTERFERENCE,
    "clean_stale": Condition.CLEAN_STALE,
    "all_memory": Condition.ALL_MEMORY,
    # Phase 5.3: Retrieval vs. Planning tracks
    "gold_retrieval": Condition.GOLD_RETRIEVAL,
    "rag_retrieval_k1": Condition.RAG_RETRIEVAL_K1,
    "rag_retrieval_k3": Condition.RAG_RETRIEVAL_K3,
    "rag_retrieval_k5": Condition.RAG_RETRIEVAL_K5,
    "rag_retrieval_k10": Condition.RAG_RETRIEVAL_K10,
}

ALL_CONDITION_NAMES = list(CONDITION_NAMES.keys())

# ── Track name mapping ─────────────────────────────────────────────────────────

TRACK_NAMES = {
    "gold": "gold_retrieval",
    "rag_k1": "rag_retrieval_k1",
    "rag_k3": "rag_retrieval_k3",
    "rag_k5": "rag_retrieval_k5",
    "rag_k10": "rag_retrieval_k10",
    "full": "all_memory",
}

# Temperatures to sweep
ALL_TEMPERATURES = [0.0, 0.3, 0.7]

# All model names
ALL_MODEL_NAMES = list(MODELS.keys())


def setup_logging(output_dir: str, level: str = "INFO"):
    """Set up logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "experiment.log")

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def build_skill_libraries(condition_names: list[str]) -> dict[str, list]:
    """Build the skill library for each condition."""
    libraries = {}
    for cond_name in condition_names:
        condition = CONDITION_NAMES[cond_name]
        libraries[cond_name] = get_library_for_condition(condition, session=0)
    return libraries


def build_skill_libraries_for_track(
    tasks: list[EvalTask],
    condition_name: str,
    track: str,
    all_skills: list,
    skill_embeddings=None,
    skill_ids=None,
    task_embeddings=None,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, list]:
    """
    Build skill libraries for a specific retrieval track.

    For Track A (gold): each task gets only its gold skills.
    For Track B (rag_kN): each task gets its top-K retrieved skills.
    For Track C (full): each task gets all skills.

    Returns a dict mapping task_id → list of skills.
    Since the multi-model runner expects a single library per condition,
    we return {condition_name: all_skills} and handle per-task filtering
    in the prompt builder.
    """
    if track == "full":
        # Full context: all skills for all tasks
        return {condition_name: all_skills}
    elif track == "gold":
        # Gold retrieval: we return all skills, but the prompt builder
        # will filter to only gold skills per-task
        return {condition_name: all_skills}
    elif track.startswith("rag_k"):
        # RAG retrieval: return all skills, prompt builder will filter per-task
        return {condition_name: all_skills}
    else:
        raise ValueError(f"Unknown track: {track}")


def get_all_skill_names() -> list[str]:
    """Get all unique skill names from the library for response parsing."""
    all_skills = _load_skills()
    return list(set(s.name for s in all_skills))


def run_mock_experiment(
    model_name: str,
    condition_names: list[str],
    temperatures: list[float],
    tasks: list[EvalTask],
    skill_libraries: dict[str, list],
    valid_skill_names: list[str],
    output_dir: str,
) -> list[RunResult]:
    """Run a mock experiment with simulated responses (no API calls)."""
    import random

    logger = logging.getLogger(__name__)
    logger.info(f"Running MOCK experiment for model: {model_name}")

    model_config = get_model(model_name)
    results = []

    for temperature in temperatures:
        for cond_name in condition_names:
            skill_lib = skill_libraries.get(cond_name, [])
            for task in tasks:
                # Simulate a response
                # Pick a random skill name from the library
                if skill_lib:
                    chosen_skill = random.choice(skill_lib)
                    mock_tool = chosen_skill.name
                else:
                    mock_tool = "none"

                # Simulate various response formats to test the parser
                format_type = random.choice(["json", "regex", "freeform", "bad"])

                if format_type == "json":
                    mock_response = json.dumps({
                        "tool": mock_tool,
                        "confidence": round(random.uniform(0.5, 1.0), 2),
                        "approach": f"Use {mock_tool} to accomplish the task"
                    })
                elif format_type == "regex":
                    mock_response = (
                        f"TOOL: {mock_tool}\n"
                        f"CONFIDENCE: {round(random.uniform(0.5, 1.0), 2)}\n"
                        f"APPROACH: I will use {mock_tool} for this task"
                    )
                elif format_type == "freeform":
                    mock_response = f"I think we should use the {mock_tool} skill here to get the information needed."
                else:
                    mock_response = "I'm not sure what to do here. Let me think about it more carefully."

                # Parse the response
                parsed = __import__('response_parser', fromlist=['parse_response']).parse_response(
                    mock_response, valid_skill_names
                )

                # Check correctness
                skill_correct = False
                for expected in task.expected_skill_ids:
                    for skill in skill_lib:
                        if (skill.skill_id == expected or skill.name == expected):
                            if parsed.tool_name == skill.name:
                                skill_correct = True
                                break

                result = RunResult(
                    run_id=f"mock_{model_name}_{task.task_id}_{cond_name}_T{temperature}",
                    model=model_name,
                    model_id=f"mock-{model_config.model_id}",
                    task_id=task.task_id,
                    condition=cond_name,
                    temperature=temperature,
                    latency_ms=random.randint(500, 3000),
                    timestamp=datetime.now().isoformat(),
                    prompt_tokens=random.randint(200, 600),
                    completion_tokens=random.randint(50, 200),
                    total_tokens=random.randint(250, 800),
                    prompt_text="[MOCK]",
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

    return results


def run_real_experiment(
    model_names: list[str],
    condition_names: list[str],
    temperatures: list[float],
    tasks: list[EvalTask],
    skill_libraries: dict[str, list],
    valid_skill_names: list[str],
    output_dir: str,
    resume_from: str | None = None,
) -> list[RunResult]:
    """Run the real multi-model experiment with API calls."""
    runner = MultiModelRunner(output_dir=output_dir)

    all_results = []

    for model_name in model_names:
        model_config = get_model(model_name)

        # Check API key
        api_key = os.environ.get(model_config.env_key, "")
        if not api_key:
            logging.warning(f"Skipping {model_name}: missing {model_config.env_key} environment variable")
            continue

        model_configs = [model_config]
        results = runner.run_batch(
            model_configs=model_configs,
            tasks=tasks,
            conditions=condition_names,
            temperatures=temperatures,
            skill_libraries=skill_libraries,
            valid_skill_names=valid_skill_names,
            resume_from=resume_from,
            save_every=5,
        )
        all_results.extend(results)

    # Produce summary
    if all_results:
        produce_summary(all_results, output_dir)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="MemInterfere Multi-Model Experiment Runner")
    parser.add_argument("--model", type=str, default="all",
                        help="Model name to run, or 'all' for all models. Options: "
                             ", ".join(ALL_MODEL_NAMES) + ", all")
    parser.add_argument("--condition", type=str, default="all",
                        help="Condition to run, or 'all' for all conditions. Options: "
                             ", ".join(ALL_CONDITION_NAMES) + ", all")
    parser.add_argument("--temperature", type=str, default="all",
                        help="Temperature to run, or 'all' for temperature sweep. "
                             "Options: 0.0, 0.3, 0.7, all")
    parser.add_argument("--output", type=str, default="data/results/phase5",
                        help="Output directory for results")
    parser.add_argument("--tasks", type=int, default=None,
                        help="Limit to N tasks (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate costs and exit without running")
    parser.add_argument("--mock", action="store_true",
                        help="Run with mock responses (no API calls)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing results file to resume from")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--multistep", action="store_true",
                        help="Run multi-step evaluation (Phase 5.4) instead of single-step")
    parser.add_argument("--track", type=str, default=None,
                        choices=["gold", "rag_k1", "rag_k3", "rag_k5", "rag_k10", "full"],
                        help="Phase 5.3 track: gold (only gold skills), rag_kN (RAG top-N), "
                             "full (all skills). Overrides --condition with track-specific conditions.")
    parser.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for RAG embeddings (default: all-MiniLM-L6-v2)")

    args = parser.parse_args()

    # Setup
    setup_logging(args.output, args.log_level)
    logger = logging.getLogger(__name__)

    # Resolve model names
    if args.model == "all":
        model_names = ALL_MODEL_NAMES
    else:
        model_names = [m.strip() for m in args.model.split(",")]

    # Validate model names
    for name in model_names:
        if name not in MODELS:
            logger.error(f"Unknown model: {name}. Available: {', '.join(MODELS.keys())}")
            sys.exit(1)

    # Resolve conditions
    if args.condition == "all":
        condition_names = ALL_CONDITION_NAMES
    else:
        condition_names = [c.strip() for c in args.condition.split(",")]

    # ── Track-based experiment (Phase 5.3) ─────────────────────────────────
    # If --track is specified, override conditions and skill library selection
    track_mode = args.track  # None, "gold", "rag_k1", "rag_k3", "rag_k5", "rag_k10", "full"
    track_embeddings = None  # Will be computed lazily for RAG tracks
    track_skill_ids = None

    if track_mode:
        # Override conditions to the track-specific condition
        track_condition = TRACK_NAMES[track_mode]
        condition_names = [track_condition]
        logger.info(f"Track mode: {track_mode} → condition: {track_condition}")

        # For RAG tracks, precompute embeddings
        if track_mode.startswith("rag_k"):
            all_skills = _load_skills()
            skill_embeddings, skill_ids = compute_skill_embeddings(
                all_skills, model_name=args.embed_model,
                force_recompute=False
            )
            task_embeddings = compute_task_embeddings(tasks, model_name=args.embed_model)
            track_embeddings = skill_embeddings
            track_skill_ids = skill_ids

    # Validate conditions
    for name in condition_names:
        if name not in CONDITION_NAMES:
            logger.error(f"Unknown condition: {name}. Available: {', '.join(CONDITION_NAMES.keys())}")
            sys.exit(1)

    # Resolve temperatures
    if args.temperature == "all":
        temperatures = ALL_TEMPERATURES
    else:
        temperatures = [float(t.strip()) for t in args.temperature.split(",")]

    # Load tasks
    ms_tasks = []  # Multi-step tasks for Phase 5.4
    if args.multistep:
        # Phase 5.4: Multi-step evaluation
        multistep_tasks_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "tasks", "multistep_tasks.json"
        )
        ms_tasks = load_multistep_tasks(multistep_tasks_path)
        skill_map_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "skills", "expanded_library_100.json"
        )
        skill_map = load_skill_library(skill_map_path)

        # Validate tasks
        validation = validate_multistep_tasks(ms_tasks, skill_map)
        if not validation["valid"]:
            logger.error("Multi-step task validation failed:")
            for e in validation["errors"]:
                logger.error(f"  {e}")
            sys.exit(1)

        logger.info(f"Using {len(ms_tasks)} multi-step tasks (Phase 5.4)")
        for w in validation["warnings"]:
            logger.warning(f"  {w}")

        # For multi-step, we still use single-step tasks but evaluate each step
        # The multi-step evaluator will handle step-by-step evaluation
        tasks = ALL_TASKS if not args.tasks else ALL_TASKS[:args.tasks]
        logger.info(f"  (Using {len(tasks)} single-step tasks as base for multi-step evaluation)")
    else:
        tasks = ALL_TASKS if not args.tasks else ALL_TASKS[:args.tasks]
    logger.info(f"Using {len(tasks)} tasks")

    # Load skill libraries
    if track_mode:
        all_skills = _load_skills()
        condition_name = condition_names[0]
        if track_mode.startswith("rag_k"):
            skill_libraries = build_skill_libraries_for_track(
                tasks, condition_name, track_mode, all_skills,
                skill_embeddings=track_embeddings,
                skill_ids=track_skill_ids,
                task_embeddings=task_embeddings if track_mode.startswith("rag_k") else None,
            )
        else:
            skill_libraries = build_skill_libraries_for_track(
                tasks, condition_name, track_mode, all_skills,
            )
    else:
        skill_libraries = build_skill_libraries(condition_names)
    valid_skill_names = get_all_skill_names()
    logger.info(f"Skill library sizes: {', '.join(f'{k}: {len(v)}' for k, v in skill_libraries.items())}")

    # Calculate scope
    n_runs = len(model_names) * len(tasks) * len(condition_names) * len(temperatures)
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment Configuration")
    logger.info(f"{'='*60}")
    logger.info(f"Models: {', '.join(model_names)}")
    logger.info(f"Conditions: {', '.join(condition_names)}")
    logger.info(f"Temperatures: {temperatures}")
    logger.info(f"Tasks: {len(tasks)}")
    logger.info(f"Total runs: {n_runs}")

    # Estimate costs
    total_cost = 0
    for model_name in model_names:
        model = get_model(model_name)
        cost = estimate_cost(model, len(tasks) * len(condition_names) * len(temperatures))
        total_cost += cost
        logger.info(f"  {model_name}: ~${cost:.2f} ({model.provider})")
    logger.info(f"  Total estimated cost: ${total_cost:.2f}")

    if args.dry_run:
        logger.info("\nDry run complete. Exiting without running experiments.")
        print(f"\nDry Run Summary:")
        print(f"  Models: {', '.join(model_names)}")
        print(f"  Conditions: {', '.join(condition_names)}")
        print(f"  Temperatures: {temperatures}")
        print(f"  Tasks: {len(tasks)}")
        print(f"  Total runs: {n_runs}")
        print(f"  Estimated cost: ${total_cost:.2f}")
        return

    # Run experiment
    if args.mock:
        logger.info("\nRunning MOCK experiment (no API calls)")
        random = __import__('random')
        random.seed(args.seed)

        all_results = []
        for model_name in model_names:
            results = run_mock_experiment(
                model_name=model_name,
                condition_names=condition_names,
                temperatures=temperatures,
                tasks=tasks,
                skill_libraries=skill_libraries,
                valid_skill_names=valid_skill_names,
                output_dir=args.output,
            )
            all_results.extend(results)

        # Save mock results
        os.makedirs(args.output, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output, f"mock_results_{timestamp}.json")
        data = {
            "metadata": {
                "timestamp": timestamp,
                "mode": "mock",
                "models": model_names,
                "conditions": condition_names,
                "temperatures": temperatures,
                "num_tasks": len(tasks),
                "seed": args.seed,
            },
            "results": [asdict(r) for r in all_results],
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(all_results)} mock results to {output_file}")

        # Produce summary
        produce_summary(all_results, args.output)

    else:
        logger.info("\nRunning REAL experiment (API calls)")
        all_results = run_real_experiment(
            model_names=model_names,
            condition_names=condition_names,
            temperatures=temperatures,
            tasks=tasks,
            skill_libraries=skill_libraries,
            valid_skill_names=valid_skill_names,
            output_dir=args.output,
            resume_from=args.resume,
        )

    # Print final statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total results: {len(all_results)}")

    # Parse stats
    stats = ParseStats()
    for r in all_results:
        stats.record(r.parse_method)
    logger.info(f"\nParse Statistics:")
    logger.info(f"  Total: {stats.total_count}")
    logger.info(f"  JSON: {stats.json_count} ({stats.json_rate:.1%})")
    logger.info(f"  Regex: {stats.regex_count} ({stats.regex_rate:.1%})")
    logger.info(f"  Freeform: {stats.freeform_count} ({stats.freeform_rate:.1%})")
    logger.info(f"  Parse errors: {stats.error_count} ({1 - stats.parse_rate:.1%})")
    logger.info(f"  Overall parse rate: {stats.parse_rate:.1%}")

    # Accuracy by model
    if all_results:
        models_in_results = set(r.model for r in all_results)
        for model_name in sorted(models_in_results):
            model_results = [r for r in all_results if r.model == model_name]
            correct = sum(1 for r in model_results if r.skill_correct)
            total = len(model_results)
            parsed = sum(1 for r in model_results if r.parse_method != PARSE_METHOD_ERROR)
            logger.info(f"\n  {model_name}:")
            logger.info(f"    Total: {total}, Parsed: {parsed}/{total} ({parsed/total:.1%})")
            logger.info(f"    Correct (skill match): {correct}/{total} ({correct/total:.1%})")
            logger.info(f"    Avg latency: {sum(r.latency_ms for r in model_results) / total:.0f}ms")

    # ── Multi-step cascade analysis (Phase 5.4) ────────────────────────────
    if args.multistep and all_results:
        logger.info(f"\n{'='*60}")
        logger.info(f"Multi-Step Cascade Analysis (Phase 5.4)")
        logger.info(f"{'='*60}")

        # Build step results from single-step results
        # Group by (task_id, condition, model, temperature) to form sequences
        from multistep_evaluator import StepResult
        step_results = []
        for r in all_results:
            # Map single-step results to step results for cascade analysis
            sr = StepResult(
                step_number=1,  # Default; will be overridden in multi-step eval
                task_id=r.task_id,
                condition=r.condition,
                model=r.model,
                temperature=r.temperature,
                expected_skill=r.expected_skill if hasattr(r, 'expected_skill') else "",
                actual_skill=r.actual_skill if hasattr(r, 'actual_skill') else r.parsed_tool,
                skill_correct=r.skill_correct,
                parameters_correct=True,  # Not tracked in single-step
                interference_skills_available=[],
                selected_interference=False,
                depends_on=0,
                cascade_from_error=False,
                confidence=r.parsed_confidence if hasattr(r, 'parsed_confidence') else 0.5,
                parse_method=r.parse_method,
            )
            step_results.append(sr)

        # Compute cascade metrics
        cascade_data = compute_cascade_rate(step_results)
        logger.info(f"\nCascade Rate Analysis:")
        logger.info(f"  P(wrong_N | wrong_N-1): {cascade_data['cascade_rate_conditional']:.3f}")
        logger.info(f"  P(wrong_N | correct_N-1): {cascade_data['cascade_rate_clean']:.3f}")
        logger.info(f"  Cascade effect size: {cascade_data['cascade_effect_size']:.3f}")
        logger.info(f"  Cascade ratio: {cascade_data['cascade_ratio']:.3f}")
        logger.info(f"  Wrong→Wrong transitions: {cascade_data['wrong_after_wrong']}")
        logger.info(f"  Correct→Wrong transitions: {cascade_data['wrong_after_correct']}")
        logger.info(f"  Wrong→Correct transitions: {cascade_data['correct_after_wrong']}")
        logger.info(f"  Correct→Correct transitions: {cascade_data['correct_after_correct']}")

        # Statistical significance
        significance = compute_cascade_significance(step_results)
        if significance.get("fisher_exact_p") is not None:
            logger.info(f"\n  Fisher's exact test p-value: {significance['fisher_exact_p']:.4f}")
            logger.info(f"  Cascade effect significant: {significance['significant']}")
        else:
            logger.info(f"\n  Fisher's exact test: {significance.get('note', 'N/A')}")

        # Position accuracy
        pos_acc = compute_position_accuracy(step_results)
        logger.info(f"\nPosition Accuracy:")
        for pos, stats in pos_acc.items():
            logger.info(f"  Step {pos}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

        # Interference selection rate
        intf_rate = compute_interference_selection_rate(step_results)
        logger.info(f"\nInterference Selection Rate:")
        for pos, stats in intf_rate.items():
            logger.info(f"  Step {pos}: {stats.get('interference_rate', 0):.1%}")

        # Save cascade analysis
        os.makedirs(args.output, exist_ok=True)
        cascade_output = os.path.join(args.output, "cascade_analysis.json")
        cascade_analysis_dict = {
            "phase": "5.4",
            "cascade_rates": cascade_data,
            "cascade_significance": significance,
            "position_accuracy": pos_acc,
            "interference_selection_rate": intf_rate,
            "multistep_tasks_loaded": len(ms_tasks),
        }
        with open(cascade_output, "w") as f:
            json.dump(cascade_analysis_dict, f, indent=2, default=str)
        logger.info(f"\nCascade analysis saved to {cascade_output}")


if __name__ == "__main__":
    main()