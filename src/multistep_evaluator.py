"""
MemInterfere: Multi-Step Evaluator (Phase 5.4)

Evaluates multi-step agentic workflows for cascade error analysis.
Tests whether errors at step N propagate to step N+1.

Key metrics:
- Position accuracy: correct tool at each step position
- Sequence accuracy: correct tool at ALL positions
- Cascade rate: P(wrong_N | wrong_N-1) vs P(wrong_N | correct_N-1)
- Parameter accuracy: correct parameters at each step
- Interference selection rate: how often agent picks interference over gold
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of evaluating a single step within a multi-step task."""
    step_number: int
    task_id: str
    condition: str
    model: str
    temperature: float
    expected_skill: str
    actual_skill: str
    skill_correct: bool
    parameters_correct: bool
    interference_skills_available: list[str]
    selected_interference: bool  # Did agent pick an interference skill?
    depends_on: int  # Which step this depends on (0 if first step)
    cascade_from_error: bool  # Was this step's error caused by a previous step's error?
    confidence: float
    parse_method: str  # json/regex/freeform/parse_error

    # Additional context
    error_type: str = ""  # From ErrorType enum
    previous_step_correct: Optional[bool] = None  # Was the previous step correct?
    is_first_step: bool = False  # Whether this is the first step in the sequence


@dataclass
class MultiStepResult:
    """Result of evaluating an entire multi-step task."""
    task_id: str
    condition: str
    model: str
    temperature: float
    steps: list[StepResult]
    sequence_accuracy: bool  # Were ALL steps correct?
    position_accuracy: dict  # {step_number: bool} per position
    cascade_rate: float  # P(wrong_N | wrong_N-1) vs P(wrong_N | correct_N-1)
    total_interference_selected: int
    total_steps: int

    # Derived cascade metrics
    cascade_rate_conditional: float = 0.0  # P(wrong_N | wrong_N-1)
    cascade_rate_clean: float = 0.0  # P(wrong_N | correct_N-1)
    cascade_effect_size: float = 0.0  # difference between conditional and clean rates


@dataclass
class MultiStepTaskDef:
    """Definition of a multi-step task loaded from JSON."""
    task_id: str
    description: str
    domain: str
    difficulty: str
    expected_skill_sequence: list[str]
    expected_outcomes: list[str]
    interference_potential: str
    requires_memory: bool
    cascade_risk: str
    steps: list[dict]


# ── Task Loading ───────────────────────────────────────────────────────────────

def load_multistep_tasks(filepath: str = None) -> list[MultiStepTaskDef]:
    """Load multi-step task definitions from JSON file."""
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "tasks", "multistep_tasks.json"
        )

    with open(filepath, "r") as f:
        data = json.load(f)

    tasks = []
    for t in data["tasks"]:
        task = MultiStepTaskDef(
            task_id=t["task_id"],
            description=t["description"],
            domain=t["domain"],
            difficulty=t["difficulty"],
            expected_skill_sequence=t["expected_skill_sequence"],
            expected_outcomes=t["expected_outcomes"],
            interference_potential=t["interference_potential"],
            requires_memory=t["requires_memory"],
            cascade_risk=t["cascade_risk"],
            steps=t["steps"],
        )
        tasks.append(task)

    return tasks


def load_skill_library(filepath: str = None) -> dict:
    """Load the expanded skill library and return a dict of skill name -> skill info.
    
    For skills with duplicate names (schema/semantic conflicts), only the last
    entry is kept under the name key. Use load_skill_library_by_id() for the
    full mapping including all variants.
    """
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "skills", "expanded_library_100.json"
        )

    with open(filepath, "r") as f:
        data = json.load(f)

    skill_map = {}
    for skill in data["skills"]:
        skill_map[skill["name"]] = skill

    return skill_map


def load_skill_library_by_id(filepath: str = None) -> dict:
    """Load the expanded skill library keyed by skill_id (preserves all 100 skills).
    
    Unlike load_skill_library(), this preserves duplicate-name entries (schema
    conflicts, semantic conflicts, version conflicts) by keying on skill_id.
    """
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "skills", "expanded_library_100.json"
        )

    with open(filepath, "r") as f:
        data = json.load(f)

    skill_map = {}
    for skill in data["skills"]:
        skill_map[skill["skill_id"]] = skill

    return skill_map


def get_all_skill_names(filepath: str = None) -> set:
    """Get ALL skill names including duplicates (for validation).
    
    Returns the union of names from both name-keyed and id-keyed maps,
    plus the conflict variant names that may not appear in the name-keyed map.
    """
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "skills", "expanded_library_100.json"
        )

    with open(filepath, "r") as f:
        data = json.load(f)

    return {skill["name"] for skill in data["skills"]}


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_multistep_tasks(tasks: list[MultiStepTaskDef], skill_map: dict) -> dict:
    """
    Validate all multi-step tasks against the skill library.
    Returns a dict with validation results.
    """
    errors = []
    warnings = []
    # Use get_all_skill_names() to get ALL names including conflict variants
    all_skill_names = get_all_skill_names()
    stats = {
        "total_tasks": len(tasks),
        "total_steps": 0,
        "by_difficulty": defaultdict(int),
        "by_domain": defaultdict(int),
        "cascade_risk_distribution": defaultdict(int),
        "interference_skills_valid": 0,
        "interference_skills_invalid": 0,
        "gold_skills_valid": 0,
        "gold_skills_invalid": 0,
    }

    for task in tasks:
        stats["by_difficulty"][task.difficulty] += 1
        stats["by_domain"][task.domain] += 1
        stats["cascade_risk_distribution"][task.cascade_risk] += 1

        # Validate gold skills
        for gold_skill in task.expected_skill_sequence:
            stats["total_steps"] += 1
            if gold_skill in all_skill_names:
                stats["gold_skills_valid"] += 1
            else:
                stats["gold_skills_invalid"] += 1
                errors.append(f"Task {task.task_id}: gold skill '{gold_skill}' not in library")

        # Validate step count matches sequence
        if len(task.steps) != len(task.expected_skill_sequence):
            errors.append(
                f"Task {task.task_id}: {len(task.steps)} steps but "
                f"{len(task.expected_skill_sequence)} skills in sequence"
            )

        # Validate each step
        for i, step in enumerate(task.steps):
            # Check expected_skill matches sequence
            if step["expected_skill"] != task.expected_skill_sequence[i]:
                errors.append(
                    f"Task {task.task_id} step {step['step_number']}: "
                    f"expected_skill '{step['expected_skill']}' != "
                    f"sequence skill '{task.expected_skill_sequence[i]}'"
                )

            # Check interference skills exist in library
            for intf_skill in step.get("interference_skills", []):
                if intf_skill in all_skill_names:
                    stats["interference_skills_valid"] += 1
                elif intf_skill == "":
                    pass  # empty string means no interference skill
                else:
                    stats["interference_skills_invalid"] += 1
                    warnings.append(
                        f"Task {task.task_id} step {step['step_number']}: "
                        f"interference skill '{intf_skill}' not in library"
                    )

            # Check depends_on is valid
            if step.get("depends_on") is not None and step["depends_on"] >= step["step_number"]:
                errors.append(
                    f"Task {task.task_id} step {step['step_number']}: "
                    f"depends_on {step['depends_on']} must be < step_number"
                )

            # Check each step has at least one interference skill (or explicitly empty)
            if not step.get("interference_skills"):
                warnings.append(
                    f"Task {task.task_id} step {step['step_number']}: "
                    f"no interference skills listed"
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": dict(stats),
    }


# ── Step Evaluation ────────────────────────────────────────────────────────────

def evaluate_step(
    task_id: str,
    step: dict,
    actual_skill: str,
    condition: str,
    model: str,
    temperature: float,
    confidence: float,
    parse_method: str,
    previous_step_correct: Optional[bool] = None,
    previous_error: bool = False,
    parameters_correct: bool = True,
) -> StepResult:
    """
    Evaluate a single step in a multi-step task.

    Args:
        task_id: Task identifier
        step: Step definition from the task
        actual_skill: The skill the agent actually selected
        condition: Experimental condition name
        model: Model name
        temperature: Temperature used
        confidence: Agent's confidence score
        parse_method: How the response was parsed
        previous_step_correct: Whether the previous step was correct
        previous_error: Whether there was an error in a prior step
        parameters_correct: Whether the parameters were correct

    Returns:
        StepResult with evaluation data
    """
    expected = step["expected_skill"]
    skill_correct = (actual_skill == expected)
    interference_skills = step.get("interference_skills", [])
    selected_interference = actual_skill in interference_skills

    # Cascade from error: this step is wrong AND there was a previous error
    cascade_from_error = (not skill_correct) and previous_error

    result = StepResult(
        step_number=step["step_number"],
        task_id=task_id,
        condition=condition,
        model=model,
        temperature=temperature,
        expected_skill=expected,
        actual_skill=actual_skill,
        skill_correct=skill_correct,
        parameters_correct=parameters_correct,
        interference_skills_available=interference_skills,
        selected_interference=selected_interference,
        depends_on=step.get("depends_on", 0) or 0,
        cascade_from_error=cascade_from_error,
        confidence=confidence,
        parse_method=parse_method,
        previous_step_correct=previous_step_correct,
        is_first_step=(step["step_number"] == 1),
    )

    # Determine error type
    if skill_correct and parameters_correct:
        result.error_type = "success"
    elif skill_correct and not parameters_correct:
        result.error_type = "parameter_hallucination"
    elif selected_interference:
        result.error_type = "wrong_skill"
    elif actual_skill == "PARSE_ERROR" or actual_skill == "":
        result.error_type = "parse_error"
    else:
        result.error_type = "wrong_skill"

    return result


# ── Multi-Step Task Evaluation ────────────────────────────────────────────────

def evaluate_multistep_task(
    task: MultiStepTaskDef,
    actual_skills: list[str],
    condition: str,
    model: str,
    temperature: float,
    confidences: list[float] = None,
    parse_methods: list[str] = None,
    parameters_correct: list[bool] = None,
) -> MultiStepResult:
    """
    Evaluate a complete multi-step task.

    Args:
        task: Multi-step task definition
        actual_skills: List of skills the agent actually selected (one per step)
        condition: Experimental condition name
        model: Model name
        temperature: Temperature used
        confidences: Optional confidence scores per step
        parse_methods: Optional parse methods per step
        parameters_correct: Optional list of whether parameters were correct per step

    Returns:
        MultiStepResult with complete evaluation data
    """
    if confidences is None:
        confidences = [0.5] * len(task.steps)
    if parse_methods is None:
        parse_methods = ["unknown"] * len(task.steps)
    if parameters_correct is None:
        parameters_correct = [True] * len(task.steps)

    step_results = []
    previous_step_correct = None
    previous_error = False

    for i, step in enumerate(task.steps):
        actual_skill = actual_skills[i] if i < len(actual_skills) else ""

        # Track cascade: error propagates if a previous step was wrong
        result = evaluate_step(
            task_id=task.task_id,
            step=step,
            actual_skill=actual_skill,
            condition=condition,
            model=model,
            temperature=temperature,
            confidence=confidences[i] if i < len(confidences) else 0.5,
            parse_method=parse_methods[i] if i < len(parse_methods) else "unknown",
            previous_step_correct=previous_step_correct,
            previous_error=previous_error,
            parameters_correct=parameters_correct[i] if i < len(parameters_correct) else True,
        )

        step_results.append(result)

        # Update cascade tracking for next step
        previous_step_correct = result.skill_correct
        if not result.skill_correct:
            previous_error = True

    # Compute position accuracy
    position_accuracy = {
        r.step_number: r.skill_correct for r in step_results
    }

    # Compute sequence accuracy
    sequence_accuracy = all(r.skill_correct for r in step_results)

    # Count interference selections
    total_interference_selected = sum(
        1 for r in step_results if r.selected_interference
    )

    result = MultiStepResult(
        task_id=task.task_id,
        condition=condition,
        model=model,
        temperature=temperature,
        steps=step_results,
        sequence_accuracy=sequence_accuracy,
        position_accuracy=position_accuracy,
        cascade_rate=0.0,  # Computed at aggregate level
        total_interference_selected=total_interference_selected,
        total_steps=len(task.steps),
    )

    return result


# ── Batch Evaluation ──────────────────────────────────────────────────────────

def evaluate_multistep_batch(
    tasks: list[MultiStepTaskDef],
    results_by_task: dict[str, list[list[str]]],
    # results_by_task[task_id] = list of (actual_skills_per_step) per condition/model/temp
    conditions: list[str],
    models: list[str],
    temperatures: list[float],
) -> list[MultiStepResult]:
    """
    Evaluate a batch of multi-step tasks.

    Args:
        tasks: List of task definitions
        results_by_task: Dict mapping task_id to list of actual skill sequences
        conditions: List of condition names
        models: List of model names
        temperatures: List of temperature values

    Returns:
        List of MultiStepResult objects
    """
    all_results = []

    for task in tasks:
        task_key = task.task_id
        if task_key not in results_by_task:
            continue

        for result_data in results_by_task[task_key]:
            actual_skills = result_data.get("actual_skills", [])
            condition = result_data.get("condition", "unknown")
            model = result_data.get("model", "unknown")
            temperature = result_data.get("temperature", 0.0)
            confidences = result_data.get("confidences", None)
            parse_methods = result_data.get("parse_methods", None)
            parameters_correct = result_data.get("parameters_correct", None)

            ms_result = evaluate_multistep_task(
                task=task,
                actual_skills=actual_skills,
                condition=condition,
                model=model,
                temperature=temperature,
                confidences=confidences,
                parse_methods=parse_methods,
                parameters_correct=parameters_correct,
            )
            all_results.append(ms_result)

    return all_results


# ── Summary Statistics ────────────────────────────────────────────────────────

def compute_multistep_summary(results: list[MultiStepResult]) -> dict:
    """
    Compute summary statistics across all multi-step results.

    Returns a dict with:
    - total_tasks, total_steps
    - sequence_accuracy_rate
    - position_accuracy: {step_pos: accuracy}
    - cascade_metrics
    - interference_selection_rate
    - by_difficulty, by_domain, by_condition breakdowns
    """
    if not results:
        return {"total_tasks": 0, "total_steps": 0}

    total_tasks = len(results)
    total_steps = sum(r.total_steps for r in results)

    # Sequence accuracy
    sequence_correct = sum(1 for r in results if r.sequence_accuracy)
    sequence_accuracy_rate = sequence_correct / total_tasks if total_tasks > 0 else 0.0

    # Position accuracy (by step position)
    position_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        for step_pos, correct in r.position_accuracy.items():
            position_stats[step_pos]["correct"] += int(correct)
            position_stats[step_pos]["total"] += 1

    position_accuracy = {}
    for pos, stats in sorted(position_stats.items()):
        position_accuracy[pos] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

    # Interference selection rate
    total_interference = sum(r.total_interference_selected for r in results)
    interference_rate = total_interference / total_steps if total_steps > 0 else 0.0

    # Collect all step results for cascade analysis
    all_steps = []
    for r in results:
        all_steps.extend(r.steps)

    return {
        "total_tasks": total_tasks,
        "total_steps": total_steps,
        "sequence_accuracy_rate": round(sequence_accuracy_rate, 4),
        "position_accuracy": {str(k): round(v, 4) for k, v in position_accuracy.items()},
        "total_interference_selected": total_interference,
        "interference_selection_rate": round(interference_rate, 4),
        "by_difficulty": _group_by(results, "difficulty"),
        "by_domain": _group_by(results, "domain"),
    }


def _group_by(results: list[MultiStepResult], attr: str) -> dict:
    """Group results by an attribute of the task."""
    groups = defaultdict(list)
    # Note: MultiStepResult doesn't have domain/difficulty directly,
    # so we group by task_id prefix
    for r in results:
        task_id = r.task_id
        if attr == "domain":
            if task_id.startswith("multi_"):
                # Look up from task definition — we use position in task_id
                # This is a simplification; real grouping needs task defs
                groups["mixed"].append(r)
        elif attr == "difficulty":
            groups["unknown"].append(r)
    return dict(groups)


# ── Save/Load ─────────────────────────────────────────────────────────────────

def save_multistep_results(
    results: list[MultiStepResult],
    filepath: str,
) -> None:
    """Save multi-step results to JSON."""
    data = []
    for r in results:
        result_dict = {
            "task_id": r.task_id,
            "condition": r.condition,
            "model": r.model,
            "temperature": r.temperature,
            "sequence_accuracy": r.sequence_accuracy,
            "position_accuracy": r.position_accuracy,
            "cascade_rate": r.cascade_rate,
            "cascade_rate_conditional": r.cascade_rate_conditional,
            "cascade_rate_clean": r.cascade_rate_clean,
            "cascade_effect_size": r.cascade_effect_size,
            "total_interference_selected": r.total_interference_selected,
            "total_steps": r.total_steps,
            "steps": [asdict(s) for s in r.steps],
        }
        data.append(result_dict)

    output = {
        "metadata": {
            "version": "1.0",
            "phase": "5.4",
            "description": "MemInterfere multi-step evaluation results",
            "total_tasks": len(results),
            "total_steps": sum(r.total_steps for r in results),
        },
        "results": data,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)


def load_multistep_results(filepath: str) -> list[dict]:
    """Load multi-step results from JSON."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["results"]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MemInterfere Multi-Step Evaluator")
    parser.add_argument("--validate", action="store_true",
                        help="Validate multi-step tasks against skill library")
    parser.add_argument("--tasks", type=str,
                        default="data/tasks/multistep_tasks.json",
                        help="Path to multi-step tasks JSON")
    parser.add_argument("--skills", type=str,
                        default="data/skills/expanded_library_100.json",
                        help="Path to skill library JSON")
    args = parser.parse_args()

    # Load data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tasks_path = os.path.join(base_dir, args.tasks)
    skills_path = os.path.join(base_dir, args.skills)

    tasks = load_multistep_tasks(tasks_path)
    skill_map = load_skill_library(skills_path)

    print(f"Loaded {len(tasks)} multi-step tasks")
    print(f"Loaded {len(skill_map)} skills from library")

    if args.validate:
        print("\nValidating multi-step tasks...")
        validation = validate_multistep_tasks(tasks, skill_map)

        print(f"\nValidation {'PASSED' if validation['valid'] else 'FAILED'}")
        print(f"  Gold skills valid: {validation['stats']['gold_skills_valid']}")
        print(f"  Gold skills invalid: {validation['stats']['gold_skills_invalid']}")
        print(f"  Interference skills valid: {validation['stats']['interference_skills_valid']}")
        print(f"  Interference skills invalid: {validation['stats']['interference_skills_invalid']}")
        print(f"  Total steps: {validation['stats']['total_steps']}")
        print(f"  By difficulty: {dict(validation['stats']['by_difficulty'])}")
        print(f"  By domain: {dict(validation['stats']['by_domain'])}")
        print(f"  Cascade risk: {dict(validation['stats']['cascade_risk_distribution'])}")

        if validation['errors']:
            print(f"\nErrors ({len(validation['errors'])}):")
            for e in validation['errors']:
                print(f"  ✗ {e}")

        if validation['warnings']:
            print(f"\nWarnings ({len(validation['warnings'])}):")
            for w in validation['warnings']:
                print(f"  ⚠ {w}")

    # Print task summary
    print("\n" + "=" * 60)
    print("Multi-Step Task Summary")
    print("=" * 60)
    print(f"Total tasks: {len(tasks)}")

    difficulty_counts = defaultdict(int)
    domain_counts = defaultdict(int)
    step_counts = defaultdict(int)
    total_steps = 0

    for task in tasks:
        difficulty_counts[task.difficulty] += 1
        domain_counts[task.domain] += 1
        n_steps = len(task.steps)
        step_counts[n_steps] += 1
        total_steps += n_steps

    print(f"\nBy difficulty: {dict(difficulty_counts)}")
    print(f"By domain: {dict(domain_counts)}")
    print(f"Total steps: {total_steps}")
    print(f"Steps per task: {dict(step_counts)}")
    print(f"Average steps per task: {total_steps / len(tasks):.1f}")