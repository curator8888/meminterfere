"""
MemInterfere: Cascade Metrics (Phase 5.4)

Statistical analysis for cascade error effects in multi-step agentic workflows.

The key research question: Does a wrong tool at step N cause errors at step N+1?
We test this by comparing:
- P(wrong step_N | wrong step_N-1) — conditional cascade rate
- P(wrong step_N | correct step_N-1) — clean baseline rate

If cascade_rate_conditional > cascade_rate_clean, errors propagate (cascade effect).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict
import math

# Import StepResult and MultiStepResult from sibling module
from multistep_evaluator import StepResult, MultiStepResult


# ── Data Classes for Cascade Analysis ────────────────────────────────────────

@dataclass
class CascadeAnalysis:
    """Results of cascade analysis across all multi-step evaluations."""
    # Overall cascade rates
    cascade_rate_conditional: float  # P(wrong_N | wrong_N-1)
    cascade_rate_clean: float  # P(wrong_N | correct_N-1)
    cascade_effect_size: float  # conditional - clean
    cascade_ratio: float  # conditional / clean (infinity if clean=0)

    # Counts
    wrong_after_wrong: int  # Number of steps wrong where previous was also wrong
    wrong_after_correct: int  # Number of steps wrong where previous was correct
    correct_after_wrong: int  # Number of steps correct where previous was wrong
    correct_after_correct: int  # Number of steps correct where previous was correct

    # Statistical significance
    fisher_exact_p: Optional[float] = None
    fisher_exact_significant: Optional[bool] = None

    # Per-position cascade
    position_cascade: dict = field(default_factory=dict)

    # Per-difficulty cascade
    difficulty_cascade: dict = field(default_factory=dict)

    # Per-domain cascade
    domain_cascade: dict = field(default_factory=dict)


# ── Core Cascade Computation ─────────────────────────────────────────────────

def compute_cascade_rate(step_results: list[StepResult]) -> dict:
    """
    Compute P(wrong_N | wrong_N-1) vs P(wrong_N | correct_N-1).

    This is the core cascade metric. We look at consecutive step pairs
    (step N-1, step N) and compute conditional error probabilities.

    Args:
        step_results: All StepResult objects from evaluation

    Returns:
        Dict with cascade_rate_conditional, cascade_rate_clean, counts, etc.
    """
    # Group steps by (task_id, condition, model, temperature) to get sequences
    sequences = defaultdict(list)
    for sr in step_results:
        key = (sr.task_id, sr.condition, sr.model, sr.temperature)
        sequences[key].append(sr)

    # Sort each sequence by step_number
    for key in sequences:
        sequences[key].sort(key=lambda s: s.step_number)

    # Compute conditional probabilities
    wrong_after_wrong = 0  # step N wrong, step N-1 wrong
    wrong_after_correct = 0  # step N wrong, step N-1 correct
    correct_after_wrong = 0  # step N correct, step N-1 wrong
    correct_after_correct = 0  # step N correct, step N-1 correct

    for key, steps in sequences.items():
        for i in range(1, len(steps)):
            prev_correct = steps[i - 1].skill_correct
            curr_correct = steps[i].skill_correct

            if prev_correct and curr_correct:
                correct_after_correct += 1
            elif prev_correct and not curr_correct:
                wrong_after_correct += 1
            elif not prev_correct and curr_correct:
                correct_after_wrong += 1
            else:  # not prev_correct and not curr_correct
                wrong_after_wrong += 1

    # Compute rates
    total_after_wrong = wrong_after_wrong + correct_after_wrong
    total_after_correct = wrong_after_correct + correct_after_correct

    cascade_rate_conditional = (
        wrong_after_wrong / total_after_wrong if total_after_wrong > 0 else 0.0
    )
    cascade_rate_clean = (
        wrong_after_correct / total_after_correct if total_after_correct > 0 else 0.0
    )

    cascade_effect_size = cascade_rate_conditional - cascade_rate_clean
    cascade_ratio = (
        cascade_rate_conditional / cascade_rate_clean
        if cascade_rate_clean > 0
        else float('inf') if cascade_rate_conditional > 0 else 0.0
    )

    return {
        "cascade_rate_conditional": cascade_rate_conditional,
        "cascade_rate_clean": cascade_rate_clean,
        "cascade_effect_size": cascade_effect_size,
        "cascade_ratio": cascade_ratio,
        "wrong_after_wrong": wrong_after_wrong,
        "wrong_after_correct": wrong_after_correct,
        "correct_after_wrong": correct_after_wrong,
        "correct_after_correct": correct_after_correct,
        "total_pairs": wrong_after_wrong + wrong_after_correct +
                       correct_after_wrong + correct_after_correct,
    }


def compute_cascade_rate_by_position(step_results: list[StepResult]) -> dict:
    """
    Compute cascade rates broken down by step position.
    E.g., for position 2: P(wrong_2 | wrong_1) vs P(wrong_2 | correct_1)
    """
    # Group steps by position
    position_pairs = defaultdict(lambda: {
        "wrong_after_wrong": 0,
        "wrong_after_correct": 0,
        "correct_after_wrong": 0,
        "correct_after_correct": 0,
    })

    # Group steps by sequence
    sequences = defaultdict(list)
    for sr in step_results:
        key = (sr.task_id, sr.condition, sr.model, sr.temperature)
        sequences[key].append(sr)

    for key, steps in sequences.items():
        steps.sort(key=lambda s: s.step_number)
        for i in range(1, len(steps)):
            position = steps[i].step_number
            prev_correct = steps[i - 1].skill_correct
            curr_correct = steps[i].skill_correct

            if prev_correct and curr_correct:
                position_pairs[position]["correct_after_correct"] += 1
            elif prev_correct and not curr_correct:
                position_pairs[position]["wrong_after_correct"] += 1
            elif not prev_correct and curr_correct:
                position_pairs[position]["correct_after_wrong"] += 1
            else:
                position_pairs[position]["wrong_after_wrong"] += 1

    # Compute rates per position
    result = {}
    for pos, counts in sorted(position_pairs.items()):
        total_after_wrong = counts["wrong_after_wrong"] + counts["correct_after_wrong"]
        total_after_correct = counts["wrong_after_correct"] + counts["correct_after_correct"]

        cond_rate = counts["wrong_after_wrong"] / total_after_wrong if total_after_wrong > 0 else 0.0
        clean_rate = counts["wrong_after_correct"] / total_after_correct if total_after_correct > 0 else 0.0

        result[pos] = {
            "cascade_rate_conditional": cond_rate,
            "cascade_rate_clean": clean_rate,
            "effect_size": cond_rate - clean_rate,
            "total_after_wrong": total_after_wrong,
            "total_after_correct": total_after_correct,
        }

    return result


# ── Position Accuracy ─────────────────────────────────────────────────────────

def compute_position_accuracy(step_results: list[StepResult]) -> dict:
    """
    Compute accuracy at each step position (1, 2, 3, ...).
    Also reports overall accuracy across all positions.
    """
    position_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for sr in step_results:
        position_stats[sr.step_number]["total"] += 1
        if sr.skill_correct:
            position_stats[sr.step_number]["correct"] += 1

    result = {}
    total_correct = 0
    total_count = 0

    for pos in sorted(position_stats.keys()):
        correct = position_stats[pos]["correct"]
        total = position_stats[pos]["total"]
        accuracy = correct / total if total > 0 else 0.0
        result[pos] = {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
        }
        total_correct += correct
        total_count += total

    result["overall"] = {
        "accuracy": round(total_correct / total_count, 4) if total_count > 0 else 0.0,
        "correct": total_correct,
        "total": total_count,
    }

    return result


# ── Sequence Accuracy ─────────────────────────────────────────────────────────

def compute_sequence_accuracy(multi_results: list[MultiStepResult]) -> float:
    """
    Compute the fraction of tasks where ALL steps are correct.
    This is the strict metric — any single error breaks the sequence.
    """
    if not multi_results:
        return 0.0

    correct_sequences = sum(1 for r in multi_results if r.sequence_accuracy)
    return correct_sequences / len(multi_results)


def compute_sequence_accuracy_by_group(
    multi_results: list[MultiStepResult],
    group_key: str,
) -> dict:
    """
    Compute sequence accuracy broken down by a grouping key.

    group_key can be 'condition', 'model', 'difficulty', or 'domain'.
    """
    groups = defaultdict(list)

    for r in multi_results:
        if group_key == "condition":
            groups[r.condition].append(r)
        elif group_key == "model":
            groups[r.model].append(r)
        elif group_key == "difficulty":
            # Infer from task_id
            task_num = int(r.task_id.split("_")[1])
            if task_num <= 10:
                groups["easy"].append(r)
            elif task_num <= 20:
                groups["medium"].append(r)
            else:
                groups["hard"].append(r)
        elif group_key == "domain":
            # Mixed by default in multi-step
            groups["mixed"].append(r)

    result = {}
    for group_name, group_results in groups.items():
        result[group_name] = compute_sequence_accuracy(group_results)

    return result


# ── Parameter Accuracy ────────────────────────────────────────────────────────

def compute_parameter_accuracy(step_results: list[StepResult]) -> dict:
    """
    Compute parameter correctness at each step, separate from selection accuracy.
    A step is parameter-correct if both the skill AND its parameters are correct.
    """
    position_stats = defaultdict(lambda: {
        "skill_correct": 0,
        "param_correct": 0,
        "both_correct": 0,
        "total": 0,
    })

    for sr in step_results:
        pos = sr.step_number
        position_stats[pos]["total"] += 1
        if sr.skill_correct:
            position_stats[pos]["skill_correct"] += 1
        if sr.parameters_correct:
            position_stats[pos]["param_correct"] += 1
        if sr.skill_correct and sr.parameters_correct:
            position_stats[pos]["both_correct"] += 1

    result = {}
    for pos in sorted(position_stats.keys()):
        stats = position_stats[pos]
        total = stats["total"]
        result[pos] = {
            "skill_accuracy": round(stats["skill_correct"] / total, 4) if total > 0 else 0.0,
            "param_accuracy": round(stats["param_correct"] / total, 4) if total > 0 else 0.0,
            "joint_accuracy": round(stats["both_correct"] / total, 4) if total > 0 else 0.0,
            "total": total,
        }

    # Overall
    total = sum(s["total"] for s in position_stats.values())
    result["overall"] = {
        "skill_accuracy": round(sum(s["skill_correct"] for s in position_stats.values()) / total, 4) if total > 0 else 0.0,
        "param_accuracy": round(sum(s["param_correct"] for s in position_stats.values()) / total, 4) if total > 0 else 0.0,
        "joint_accuracy": round(sum(s["both_correct"] for s in position_stats.values()) / total, 4) if total > 0 else 0.0,
        "total": total,
    }

    return result


# ── Interference Selection Rate ──────────────────────────────────────────────

def compute_interference_selection_rate(step_results: list[StepResult]) -> dict:
    """
    Compute how often the agent picks an interference skill over the gold skill.
    Broken down by step position.
    """
    position_stats = defaultdict(lambda: {
        "interference_selected": 0,
        "gold_selected": 0,
        "other_selected": 0,
        "total": 0,
    })

    for sr in step_results:
        pos = sr.step_number
        position_stats[pos]["total"] += 1

        if sr.selected_interference:
            position_stats[pos]["interference_selected"] += 1
        elif sr.skill_correct:
            position_stats[pos]["gold_selected"] += 1
        else:
            position_stats[pos]["other_selected"] += 1

    result = {}
    total_interference = 0
    total_steps = 0

    for pos in sorted(position_stats.keys()):
        stats = position_stats[pos]
        total = stats["total"]
        result[pos] = {
            "interference_rate": round(stats["interference_selected"] / total, 4) if total > 0 else 0.0,
            "gold_rate": round(stats["gold_selected"] / total, 4) if total > 0 else 0.0,
            "other_rate": round(stats["other_selected"] / total, 4) if total > 0 else 0.0,
            "total": total,
        }
        total_interference += stats["interference_selected"]
        total_steps += total

    result["overall"] = {
        "interference_rate": round(total_interference / total_steps, 4) if total_steps > 0 else 0.0,
        "total_interference": total_interference,
        "total_steps": total_steps,
    }

    return result


# ── Fisher's Exact Test ───────────────────────────────────────────────────────

def compute_cascade_significance(step_results: list[StepResult]) -> dict:
    """
    Compute Fisher's exact test for cascade effect significance.

    Tests whether P(wrong_N | wrong_N-1) is significantly different from
    P(wrong_N | correct_N-1).

    The 2x2 contingency table is:
                    Step N Wrong    Step N Correct
    Step N-1 Wrong      a                b
    Step N-1 Correct     c                d

    Where:
    a = wrong_after_wrong
    b = correct_after_wrong
    c = wrong_after_correct
    d = correct_after_correct
    """
    cascade_data = compute_cascade_rate(step_results)

    a = cascade_data["wrong_after_wrong"]
    b = cascade_data["correct_after_wrong"]
    c = cascade_data["wrong_after_correct"]
    d = cascade_data["correct_after_correct"]

    if a + b + c + d == 0:
        return {
            "fisher_exact_p": None,
            "significant": False,
            "note": "No step pairs available for analysis",
            "contingency_table": {"a": 0, "b": 0, "c": 0, "d": 0},
        }

    # Compute Fisher's exact test
    try:
        p_value = _fisher_exact(a, b, c, d)
    except Exception:
        p_value = None

    result = {
        "fisher_exact_p": p_value,
        "significant": p_value is not None and p_value < 0.05,
        "contingency_table": {
            "wrong_after_wrong": a,
            "correct_after_wrong": b,
            "wrong_after_correct": c,
            "correct_after_correct": d,
        },
        "cascade_rate_conditional": cascade_data["cascade_rate_conditional"],
        "cascade_rate_clean": cascade_data["cascade_rate_clean"],
        "effect_size": cascade_data["cascade_effect_size"],
        "total_pairs": cascade_data["total_pairs"],
    }

    return result


def _fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """
    Compute Fisher's exact test for a 2x2 contingency table.

    Uses scipy if available, otherwise falls back to a simple approximation.
    """
    try:
        from scipy.stats import fisher_exact
        _, p_value = fisher_exact([[a, b], [c, d]])
        return float(p_value)
    except ImportError:
        # Fallback: use chi-squared approximation with Yates' correction
        # This is less accurate for small samples but doesn't require scipy
        n = a + b + c + d
        if n == 0:
            return 1.0

        # Yates' corrected chi-squared
        ad_bc = abs(a * d - b * c)
        denom = (a + b) * (c + d) * (a + c) * (b + d)
        if denom == 0:
            return 1.0

        chi2 = n * (ad_bc - n / 2) ** 2 / denom if ad_bc > n / 2 else 0.0

        # Approximate p-value from chi-squared with 1 df
        # Using the approximation: p ≈ exp(-chi2/2)
        if chi2 <= 0:
            return 1.0
        p_value = math.exp(-chi2 / 2)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, p_value))


# ── Comprehensive Cascade Analysis ────────────────────────────────────────────

def compute_full_cascade_analysis(
    step_results: list[StepResult],
    multi_results: list[MultiStepResult],
) -> CascadeAnalysis:
    """
    Compute comprehensive cascade analysis across all evaluation results.

    Returns a CascadeAnalysis object with all metrics.
    """
    # Overall cascade rates
    cascade_data = compute_cascade_rate(step_results)

    # Statistical significance
    significance = compute_cascade_significance(step_results)

    # Per-position cascade
    position_cascade = compute_cascade_rate_by_position(step_results)

    # Position accuracy
    position_accuracy = compute_position_accuracy(step_results)

    # Sequence accuracy
    seq_accuracy = compute_sequence_accuracy(multi_results)

    # Parameter accuracy
    param_accuracy = compute_parameter_accuracy(step_results)

    # Interference selection rate
    interference_rate = compute_interference_selection_rate(step_results)

    # Build result
    analysis = CascadeAnalysis(
        cascade_rate_conditional=cascade_data["cascade_rate_conditional"],
        cascade_rate_clean=cascade_data["cascade_rate_clean"],
        cascade_effect_size=cascade_data["cascade_effect_size"],
        cascade_ratio=cascade_data["cascade_ratio"],
        wrong_after_wrong=cascade_data["wrong_after_wrong"],
        wrong_after_correct=cascade_data["wrong_after_correct"],
        correct_after_wrong=cascade_data["correct_after_wrong"],
        correct_after_correct=cascade_data["correct_after_correct"],
        fisher_exact_p=significance.get("fisher_exact_p"),
        fisher_exact_significant=significance.get("significant"),
        position_cascade=position_cascade,
    )

    return analysis


def cascade_analysis_to_dict(analysis: CascadeAnalysis) -> dict:
    """Convert a CascadeAnalysis to a serializable dict."""
    return {
        "cascade_rate_conditional": analysis.cascade_rate_conditional,
        "cascade_rate_clean": analysis.cascade_rate_clean,
        "cascade_effect_size": analysis.cascade_effect_size,
        "cascade_ratio": analysis.cascade_ratio,
        "wrong_after_wrong": analysis.wrong_after_wrong,
        "wrong_after_correct": analysis.wrong_after_correct,
        "correct_after_wrong": analysis.correct_after_wrong,
        "correct_after_correct": analysis.correct_after_correct,
        "fisher_exact_p": analysis.fisher_exact_p,
        "fisher_exact_significant": analysis.fisher_exact_significant,
        "position_cascade": analysis.position_cascade,
        "difficulty_cascade": analysis.difficulty_cascade,
        "domain_cascade": analysis.domain_cascade,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("MemInterfere Cascade Metrics Module")
    print("=" * 60)
    print()
    print("This module provides statistical analysis for cascade error effects.")
    print()
    print("Key functions:")
    print("  compute_cascade_rate()        - P(wrong_N | wrong_N-1) vs P(wrong_N | correct_N-1)")
    print("  compute_position_accuracy()    - Accuracy at each step position")
    print("  compute_sequence_accuracy()    - Fraction of tasks with ALL steps correct")
    print("  compute_parameter_accuracy()   - Parameter correctness separate from selection")
    print("  compute_interference_selection_rate() - How often agent picks interference skill")
    print("  compute_cascade_significance() - Fisher's exact test for cascade effect")
    print("  compute_full_cascade_analysis() - Comprehensive analysis")
    print()
    print("Usage:")
    print("  from cascade_metrics import *")
    print("  analysis = compute_full_cascade_analysis(step_results, multi_results)")