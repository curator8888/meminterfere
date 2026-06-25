"""
MemInterfere: Retrieval vs. Planning Failure Analysis (Phase 5.3)

This module analyzes results from the three experimental tracks to attribute
failures to either retrieval or planning:

  - Track A (Gold Retrieval): Only gold skills in prompt → measures pure planning ability
  - Track B (RAG Retrieval): Top-K retrieved skills → measures retrieval + planning
  - Track C (Full Context): All skills in prompt → baseline

Failure attribution:
  - Planning failure: Agent fails EVEN WITH gold skills in prompt (Track A error)
  - Retrieval failure: Agent fails in Track B but NOT in Track A
    (gold skill was not retrieved, so agent never had a chance)
  - Formula: planning_failures = Track_A_errors
                retrieval_failures = Track_B_errors - Track_A_errors

If Track_A_errors > Track_B_errors, some retrieval errors were "lucky" retrievals
that still failed (net retrieval errors = max(0, Track_B_errors - Track_A_errors))
"""

import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TrackMetrics:
    """Metrics for a single track."""
    track_name: str
    condition_name: str
    num_tasks: int
    selection_accuracy: float  # Fraction where correct skill was chosen
    invocation_accuracy: float  # Fraction where correct skill + correct params
    ece: float  # Expected Calibration Error
    avg_latency_ms: float
    avg_confidence: float
    num_errors: int
    error_distribution: dict  # ErrorType -> count


@dataclass
class FailureAttribution:
    """Attribution of errors to retrieval vs planning."""
    total_tasks: int
    track_a_errors: int
    track_b_errors: int
    track_c_errors: int
    planning_failure_count: int  # = Track A errors (agent fails with gold skills)
    retrieval_failure_count: int  # = max(0, Track B errors - Track A errors)
    planning_failure_rate: float  # fraction of all errors that are planning
    retrieval_failure_rate: float  # fraction of all errors that are retrieval
    # Per-task breakdown
    task_breakdown: list[dict] = field(default_factory=list)


@dataclass
class ModelBreakdown:
    """Per-model breakdown of failure attribution."""
    model: str
    track_a_metrics: TrackMetrics
    track_b_metrics: TrackMetrics  # At best K
    track_c_metrics: TrackMetrics
    failure_attribution: FailureAttribution


# ── Core analysis functions ───────────────────────────────────────────────────

def compute_track_metrics(
    results: list[dict],
    track_name: str,
    condition_name: str,
) -> TrackMetrics:
    """
    Compute aggregate metrics for a single track from RunResult dicts.

    Args:
        results: List of result dicts (from multi_model_runner JSON output).
        track_name: Track identifier (e.g., "gold", "rag_k5", "full").
        condition_name: Condition name.

    Returns:
        TrackMetrics with aggregate statistics.
    """
    if not results:
        return TrackMetrics(
            track_name=track_name,
            condition_name=condition_name,
            num_tasks=0,
            selection_accuracy=0.0,
            invocation_accuracy=0.0,
            ece=0.0,
            avg_latency_ms=0.0,
            avg_confidence=0.0,
            num_errors=0,
            error_distribution={},
        )

    n = len(results)
    correct = sum(1 for r in results if r.get("skill_correct", False))
    selection_accuracy = correct / n

    # Invocation accuracy: correct skill + reasonable confidence
    # (Since we don't have param-level validation, use skill_correct as proxy)
    invocation_accuracy = selection_accuracy  # Could be refined later

    # ECE computation
    confidences = [r.get("parsed_confidence", 0.0) for r in results]
    corrects = [r.get("skill_correct", False) for r in results]
    ece = _compute_ece(confidences, corrects)

    # Latency
    latencies = [r.get("latency_ms", 0) for r in results]
    avg_latency = np.mean(latencies) if latencies else 0.0

    # Confidence
    avg_confidence = np.mean(confidences) if confidences else 0.0

    # Errors
    num_errors = n - correct
    error_dist = {}
    for r in results:
        parse_method = r.get("parse_method", "")
        if not r.get("skill_correct", False):
            error_type = "wrong_skill"
            if parse_method == "parse_error":
                error_type = "parse_error"
            error_dist[error_type] = error_dist.get(error_type, 0) + 1

    return TrackMetrics(
        track_name=track_name,
        condition_name=condition_name,
        num_tasks=n,
        selection_accuracy=round(selection_accuracy, 4),
        invocation_accuracy=round(invocation_accuracy, 4),
        ece=round(ece, 4),
        avg_latency_ms=round(float(avg_latency), 1),
        avg_confidence=round(float(avg_confidence), 4),
        num_errors=num_errors,
        error_distribution=error_dist,
    )


def compute_failure_attribution(
    track_a_results: list[dict],
    track_b_results: list[dict],
    track_c_results: list[dict],
    rag_k: int = 5,
) -> FailureAttribution:
    """
    Compute failure attribution between retrieval and planning.

    Key insight:
      - If the agent fails in Track A (gold retrieval), that's a PLANNING failure.
        The agent had the right skill but chose wrong anyway.
      - If the agent succeeds in Track A but fails in Track B, that's a RETRIEVAL
        failure. The retrieval didn't surface the right skill.
      - If the agent fails in both Track A and Track B, that's a planning failure
        (retrieval didn't matter because even with the gold skill, it failed).

    Args:
        track_a_results: Results from Track A (gold retrieval).
        track_b_results: Results from Track B (RAG top-K).
        track_c_results: Results from Track C (full context).
        rag_k: K value for Track B (for labeling purposes).

    Returns:
        FailureAttribution with per-task and aggregate breakdowns.
    """
    # Index results by task_id for cross-referencing
    a_by_task = {r["task_id"]: r for r in track_a_results}
    b_by_task = {r["task_id"]: r for r in track_b_results}
    c_by_task = {r["task_id"]: r for r in track_c_results}

    all_task_ids = set(a_by_task.keys()) | set(b_by_task.keys()) | set(c_by_task.keys())

    track_a_errors = 0
    track_b_errors = 0
    track_c_errors = 0

    planning_failures = 0
    retrieval_failures = 0

    task_breakdown = []

    for task_id in sorted(all_task_ids):
        a_result = a_by_task.get(task_id)
        b_result = b_by_task.get(task_id)
        c_result = c_by_task.get(task_id)

        a_correct = a_result.get("skill_correct", False) if a_result else None
        b_correct = b_result.get("skill_correct", False) if b_result else None
        c_correct = c_result.get("skill_correct", False) if c_result else None

        a_error = a_result is not None and not a_correct
        b_error = b_result is not None and not b_correct
        c_error = c_result is not None and not c_correct

        if a_error:
            track_a_errors += 1
        if b_error:
            track_b_errors += 1
        if c_error:
            track_c_errors += 1

        # Attribute failure
        if a_error:
            # Agent failed even with gold skill → planning failure
            failure_type = "planning"
            planning_failures += 1
        elif b_error:
            # Agent succeeded with gold skill but failed with RAG retrieval → retrieval failure
            failure_type = "retrieval"
            retrieval_failures += 1
        elif c_error:
            # Agent succeeded in Track B but failed with full context
            # This could be context overload, not retrieval or planning
            failure_type = "context_overload"
        else:
            failure_type = "none"

        task_breakdown.append({
            "task_id": task_id,
            "track_a_correct": a_correct,
            "track_b_correct": b_correct,
            "track_c_correct": c_correct,
            "failure_type": failure_type,
        })

    total_tasks = len(all_task_ids)
    total_errors = track_a_errors + track_b_errors + track_c_errors

    # Net retrieval failures: errors in Track B that aren't explained by Track A
    net_retrieval_failures = max(0, track_b_errors - track_a_errors)

    return FailureAttribution(
        total_tasks=total_tasks,
        track_a_errors=track_a_errors,
        track_b_errors=track_b_errors,
        track_c_errors=track_c_errors,
        planning_failure_count=planning_failures,
        retrieval_failure_count=net_retrieval_failures,
        planning_failure_rate=round(planning_failures / total_errors, 4) if total_errors > 0 else 0.0,
        retrieval_failure_rate=round(net_retrieval_failures / total_errors, 4) if total_errors > 0 else 0.0,
        task_breakdown=task_breakdown,
    )


def compute_model_breakdowns(
    results: list[dict],
    models: list[str] = None,
) -> list[ModelBreakdown]:
    """
    Compute per-model breakdowns of failure attribution.

    Args:
        results: All results from all tracks, with 'model' and 'condition' fields.
        models: List of model names to include (or None for all).

    Returns:
        List of ModelBreakdown objects.
    """
    if models is None:
        models = list(set(r.get("model", "") for r in results))

    breakdowns = []

    # Map condition names to tracks
    track_conditions = {
        "gold_retrieval": "gold",
        "rag_retrieval_k1": "rag_k1",
        "rag_retrieval_k3": "rag_k3",
        "rag_retrieval_k5": "rag_k5",
        "rag_retrieval_k10": "rag_k10",
        "all_memory": "full",
    }

    for model in models:
        model_results = [r for r in results if r.get("model") == model]

        # Get results for each track
        a_results = [r for r in model_results if r.get("condition") == "gold_retrieval"]
        b_results = [r for r in model_results if r.get("condition") == "rag_retrieval_k5"]
        c_results = [r for r in model_results if r.get("condition") == "all_memory"]

        a_metrics = compute_track_metrics(a_results, "gold", "gold_retrieval")
        b_metrics = compute_track_metrics(b_results, "rag_k5", "rag_retrieval_k5")
        c_metrics = compute_track_metrics(c_results, "full", "all_memory")

        attribution = compute_failure_attribution(a_results, b_results, c_results)

        breakdowns.append(ModelBreakdown(
            model=model,
            track_a_metrics=a_metrics,
            track_b_metrics=b_metrics,
            track_c_metrics=c_metrics,
            failure_attribution=attribution,
        ))

    return breakdowns


# ── Comparison table generation ───────────────────────────────────────────────

def generate_comparison_table(
    track_metrics: dict[str, TrackMetrics],
) -> str:
    """
    Generate a formatted comparison table of metrics across tracks.

    Args:
        track_metrics: Dict mapping track names to TrackMetrics.

    Returns:
        Formatted string table.
    """
    lines = []
    lines.append(f"{'Track':<20} {'Condition':<25} {'N':>5} {'Sel.Acc':>9} {'Inv.Acc':>9} "
                 f"{'ECE':>6} {'Lat(ms)':>9} {'Conf':>6} {'Errors':>7}")
    lines.append("-" * 100)

    for name, m in track_metrics.items():
        lines.append(
            f"{m.track_name:<20} {m.condition_name:<25} {m.num_tasks:>5} "
            f"{m.selection_accuracy:>9.4f} {m.invocation_accuracy:>9.4f} "
            f"{m.ece:>6.4f} {m.avg_latency_ms:>9.1f} {m.avg_confidence:>6.4f} "
            f"{m.num_errors:>7}"
        )

    return "\n".join(lines)


def generate_failure_attribution_table(attribution: FailureAttribution) -> str:
    """Generate a formatted failure attribution summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("FAILURE ATTRIBUTION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total tasks:                  {attribution.total_tasks}")
    lines.append(f"Track A errors (planning):    {attribution.track_a_errors}")
    lines.append(f"Track B errors (retr+plan):   {attribution.track_b_errors}")
    lines.append(f"Track C errors (full context): {attribution.track_c_errors}")
    lines.append("")
    lines.append(f"Planning failures:            {attribution.planning_failure_count} "
                 f"({attribution.planning_failure_rate:.1%} of total errors)")
    lines.append(f"Retrieval failures:            {attribution.retrieval_failure_count} "
                 f"({attribution.retrieval_failure_rate:.1%} of total errors)")
    lines.append("")

    # Categorize task breakdown
    categories = {}
    for tb in attribution.task_breakdown:
        cat = tb["failure_type"]
        categories[cat] = categories.get(cat, 0) + 1

    lines.append("Task-level breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        lines.append(f"  {cat}: {count} tasks")

    lines.append("=" * 60)
    return "\n".join(lines)


# ── ECE computation (duplicated from metrics for standalone use) ───────────────

def _compute_ece(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    if not confidences:
        return 0.0

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    for i in range(n_bins):
        in_bin = [j for j, c in enumerate(confidences)
                  if bin_boundaries[i] <= c < bin_boundaries[i + 1]]
        if not in_bin:
            continue
        bin_conf = sum(confidences[j] for j in in_bin) / len(in_bin)
        bin_acc = sum(1 for j in in_bin if corrects[j]) / len(in_bin)
        ece += len(in_bin) * abs(bin_acc - bin_conf)
    ece /= len(confidences)
    return ece


# ── Load results from JSON ─────────────────────────────────────────────────────

def load_results_from_json(filepath: str) -> list[dict]:
    """Load experiment results from a JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    if "results" in data:
        # Multi-model runner format
        return data["results"]
    elif isinstance(data, list):
        return data
    else:
        # Try to extract from any nested structure
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                if "task_id" in val[0] or "skill_correct" in val[0]:
                    return val
        return []


# ── Main analysis pipeline ─────────────────────────────────────────────────────

def run_analysis(
    track_a_path: str = None,
    track_b_path: str = None,
    track_c_path: str = None,
    combined_path: str = None,
    output_dir: str = "data/results/phase5_3",
) -> dict:
    """
    Run the full retrieval vs. planning analysis.

    Either provide separate paths for each track, or a combined results file
    that contains results from all tracks with different 'condition' values.
    """
    os.makedirs(output_dir, exist_ok=True)

    if combined_path:
        all_results = load_results_from_json(combined_path)
        # Split by condition
        track_a_results = [r for r in all_results if r.get("condition") == "gold_retrieval"]
        track_b_results = [r for r in all_results if r.get("condition") == "rag_retrieval_k5"]
        track_c_results = [r for r in all_results if r.get("condition") == "all_memory"]
    elif track_a_path and track_b_path and track_c_path:
        track_a_results = load_results_from_json(track_a_path)
        track_b_results = load_results_from_json(track_b_path)
        track_c_results = load_results_from_json(track_c_path)
    else:
        logger.warning("No results provided. Run retrieval_simulator.py and experiment first.")
        # Return empty analysis
        return {
            "track_metrics": {},
            "failure_attribution": asdict(FailureAttribution(total_tasks=0)),
            "model_breakdowns": [],
        }

    # Compute per-track metrics
    track_metrics = {
        "gold": compute_track_metrics(track_a_results, "gold", "gold_retrieval"),
        "rag_k5": compute_track_metrics(track_b_results, "rag_k5", "rag_retrieval_k5"),
        "full": compute_track_metrics(track_c_results, "full", "all_memory"),
    }

    # Also compute for other K values if available
    all_results_combined = track_a_results + track_b_results + track_c_results
    for k in [1, 3, 10]:
        rag_results = [r for r in all_results_combined if r.get("condition") == f"rag_retrieval_k{k}"]
        if rag_results:
            track_metrics[f"rag_k{k}"] = compute_track_metrics(
                rag_results, f"rag_k{k}", f"rag_retrieval_k{k}"
            )

    # Compute failure attribution
    attribution = compute_failure_attribution(track_a_results, track_b_results, track_c_results)

    # Compute model breakdowns
    model_breakdowns = compute_model_breakdowns(all_results_combined)

    # Generate outputs
    comparison_table = generate_comparison_table(track_metrics)
    attribution_table = generate_failure_attribution_table(attribution)

    # Print results
    print("\n" + comparison_table)
    print("\n" + attribution_table)

    # Per-model breakdowns
    for mb in model_breakdowns:
        print(f"\n--- Model: {mb.model} ---")
        print(f"  Track A (Gold):  Acc={mb.track_a_metrics.selection_accuracy:.4f}, "
              f"Errors={mb.track_a_metrics.num_errors}")
        print(f"  Track B (RAG):   Acc={mb.track_b_metrics.selection_accuracy:.4f}, "
              f"Errors={mb.track_b_metrics.num_errors}")
        print(f"  Track C (Full):  Acc={mb.track_c_metrics.selection_accuracy:.4f}, "
              f"Errors={mb.track_c_metrics.num_errors}")
        print(f"  Planning failures: {mb.failure_attribution.planning_failure_count}, "
              f"Retrieval failures: {mb.failure_attribution.retrieval_failure_count}")

    # Save summary JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "metadata": {
            "timestamp": timestamp,
            "analysis_type": "retrieval_vs_planning",
        },
        "track_metrics": {k: asdict(v) for k, v in track_metrics.items()},
        "failure_attribution": asdict(attribution),
        "model_breakdowns": [
            {
                "model": mb.model,
                "track_a": asdict(mb.track_a_metrics),
                "track_b": asdict(mb.track_b_metrics),
                "track_c": asdict(mb.track_c_metrics),
                "failure_attribution": asdict(mb.failure_attribution),
            }
            for mb in model_breakdowns
        ],
        "comparison_table": comparison_table,
        "attribution_table": attribution_table,
    }

    summary_path = os.path.join(output_dir, f"retrieval_analysis_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved analysis to {summary_path}")

    return summary


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    logging.basicConfig(level=logging.INFO)

    parser = __import__("argparse").ArgumentParser(
        description="Analyze retrieval vs. planning failures from Phase 5.3 results"
    )
    parser.add_argument("--track-a", type=str, help="Path to Track A (gold retrieval) results JSON")
    parser.add_argument("--track-b", type=str, help="Path to Track B (RAG retrieval) results JSON")
    parser.add_argument("--track-c", type=str, help="Path to Track C (full context) results JSON")
    parser.add_argument("--combined", type=str, help="Path to combined results JSON with all tracks")
    parser.add_argument("--output", type=str, default="data/results/phase5_3",
                        help="Output directory for analysis")

    args = parser.parse_args()

    summary = run_analysis(
        track_a_path=args.track_a,
        track_b_path=args.track_b,
        track_c_path=args.track_c,
        combined_path=args.combined,
        output_dir=args.output,
    )

    print(f"\nAnalysis complete. Summary saved to {args.output}")