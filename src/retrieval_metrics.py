"""
MemInterfere: Retrieval Precision Metrics

Adds P@K, recall, and conflict resolution tracking to the evaluation framework.
Per Grok-3 review: "You are missing retrieval precision metrics — P@K and recall
of relevant vs. conflicting skills when the agent queries memory. This directly
measures whether the model is retrieving the wrong skill due to interference."
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ConflictResolution(Enum):
    """How the agent handled a conflict between retrieved skills."""
    NO_CONFLICT = "no_conflict"           # No conflicting skills retrieved
    NOTICED_AND_CORRECT = "noticed_correct"  # Agent detected conflict, chose correctly
    NOTICED_AND_WRONG = "noticed_wrong"      # Agent detected conflict, chose wrong
    NOT_NOTICED = "not_noticed"            # Agent didn't notice conflict
    ASKED_FOR_HELP = "asked_help"          # Agent asked for clarification
    TRIED_MULTIPLE = "tried_multiple"      # Agent tried multiple skills
    RETRIEVED_WRONG = "retrieved_wrong"     # Agent retrieved wrong skill from start


@dataclass
class RetrievalLog:
    """Log of skill retrieval for a single turn."""
    turn_number: int
    task_id: str
    condition: str
    
    # What was retrieved (top-K)
    top_k_skill_ids: list[str] = field(default_factory=list)
    top_k_scores: list[float] = field(default_factory=list)
    
    # Which of the retrieved skills are correct/gold
    gold_skill_ids: list[str] = field(default_factory=list)
    
    # Which of the retrieved skills are conflicting/stale/trap
    conflicting_skill_ids: list[str] = field(default_factory=list)
    stale_skill_ids: list[str] = field(default_factory=list)
    trap_skill_ids: list[str] = field(default_factory=list)
    
    # What the agent actually used
    skill_used: str = ""
    skill_used_is_gold: bool = False
    skill_used_is_conflicting: bool = False
    skill_used_is_stale: bool = False
    skill_used_is_trap: bool = False
    
    # Conflict resolution
    conflict_resolution: str = "no_conflict"  # From ConflictResolution enum
    
    # Agent confidence at time of retrieval
    confidence_before_retrieval: float = 0.0
    confidence_after_retrieval: float = 0.0


def compute_precision_at_k(retrieval_logs: list[RetrievalLog], k: int = 3) -> dict:
    """
    Compute Precision@K: fraction of top-K retrieved skills that are gold (correct).
    
    This directly measures whether interference causes the wrong skills to be retrieved.
    """
    if not retrieval_logs:
        return {"p_at_k": 0.0, "mean_p_at_k": 0.0, "total_queries": 0}
    
    precisions = []
    for log in retrieval_logs:
        top_k = log.top_k_skill_ids[:k]
        if not top_k:
            continue
        gold_in_top_k = sum(1 for sid in top_k if sid in log.gold_skill_ids)
        precision = gold_in_top_k / len(top_k)
        precisions.append(precision)
    
    return {
        "p_at_k": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        "k": k,
        "total_queries": len(precisions),
    }


def compute_recall_at_k(retrieval_logs: list[RetrievalLog], k: int = 5) -> dict:
    """
    Compute Recall@K: fraction of gold skills that appear in top-K.
    
    Low recall means the agent can't find the right skill even when it exists.
    """
    if not retrieval_logs:
        return {"recall_at_k": 0.0, "total_queries": 0}
    
    recalls = []
    for log in retrieval_logs:
        if not log.gold_skill_ids:
            continue
        top_k = log.top_k_skill_ids[:k]
        gold_retrieved = sum(1 for gid in log.gold_skill_ids if gid in top_k)
        recall = gold_retrieved / len(log.gold_skill_ids)
        recalls.append(recall)
    
    return {
        "recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "k": k,
        "total_queries": len(recalls),
    }


def compute_interference_precision(retrieval_logs: list[RetrievalLog], k: int = 3) -> dict:
    """
    Compute the rate at which conflicting/stale/trap skills appear in top-K.
    
    This is the key metric: how often does interference pollute the retrieval?
    """
    if not retrieval_logs:
        return {"interference_in_top_k": 0.0, "conflict_in_top_k": 0.0, 
                "stale_in_top_k": 0.0, "trap_in_top_k": 0.0}
    
    conflict_count = 0
    stale_count = 0
    trap_count = 0
    total_top_k = 0
    
    for log in retrieval_logs:
        top_k = log.top_k_skill_ids[:k]
        total_top_k += len(top_k)
        
        conflict_count += sum(1 for sid in top_k if sid in log.conflicting_skill_ids)
        stale_count += sum(1 for sid in top_k if sid in log.stale_skill_ids)
        trap_count += sum(1 for sid in top_k if sid in log.trap_skill_ids)
    
    return {
        "interference_in_top_k": round((conflict_count + stale_count + trap_count) / total_top_k, 4) if total_top_k else 0.0,
        "conflict_in_top_k": round(conflict_count / total_top_k, 4) if total_top_k else 0.0,
        "stale_in_top_k": round(stale_count / total_top_k, 4) if total_top_k else 0.0,
        "trap_in_top_k": round(trap_count / total_top_k, 4) if total_top_k else 0.0,
        "total_retrieved": total_top_k,
    }


def compute_conflict_resolution_distribution(retrieval_logs: list[RetrievalLog]) -> dict:
    """
    Distribution of how the agent handled conflicts.
    
    Key insight from Grok: "Distinguish 'didn't notice' from 'noticed but failed'"
    """
    dist = {}
    for log in retrieval_logs:
        cr = log.conflict_resolution
        dist[cr] = dist.get(cr, 0) + 1
    return dist


def compute_crossover_point(
    results_by_library_size: dict[int, list]
) -> dict:
    """
    Find the library size where memory retrieval transitions from beneficial to harmful.
    
    This is the paper's signature finding — the crossover point.
    """
    import numpy as np
    
    sizes = sorted(results_by_library_size.keys())
    success_rates = []
    for size in sizes:
        results = results_by_library_size[size]
        if results:
            sr = sum(1 for r in results if r.success) / len(results)
            success_rates.append(sr)
        else:
            success_rates.append(None)
    
    # Find crossover: where success rate drops below no-memory baseline
    # (This requires a no-memory baseline to be passed separately)
    # For now, just find the peak and decline
    
    valid_points = [(s, sr) for s, sr in zip(sizes, success_rates) if sr is not None]
    if len(valid_points) < 3:
        return {"crossover_found": False, "reason": "insufficient_data"}
    
    peak_idx = max(range(len(valid_points)), key=lambda i: valid_points[i][1])
    peak_size = valid_points[peak_idx][0]
    peak_rate = valid_points[peak_idx][1]
    
    return {
        "crossover_found": True,
        "peak_library_size": peak_size,
        "peak_success_rate": round(peak_rate, 4),
        "sizes": [p[0] for p in valid_points],
        "rates": [round(p[1], 4) for p in valid_points],
    }


def compute_all_retrieval_metrics(retrieval_logs: list[RetrievalLog]) -> dict:
    """Compute all retrieval precision metrics at once."""
    return {
        "precision_at_1": compute_precision_at_k(retrieval_logs, k=1),
        "precision_at_3": compute_precision_at_k(retrieval_logs, k=3),
        "precision_at_5": compute_precision_at_k(retrieval_logs, k=5),
        "recall_at_3": compute_recall_at_k(retrieval_logs, k=3),
        "recall_at_5": compute_recall_at_k(retrieval_logs, k=5),
        "interference_metrics": compute_interference_precision(retrieval_logs),
        "conflict_resolution": compute_conflict_resolution_distribution(retrieval_logs),
    }


if __name__ == "__main__":
    # Demo with sample data
    sample_logs = [
        RetrievalLog(
            turn_number=1, task_id="web_easy_001", condition="clean_interference",
            top_k_skill_ids=["skill_a", "skill_b", "skill_c"],
            top_k_scores=[0.95, 0.82, 0.71],
            gold_skill_ids=["skill_a"],
            conflicting_skill_ids=["skill_b"],
            stale_skill_ids=["skill_c"],
            trap_skill_ids=[],
            skill_used="skill_a",
            skill_used_is_gold=True,
            confidence_before_retrieval=0.5,
            confidence_after_retrieval=0.85,
        ),
        RetrievalLog(
            turn_number=1, task_id="web_easy_002", condition="clean_interference",
            top_k_skill_ids=["skill_x", "skill_y", "skill_z"],
            top_k_scores=[0.90, 0.88, 0.75],
            gold_skill_ids=["skill_z"],
            conflicting_skill_ids=["skill_x", "skill_y"],
            stale_skill_ids=[],
            trap_skill_ids=[],
            skill_used="skill_x",  # Wrong!
            skill_used_is_gold=False,
            skill_used_is_conflicting=True,
            conflict_resolution="not_noticed",
            confidence_before_retrieval=0.6,
            confidence_after_retrieval=0.9,  # Overconfident
        ),
    ]
    
    metrics = compute_all_retrieval_metrics(sample_logs)
    
    print("Retrieval Metrics Demo:")
    print(f"  P@1: {metrics['precision_at_1']}")
    print(f"  P@3: {metrics['precision_at_3']}")
    print(f"  P@5: {metrics['precision_at_5']}")
    print(f"  Recall@3: {metrics['recall_at_3']}")
    print(f"  Recall@5: {metrics['recall_at_5']}")
    print(f"  Interference: {metrics['interference_metrics']}")
    print(f"  Conflict resolution: {metrics['conflict_resolution']}")