"""
MemInterfere: Evaluation Runner

Runs the agent under controlled conditions and collects metrics.
This is the Phase 2 framework — actual agent execution will be
done in Phase 3 with a real agent.
"""

import json
import os
import time
import random
from dataclasses import asdict
from datetime import datetime

# Import from our modules
import sys
sys.path.insert(0, os.path.dirname(__file__))

from interference_library import (
    ALL_SKILLS, get_skills_by_type, get_library_stats, validate_library,
    Skill, ConflictType, Domain, StalenessLevel
)
from metrics import (
    EvalTask, EvalResult, TurnLog, Condition, ErrorType,
    EVAL_TASKS, get_task_stats, get_tasks_by_difficulty,
    compute_ece, compute_success_rate, compute_partial_credit_rate,
    compute_error_distribution, compute_tokens_per_success, compute_degradation_slope
)


def get_library_for_condition(condition: Condition, session: int = 0) -> list[Skill]:
    """Get the skill library for a given experimental condition."""
    clean = get_skills_by_type("clean")
    interference = get_skills_by_type("interference")
    stale = get_skills_by_type("stale")
    trap = get_skills_by_type("trap")
    
    if condition == Condition.ORACLE:
        return clean  # Only clean skills, but gold one always provided
    elif condition == Condition.NO_MEMORY:
        return []
    elif condition == Condition.CLEAN_MEMORY:
        return clean
    elif condition == Condition.CLEAN_INTERFERENCE:
        return clean + interference + trap
    elif condition == Condition.CLEAN_STALE:
        return clean + stale
    elif condition == Condition.ALL_MEMORY:
        return clean + interference + stale + trap
    elif condition == Condition.GROWING:
        # Start with 10 clean, add 5 per session
        base = clean[:10]
        additions_per_session = 5
        all_additions = interference[:5] + stale[:5] + list(reversed(interference[5:10])) + list(reversed(stale[5:]))
        num_to_add = min(session * additions_per_session, len(all_additions))
        return base + all_additions[:num_to_add]
    else:
        raise ValueError(f"Unknown condition: {condition}")


def simulate_agent_turn(task: EvalTask, library: list[Skill], condition: Condition) -> TurnLog:
    """
    Simulate a single agent turn (placeholder for real agent execution).
    
    In Phase 3, this will be replaced with actual Hermes agent execution.
    For now, returns a mock turn log based on library characteristics.
    """
    # This is a simulation — Phase 3 will use the real agent
    library_size = len(library)
    interference_count = sum(1 for s in library if s.conflict_type != "none" and not s.is_trap and not s.is_stale)
    stale_count = sum(1 for s in library if s.is_stale)
    trap_count = sum(1 for s in library if s.is_trap)
    
    # Find matching skills in library
    matching_skills = [s for s in library if s.name in task.expected_skill_ids or 
                       s.skill_id in task.expected_skill_ids]
    
    # Determine if the correct skill is available
    correct_available = len(matching_skills) > 0
    
    # Base success probability
    base_success = 0.9 if task.difficulty == "easy" else 0.7 if task.difficulty == "medium" else 0.5
    
    # Interference effects (these will be measured empirically in Phase 3)
    interference_penalty = interference_count * 0.02  # 2% per interfering skill
    stale_penalty = stale_count * 0.03  # 3% per stale skill
    trap_risk = trap_count * 0.05  # 5% per trap skill
    
    if condition == Condition.NO_MEMORY:
        success_prob = base_success * 0.6 if task.requires_memory else base_success
    elif condition == Condition.ORACLE:
        success_prob = base_success  # Always gets the right skill
    elif condition == Condition.CLEAN_MEMORY:
        success_prob = base_success
    else:
        success_prob = max(0.1, base_success - interference_penalty - stale_penalty - trap_risk)
    
    # Simulate confidence (in Phase 3, this comes from the real agent)
    if condition == Condition.NO_MEMORY:
        confidence = 0.3 if task.requires_memory else 0.7
    elif condition == Condition.CLEAN_MEMORY:
        confidence = 0.85
    elif condition in (Condition.CLEAN_INTERFERENCE, Condition.CLEAN_STALE, Condition.ALL_MEMORY):
        confidence = 0.75 + random.uniform(-0.15, 0.15)  # Overconfident
    else:
        confidence = 0.8
    
    # Determine error type
    success = random.random() < success_prob
    if success:
        error_type = ErrorType.SUCCESS.value
    elif trap_count > 0 and random.random() < trap_risk:
        error_type = ErrorType.TRAP_ACCEPTED.value
    elif stale_count > 0 and random.random() < 0.3:
        error_type = ErrorType.STALE_API_CALL.value
    elif interference_count > 0 and random.random() < 0.4:
        error_type = ErrorType.WRONG_SKILL.value
    elif task.requires_memory and condition == Condition.NO_MEMORY:
        error_type = ErrorType.RETRIEVAL_OMISSION.value
    else:
        error_type = ErrorType.PARAMETER_HALLUCINATION.value
    
    # Simulate tokens
    base_tokens = 500 if task.difficulty == "easy" else 1000 if task.difficulty == "medium" else 2000
    if condition != Condition.NO_MEMORY:
        base_tokens += library_size * 50  # Each skill adds context
    
    return TurnLog(
        turn_number=1,
        task_id=task.task_id,
        condition=condition.value,
        seed=0,
        retrieved_skill_ids=[s.skill_id for s in random.sample(library, min(3, len(library)))] if library else [],
        retrieval_scores=[random.uniform(0.5, 0.95) for _ in range(min(3, len(library)))] if library else [],
        agent_confidence=round(confidence, 3),
        skill_used=matching_skills[0].skill_id if matching_skills else "",
        skill_correct=correct_available and success,
        execution_result="success" if success else "failure",
        error_type=error_type,
        tokens_used=base_tokens + random.randint(-200, 200),
        retrieval_calls=1 if library else 0,
        response_time_ms=random.randint(1000, 5000),
    )


def run_evaluation(
    conditions: list[Condition] = None,
    tasks: list[EvalTask] = None,
    seeds: list[int] = None,
    output_dir: str = "data/results"
) -> dict:
    """
    Run evaluation across conditions, tasks, and seeds.
    
    Returns a dictionary of results keyed by condition.
    """
    if conditions is None:
        conditions = list(Condition)
    if tasks is None:
        tasks = EVAL_TASKS
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]  # 5 seeds as per GLM-5.1 recommendation
    
    all_results = {}
    
    for condition in conditions:
        condition_results = []
        for seed in seeds:
            random.seed(seed)
            for task in tasks:
                # Get library for this condition
                library = get_library_for_condition(condition, session=0)
                
                # Simulate agent turn
                turn = simulate_agent_turn(task, library, condition)
                
                # Create result
                result = EvalResult(
                    task_id=task.task_id,
                    condition=condition.value,
                    seed=seed,
                    success=turn.skill_correct and turn.error_type == ErrorType.SUCCESS.value,
                    partial_credit=1.0 if turn.skill_correct else 0.5 if turn.execution_result == "partial" else 0.0,
                    error_type=turn.error_type,
                    confidence=turn.agent_confidence,
                    tokens_used=turn.tokens_used,
                    retrieval_calls=turn.retrieval_calls,
                    turns=[turn],
                    library_size_at_time=len(library),
                    interference_skills_available=sum(1 for s in library if s.conflict_type != "none"),
                )
                condition_results.append(result)
        
        all_results[condition.value] = condition_results
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    
    serializable = {}
    for cond, results in all_results.items():
        serializable[cond] = [asdict(r) for r in results]
    
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "conditions": [c.value for c in conditions],
                "num_tasks": len(tasks),
                "seeds": seeds,
                "note": "PHASE 2 SIMULATION - not real agent results"
            },
            "results": serializable
        }, f, indent=2)
    
    print(f"Saved results to {output_path}")
    return all_results


def analyze_results(results: dict) -> dict:
    """Analyze evaluation results and compute all metrics."""
    analysis = {}
    
    for condition, condition_results in results.items():
        success_rate = compute_success_rate(condition_results)
        partial_credit = compute_partial_credit_rate(condition_results)
        error_dist = compute_error_distribution(condition_results)
        tokens_per = compute_tokens_per_success(condition_results)
        
        # Compute calibration
        confidences = [r.confidence for r in condition_results]
        corrects = [r.success for r in condition_results]
        ece = compute_ece(confidences, corrects)
        
        analysis[condition] = {
            "success_rate": round(success_rate, 4),
            "partial_credit_rate": round(partial_credit, 4),
            "ece": round(ece, 4),
            "tokens_per_success": round(tokens_per, 1) if tokens_per != float('inf') else None,
            "error_distribution": error_dist,
            "num_results": len(condition_results),
        }
    
    return analysis


if __name__ == "__main__":
    print("=" * 60)
    print("MemInterfere Phase 2: Skill Library + Evaluation Framework")
    print("=" * 60)
    
    # Validate library
    errors = validate_library()
    if errors:
        print("\nLibrary validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n✓ Skill library validated successfully")
    
    # Print library stats
    stats = get_library_stats()
    print(f"\nSkill Library:")
    print(f"  Total: {stats['total']}")
    print(f"  Clean: {stats['clean']}")
    print(f"  Interference: {stats['interference']}")
    print(f"  Stale: {stats['stale']}")
    print(f"  Trap: {stats['trap']}")
    
    # Print task stats
    task_stats = get_task_stats()
    print(f"\nEvaluation Tasks:")
    print(f"  Total: {task_stats['total']}")
    print(f"  By difficulty: {task_stats['by_difficulty']}")
    print(f"  By interference: {task_stats['by_interference']}")
    
    # Run simulation (Phase 2 placeholder)
    print(f"\nRunning Phase 2 simulation (NOT real agent results)...")
    results = run_evaluation(
        conditions=[Condition.NO_MEMORY, Condition.CLEAN_MEMORY, Condition.CLEAN_INTERFERENCE, Condition.ALL_MEMORY],
        tasks=EVAL_TASKS[:20],  # Use subset for simulation
        seeds=[42]
    )
    
    analysis = analyze_results(results)
    print(f"\nPhase 2 Simulation Results (placeholder):")
    for condition, metrics in analysis.items():
        print(f"\n  {condition}:")
        print(f"    Success rate: {metrics['success_rate']}")
        print(f"    ECE: {metrics['ece']}")
        print(f"    Tokens/success: {metrics['tokens_per_success']}")
    
    print(f"\n{'=' * 60}")
    print("Phase 2 complete. Ready for Phase 3: Real agent evaluation.")
    print("Next steps:")
    print("  1. Implement real agent execution in evaluate_agent.py")
    print("  2. Add instrumentation layer to Hermes")
    print("  3. Run pilot with 20 tasks across all conditions")