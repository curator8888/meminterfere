# Phase 3: Expanded Tasks, Retrieval Metrics, and Growing Condition

**Date:** 2026-06-25
**Status:** Complete — reviewed by Grok-3

## Changes from Phase 2.5

### 3.1: Task Expansion (47 → 80)
- Added 15 natural interference tasks (tasks where interference emerges organically)
- Added 10 hard multi-step composition tasks (3+ skills)
- Added 8 medium difficulty tasks
- Distribution: 33 easy, 29 medium, 18 hard
- Natural interference: 15 (was 0)
- High interference: 49/80 tasks

### 3.2: Retrieval Precision Metrics
New file: `src/retrieval_metrics.py`
- **P@K (1, 3, 5)**: Fraction of gold skills in top-K retrieved
- **Recall@K (3, 5)**: Fraction of gold skills that appear in top-K
- **Interference precision**: Rate of conflicting/stale/trap skills in top-K
- **Conflict resolution distribution**: How agent handles conflicts (noticed_correct, noticed_wrong, not_noticed, asked_help, tried_multiple)
- **Crossover point computation**: Library size where memory becomes harmful
- **Harmful retrieval rate**: % of top-K that are interference/stale/trap (Grok suggestion)

### 3.3: Growing Condition Fixed
Was: add 5 clean skills per session (unrealistic)
Now: realistic growth schedule:
- Sessions 1-2: 3 clean + 1 stale + 1 interference
- Sessions 3-5: 2 clean + 1 stale + 2 interference + 1 trap
- Sessions 6-9: 0 clean + 1 stale + 3 interference + 1 trap
- Library grows from 10 (clean) to 55 (17C + 23I + 10S + 5T)

### 3.3b: Library Rebalanced
- 68 total (was 75): 30 clean, 23 interference, 10 stale, 5 trap
- Conflict types: schema=7, semantic=5, version=6, near_dup=5
- All conflict groups have ≥2 members
- Loads from validated JSON instead of Python source

## Grok-3 Review Feedback

### Approved
- 80 tasks is reasonable for controlled simulation
- Retrieval metrics are "the right core set"
- Growing condition is "realistic and well-designed"
- Phase 4 pilot: 25-30 tasks × 5 conditions = 125-150 runs

### Suggestions for Phase 4
1. Add 10-15 more natural interference tasks to reach ~90-95 total
2. Add "mature stable" growing variant (grows but interference stays low)
3. Add harmful retrieval rate and crossover by conflict type
4. Report no-memory performance separately for memory-required vs non-required tasks
5. Validate natural interference tasks with external review
6. Ensure deterministic seeding for reproducibility
7. Consider more even conflict type distribution (currently schema_conflict=7 is dominant)

## Next Steps (Phase 4)
- Implement real agent evaluation harness
- Run pilot: 25-30 tasks × 5 conditions × 5 seeds
- Collect actual agent data
- Analyze crossover point and interference effects