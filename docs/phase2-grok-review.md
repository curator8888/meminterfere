# Phase 2: Grok-3 Review

**Date:** 2026-06-25
**Model:** grok-3
**Purpose:** Review of skill library + evaluation framework

**Review of Phase 2 Implementation**

Here's a direct, critical assessment:

### 1. Skill Library
**Does not match the stated design.**

You said interference would be split into 4 conflict types *each with 5 instances* (20 total), but the stats show:
- `semantic_conflict`: 10
- `near_duplicate`: 10
- `schema_conflict`: 5
- `version_conflict`: 5

This is inconsistent. You either changed the design or made an error in construction.

Other issues:
- **Trap skills** (5) are listed separately from interference, which is fine, but you should confirm whether they are meant to be *always wrong* or *conditionally wrong*.
- Staleness distribution is reasonable, though 180d having 4 while others have 2 is slightly imbalanced.
- Domain split is acceptable.

**Fix**: Either correct the counts to 5/5/5/5 or update the design doc to reflect the actual 5/10/10/5 split and justify why semantic and near-duplicate conflicts deserve double weight.

### 2. Evaluation Tasks
**Too few + structurally incomplete.**

47 tasks is marginal for 7 conditions, especially once you stratify by difficulty, domain, and interference level. The hard bucket (only 6) is particularly thin.

More important problems:
- `by_natural_interference` is all `no`. This is a serious gap. You currently have no tasks where interference emerges *organically* from task phrasing rather than being artificially triggered. This weakens ecological validity.
- The "requires memory" split (24/23) is good, but you should also track *which specific skills* each task is designed to surface.

**Recommendation**: Target 70–80 tasks minimum. Add at least 8–10 hard tasks and a meaningful set of tasks with natural interference (probably 12–15).

### 3. Conditions
The 7 conditions are reasonable. The main issue is the **Growing Library** condition.

Starting with 10 clean skills and adding 5 per session is clean for measuring gradual degradation, but it delays the introduction of interference too much. In reality, agents usually accumulate both good and bad skills over time.

**Better design**: Add a mix after the first 3–4 sessions (e.g., 3 clean + 2 interference/stale per addition). This better simulates real skill library growth and lets you observe when interference starts dominating.

### 4. Metrics
The metric set is strong, particularly the 8-type error taxonomy. However, you are missing two useful measures:

- **Interference-specific retrieval metrics**: Precision@K and recall of relevant vs. conflicting skills when the agent queries memory. This directly measures whether the model is retrieving the wrong skill due to interference.
- **Conflict resolution attempts**: Count of times the agent detects a conflict and either (a) asks for clarification, (b) tries multiple skills, or (c) silently picks one. This distinguishes "didn't notice" from "noticed but failed."

### 5. Simulation Results
The pattern is directionally correct but the numbers are too clean:

- 0% with "No Memory" is unrealistic given that 23 tasks don't require memory.
- The drop from Clean (80%) to Clean+Interference (20%) is very large. Real results will likely be less dramatic (probably 45–60% range).
- You should also simulate a condition where interference is present but *not retrieved* (to separate retrieval failure from interference effects).

### 6. Phase 3 Plan (Concrete)

Do the following in order:

1. **Fix the library first** (1–2 days)
   - Reconcile the conflict type counts.
   - Add natural interference labels to existing tasks or create new ones.
   - Implement a simple skill versioning system so you can track which version of a skill is being called.

2. **Expand the task set** (3–4 days)
   - Increase to ~75 tasks.
   - Add 10–12 hard tasks and 12–15 tasks with natural interference.
   - Create a task-skill mapping file (which skills each task should ideally use).

3. **Implement proper evaluation harness** (using Hermes)
   - Run each condition with the *same* task order across multiple random seeds (minimum 3).
   - For the Growing condition, log the exact skill library state at each session.
   - Capture full traces: retrieved skills, final skill chosen, parameters, and error type.

4. **Add diagnostic logging**
   - Log the top-5 retrieved skills for every memory-using call.
   - Record whether a conflicting skill was in the top-k.

5. **Analysis priorities** (run these first)
   - Degradation slope on the Growing condition.
   - Error type distribution, especially `conflict deadlock` vs `wrong skill`.
   - Retrieval precision of correct vs interfering skills.

Would you like me to give you the revised skill counts and a suggested task distribution that fixes the current imbalances?