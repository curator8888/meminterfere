# Phase 3: Grok-3 Review

**Date:** 2026-06-25
**Model:** grok-3
**Purpose:** Review of expanded tasks, retrieval metrics, and growing condition

**Phase 3 Review**

You've made solid progress addressing the Phase 2 issues. The move to 80 tasks, addition of natural interference, mixed-skill growing condition, and proper retrieval metrics are all improvements. Below are direct answers to your questions.

### A) Is 80 tasks enough?

80 is reasonable for a controlled simulation study, especially with the current difficulty and interference distributions. However, you are still light on **organic interference** (only 15/80). Most of your interference is still artificially injected.

**Missing task types:**
- Long-horizon tasks (6+ steps) where interference accumulates across multiple decisions
- Tasks with **temporal staleness** (e.g., "book the same flight you saw yesterday" when prices changed)
- Tasks requiring **skill composition** under interference (e.g., "use skill A then skill B" where a near-duplicate of A exists)
- More tasks where memory is *harmful but not obviously so* (subtle semantic conflicts)

You could probably stop at ~90–95 if you add 10–15 more high-quality natural interference tasks.

### B) Retrieval metrics

The metrics you added (P@K, Recall@K, interference precision, conflict resolution distribution, crossover point) are the right core set. They directly target the research question.

**Useful additions:**
- **Harmful retrieval rate**: % of top-K that are interference/stale/trap skills (more direct than "interference precision")
- **Clean vs. polluted precision**: Separate P@K for clean skills vs. conflicting skills
- **Resolution latency**: How many steps it takes the agent to recover after retrieving a conflicting skill (important for real agents)
- **Crossover point by conflict type** (not just overall) — schema conflicts usually cause earlier harm than near-duplicates

Your current set is already strong; the above are refinements rather than requirements.

### C) Growing condition schedule

The schedule is realistic and well-designed. Progressively reducing clean skills while increasing interference/stale/trap skills properly simulates library bloat. The final state (17 clean + 23 interference + 10 stale + 5 trap) creates a plausible "mature but messy" library.

One minor suggestion: consider adding a **"mature stable"** variant where the library grows but interference stays low (to separate size from pollution effects).

### D) Phase 4 minimum viable pilot

For real agent evaluation, I recommend:

- **25–30 tasks** (stratified across difficulty × interference level × memory requirement)
- **5 conditions** (no memory, clean-only, growing, static polluted, oracle)
- Total: **125–150 runs** (very manageable)

Stratification is critical:
- At least 8–10 high-interference tasks
- At least 8–10 tasks that require memory
- 5–6 natural interference tasks

This gives you enough power to detect the crossover effect without excessive cost.

### E) Other concerns

1. **No-memory baseline calibration**: With 31 tasks that don't require memory, the no-memory condition should now look reasonable. Make sure you report performance on the 49 memory-required tasks separately.
2. **Natural interference validation**: Have at least one other person review the 15 natural interference tasks to confirm the interference is actually present and not just labeled as such.
3. **Conflict type balance**: You now have ≥2 per type, but schema_conflict (7) is still dominant. Consider whether this reflects reality or if you want more even distribution.
4. **Reproducibility**: Loading from validated JSON is good. Make sure task generation + library sampling is fully deterministic with seeds.

**Overall**: You're in good shape to move to Phase 4. The main remaining risk is still having too many artificial interference tasks. If you can push natural interference to ~25–30%, the study becomes significantly stronger.