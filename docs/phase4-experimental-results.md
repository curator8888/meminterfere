# Phase 4: Experimental Results and Peer Review

**Date:** 2026-06-25
**Status:** Preliminary — 54 total runs across 4 experiments

## Experimental Results

### Experiment 1: Skill Selection Under Interference (25 runs)
- 5 tasks × 5 conditions (no_memory, clean, clean+interference, clean+stale, all_memory)
- **Finding**: Skill selection was 100% correct across ALL conditions
- Confidence: no_memory=18%, clean=85%, clean+interf=85%, clean+stale=85%, all=83%
- Interference did NOT change tool selection when skill names were distinct

### Experiment 2: Conflict Disambiguation (20 runs)
- 10 tasks × 2 conditions (clean only vs with all conflicting versions)
- **Finding**: 90% correct in conflict condition vs 100% in clean
- 1 near-duplicate chosen incorrectly (browse_website instead of navigate_url)
- Semantic conflicts (same name, different return format) NEVER chosen over clean versions

### Experiment 3: Staleness Effect (6 runs)
- 3 tasks × 2 conditions (clean only vs clean + stale)
- **Finding**: Tool selection was SAME in both conditions
- BUT: Parameter formatting CHANGED in 2/3 cases when stale versions were visible
- Staleness affects HOW you call a skill, not WHICH skill you call

### Experiment 4: Trap Skill Acceptance (3 runs)
- Agent chose the TRAP skill in ALL 3 cases
- The agent has NO mechanism to detect subtly wrong descriptions

## Three Key Findings

1. **LLMs are SURPRISINGLY ROBUST to name-distinct interference** — schema, semantic, and version conflicts that differ in name do NOT confuse the agent
2. **NEAR-DUPLICATES are the primary failure mode** — similar names cause wrong selection (1/5)
3. **TRAP SKILLS are NEVER detected** — subtly wrong descriptions are always accepted (3/3)

## GLM-5.1 Critical Review

### Is this publishable?
**No, not yet.** 54 runs with N=3 for traps is statistically invalid. Need 300-500 total runs.

### Is "selective harm" novel?
The broad finding is obvious in retrospect. But the NUANCE is novel: interference corrupts tool INVOCATION (parameter formatting), not just tool SELECTION. This is a dangerous failure mode.

### Biggest threats to validity
1. Single model (Grok-3-mini only)
2. No separation of retrieval vs planning failures
3. Synthetic skill libraries, not real ones
4. Simple tasks don't require memory reliance
5. Near-duplicate taxonomy is undefined

### Strongest paper framing
**Drop "memory hurts" thesis.** Reframe as:

**"The Anatomy of Tool Interference: Why LLM Agents Fail at the Margins"**

Thesis: LLM agents are robust to semantic and name-distinct interference, but exhibit critical silent failure modes at the margins — specifically near-duplicates and adversarial traps. Interference corrupts tool INVOCATION (parameters), not just SELECTION.

Contributions:
1. Formal taxonomy of skill library interference (Distinct, Semantic, Versioned, Near-Duplicate, Trap)
2. Benchmark (MemInterfere) showing failure shifts from overt (wrong tool) to covert (right tool, wrong parameters)
3. Analysis showing retrieval and planning must be evaluated separately

### Next steps
1. Scale to 300-500 runs across multiple models
2. Add 2-3 more LLMs (Llama-3, Claude, GPT-4o-mini)
3. Decouple retrieval from planning
4. Add multi-step agentic workflows
5. Formalize near-duplicate taxonomy