# Phase 1: Consolidated Literature Review and Experimental Design

**Date:** 2026-06-25
**Reviews:** Grok-3 (initial) + GLM-5.1 (critical review)
**Status:** Complete — ready for Phase 2

---

## 1. Literature Review (Consolidated)

### Core Papers on Agent Memory and Interference

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| Reflexion (Shinn et al., 2023) | Self-reflection + episodic memory improves performance but can reinforce incorrect trajectories when reflection is flawed | Direct: memory can entrench errors |
| ExpeL (Zhao et al., 2023) | Agents that extract "insights" from past trajectories improve but performance plateaus/drops with low-quality insights | Direct: unfiltered skill libraries degrade reliability |
| Self-RAG (Asai et al., 2023) | Learned retrieval decision + critique reduces hallucination; retrieval hurts when not selective | Strongest existing evidence retrieval can hurt; extends to multi-skill agents |
| CRITIC (Gou et al., 2023) | External tool feedback + self-critique improves reasoning, but benefit reverses when stored critiques are noisy | Direct: interference between stored critiques and new tool calls |
| Corrective RAG (Yan et al., 2024) | Explicit corrective step after retrieval; standard RAG frequently introduces conflicting/low-value passages | Provides corrective mechanism to test as mitigation |
| MemGPT (Packer et al., 2023) | OS-style memory paging improves long-horizon performance but context-management errors cause catastrophic drops | Closest to ByteRover-style persistent trees; highlights failure modes |
| Voyager (Wang et al., 2023) | Automatic skill library growth enables open-ended progress but library accumulates near-duplicate/low-utility skills | Direct: real-world demonstration of skill-library bloat |
| The AI Scientist (Lu et al., 2024) | Agents storing/reusing research "ideas" generate papers but fall into low-novelty loops | Shows interference in creative/agentic setting |
| Gorilla (Patil et al., 2023) | API hallucination and confusion between similar APIs on APIBench | Direct predecessor to schema conflict mechanism |
| ToolLLM (Qin et al., 2023) | Performance degrades as tool set grows; documented accuracy drops with more available tools | Direct predecessor to bloat claim |
| MemoryBank / RecallM (2024) | Long-term memory improves consistency but introduces stale-memory errors when facts change | Direct: stale memory condition |

### Cognitive Science Foundations (Added per GLM-5.1)

| Concept | Source | Relevance |
|---------|--------|-----------|
| Fan Effect | Anderson, 1974 (ACT-R) | As more facts are associated with a concept, retrieval slows and errors increase — exactly our bloat mechanism |
| Retroactive/Proactive Interference | Underwood, 1957; McGeoch, 1932 | New skills interfering with old (retroactive) and old interfering with new (proactive) maps to staleness vs. conflict |
| Retrieval-Induced Forgetting | Anderson et al., 1994 | Practicing some skills suppresses related but unpracticed alternatives — predicts specific failure mode |

### Knowledge Conflict and ICL Interference

| Paper | Key Finding | Relevance |
|-------|-------------|-----------|
| KILM (Xu et al., 2023) | Injecting facts into LLMs creates conflicts with parametric knowledge | Analog to skill-library vs. base-model conflicts |
| "When Not to Trust LMs" (Chen et al., 2024) | Taxonomy of parametric vs. contextual knowledge conflicts | Comprehensive framework for conflict types |
| "Adaptive Chameleon or Stubborn Sloth" (Xie et al., 2024) | LLMs are stubborn about parametric knowledge even with conflicting context | Predicts agents will sometimes ignore retrieved skills |
| Min et al. (2022) | ICL is primarily label-space; conflicting labels in demonstrations cause disproportionate damage | Skill retrieval = dynamic ICL with potential label conflicts |
| "RAG Can Be Harmful" (Cuconasu et al., 2024) | Strong retrievers degrade QA when passages contain distractors/contradictions | Direct empirical support for retrieval-hurts thesis |
| PoisonedRAG (Zou et al., 2024) | Retrieval is an attack surface for adversarial injection | Security angle: deliberately misleading skills |

---

## 2. Experimental Design (Consolidated)

### 2.1 Skill Library Structure

**60 skills across 2 domains** (reduced from 4 per GLM-5.1's recommendation):

- **Web Navigation** (30 skills): search, browse, extract, fill forms
- **API Calling** (30 skills): calendar, email, file management, database

**Conflict taxonomy** (4 types, 5 instances each = 20 interference skills):

| Conflict Type | Description | Example |
|---------------|-------------|---------|
| Schema conflict | Same intent, different parameters | `search_web(query, limit)` vs `search_web(q, num_results, safe_mode)` |
| Semantic conflict | Same name, different behavior | `get_events(date)` returns JSON vs CSV |
| Version conflict | Same skill, different API version | Calendar API v2 requires auth that v1 didn't |
| Near-duplicate | Different skills, high embedding overlap | `find_restaurants(city)` vs `find_dining(city)` |

**Clean set**: 30 high-precision skills (verified >90% success rate)
**Interference set**: 20 skills with controlled conflicts (5 per type)
**Stale set**: 10 skills with graduated staleness (1 day, 1 week, 1 month, 6 months)
**Trap set**: 5 subtly wrong skills (plausible but incorrect)

**Skill metadata** (per GLM-5.1):
```json
{
  "skill_id": "web_search_v2",
  "name": "search_web",
  "domain": "web_navigation",
  "parameters": {...},
  "return_format": "json_list",
  "success_rate": 0.92,
  "last_verified": "2026-06-01",
  "conflict_group_id": "search_web_group",
  "conflict_type": "schema_conflict",
  "fan_degree": 3,
  "staleness_days": 0,
  "is_trap": false,
  "parametric_overlap": true
}
```

### 2.2 Conditions (6-way)

| Condition | Description | Purpose |
|-----------|-------------|---------|
| 0. Oracle | Gold skill always provided | Separate retrieval vs. execution failures |
| 1. No memory | Baseline ReAct without retrieval | Measure baseline without memory |
| 2. Clean memory | Only correct, verified skills | Replicate Voyager-type "memory helps" result |
| 3. Clean + interference | Correct + conflicting/near-duplicate skills | Test schema/semantic/near-duplicate conflicts |
| 4. Clean + stale | Correct + outdated skills | Test staleness |
| 5. Clean + interference + stale | All memory types combined | Test interaction effects |
| 6. Growing library | Start clean, add 5 skills/session over 12 sessions | Test compounding degradation |

### 2.3 Tasks

- **Existing benchmarks**: ToolBench (200 queries subset), API-Bank (Level-2), WebArena (50 tasks)
- **Custom interference tasks**: 80 tasks designed to trigger interference (e.g., "book a meeting" when two calendar skills exist)
- **Natural interference analysis**: Tag which ToolBench tasks happen to have potential interference in library; analyze natural vs. designed interference separately

### 2.4 Metrics

| Metric | Type | Notes |
|--------|------|-------|
| Task success rate | Binary + partial credit | Primary metric |
| Calibration (ECE) | Continuous | Expected Calibration Error on confidence scores; use binning + reliability diagrams |
| Error taxonomy | Categorical | Wrong skill, parameter hallucination, stale API, conflict deadlock, infinite retry, parametric override, retrieval omission, compositional failure |
| Tokens per success | Efficiency | Token cost per successful task |
| Degradation slope | Continuous | Success rate vs. number of interfering skills |
| Crossover point | Continuous | Library size where memory retrieval transitions from beneficial to harmful |

### 2.5 Mitigations (Ablations)

1. **Metadata-augmented retrieval**: Add success_rate, recency, conflict_group as retrieval filters
2. **Consistency check**: Retrieve top-k, run lightweight verifier on contradictions
3. **Confidence calibration**: Verbalized uncertainty to gate retrieval
4. **Versioning/TTL**: Expire skills after staleness threshold

### 2.6 Sample Size

- 280 tasks × 7 conditions × 5 random seeds = 9,800 runs
- Primary test: paired bootstrap or McNemar's test on success rate; 95% CI
- Target: detect ≥8% absolute drop with 80% power

---

## 3. Novelty Claim (Revised)

**Core claim**: In persistent, multi-skill agent libraries, memory interference produces reliability degradation that compounds with library size and skill age. We identify three mechanisms — schema conflicts, temporal staleness, and retrieval noise from library bloat — and demonstrate that their effects interact: stale skills become harder to identify as libraries grow, and schema conflicts become harder to resolve with more competing alternatives. We quantify the crossover point where memory retrieval transitions from beneficial to harmful, and show that standard agent architectures lack retrieval-gating and consistency-checking mechanisms needed to maintain reliability at scale. These failure modes are qualitatively distinct from single-shot RAG errors because they accumulate over the agent's lifetime and produce systematic rather than random degradation.

**What's novel vs. existing work**:
- vs. Self-RAG: We study library-level interference over time, not single-shot retrieval
- vs. Gorilla: We study accumulation and interaction of failure modes, not single-API confusion
- vs. Voyager: We quantify when memory hurts (crossover point), not just that it helps
- vs. ExpeL: We study interference mechanisms (schema, staleness, bloat), not just quality filtering

---

## 4. First Week Plan (Consolidated from both reviews)

### Day 1-2: Skill library schema and conflict taxonomy
- Design JSON schema for 60 skills with all metadata fields
- Define 4 conflict types with 5 instances each
- Create staleness levels (0, 1, 7, 30, 180 days)
- Write validation script for consistency

### Day 2-3: Retrieval modes in ByteRover
- Implement 7 retrieval conditions as config flags
- Oracle, None, Clean, Clean+Interference, Clean+Stale, All, Growing

### Day 3-4: Instrumentation layer
- Add per-turn logging to Hermes: retrieved_skill_ids, retrieval_scores, agent_confidence, skill_used, execution_result, error_type
- JSONL output format

### Day 4-5: Pilot run
- 20 tasks, all conditions
- Validate logging, metrics pipeline, and interference injection
- Check that interference effects are detectable before scaling up

### Day 5-7: Scale up
- Full 280-task evaluation
- 5 seeds per condition
- Begin analysis and crossover point identification