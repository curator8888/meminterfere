# Phase 1: GLM-5.1 Review of Literature Review and Experimental Design

**Date:** 2026-06-25
**Model:** glm-5.1
**Purpose:** Critical review of Grok-3's Phase 1 output

# Senior Review: Phase 1 — "When Memory Misleads"

Overall: This is a solid starting point. The literature is well-curated for the RAG/agent intersection, the 4-way condition structure is clean, and the novelty claim identifies a real gap. But there are structural problems — the biggest being that the experimental design **doesn't actually test the core novelty claim** (compounding degradation over time), the literature review misses foundational cognitive science and key tool-use papers, and the skill library is underspecified for the interference phenomena you want to study. Detailed breakdown below.

---

## 1. Literature Gaps

### Critical omissions

**Cognitive science foundations on memory interference.** The entire concept of "interference" in memory comes from psychology, and none of the foundational work is cited. This isn't just academic completeness — these papers give you *theoretical vocabulary and predicted functional forms* that your paper needs:

- **Anderson's Fan Effect** (Anderson, 1974; ACT-R literature): As more facts are associated with a concept, retrieval slows and error rates increase. This is *exactly* your library bloat mechanism, with decades of modeling behind it. You should cite this and test whether skill libraries produce fan-effect curves.
- **Retroactive and proactive interference** (Underwood, 1957; McGeoch, 1932): The distinction between new skills interfering with old (retroactive) and old skills interfering with new (proactive) maps directly onto your staleness vs. conflict conditions. These terms give you a language that reviewers will recognize.
- **Retrieval-induced forgetting** (Anderson et al., 1994): Practicing some skills makes related but unpracticed skills harder to retrieve. This predicts a specific failure mode your library will exhibit — frequent-use skills will suppress rare-but-correct alternatives.

**Tool-use conflict papers.** You're missing the most directly relevant empirical work:

- **Gorilla** (Patil et al., 2023): Explicitly studied API hallucination and confusion between similar APIs on APIBench. This is the closest existing work to your "schema conflict" mechanism. Not citing it is a gap that reviewers will catch immediately.
- **ToolLLM** (Qin et al., 2023): Large-scale tool-use evaluation that documents accuracy drops as the tool set grows. Their finding that performance degrades with more available tools is a direct predecessor to your bloat claim.
- **ToolFormer** (Schick et al., 2023): Shows models can learn to call APIs but doesn't address what happens when the API space gets large or conflicting — your gap.

**In-context learning interference.** The ICL literature has studied how conflicting or noisy demonstrations degrade performance:

- **Min et al. (2022)** on how ICL is primarily a label-space phenomenon — conflicting labels in demonstrations cause disproportionate damage.
- **Yoo et al. (2022)** on how the *diversity* and *quantity* of in-context examples affects performance, with diminishing and eventually negative returns.

This matters because skill library retrieval is essentially dynamic ICL — retrieved skills become in-context demonstrations.

**Knowledge conflict surveys.** You cite KILM but miss the broader 2024 survey literature:

- **Chen et al. (2024)** "When Not to Trust Language Models" — comprehensive taxonomy of parametric vs. contextual knowledge conflicts.
- **Xie et al. (2024)** "Adaptive Chameleon or Stubborn Sloth" — shows LLMs are surprisingly stubborn about parametric knowledge even with conflicting context. This predicts your agents will sometimes *ignore* retrieved skills and fall back on parametric knowledge, a failure mode not in your error taxonomy.

**Software engineering analogs.** The "dependency hell" and API versioning literature (e.g., Raemaekers et al., 2014 on API deprecation) is a direct real-world analog. Your staleness mechanism is literally API deprecation — cite the SE literature to show this is a known, unsolved problem that you're reframing for agents.

### Missing angles

- **Security/poisoning**: No adversarial literature. What if skills are *deliberately* misleading? Papers on backdoor attacks in RAG (e.g., "PoisonedRAG" by Zou et al., 2024) are relevant even if you don't test adversarial settings, because they establish that retrieval is an attack surface.
- **Multi-agent memory sharing**: When agents share skill libraries (a natural use case), interference compounds. No literature on this.
- **Forgetting in compositional reasoning**: Recent work showing LLMs struggle to compose skills that work individually (e.g., Kiri et al., 2024). This is a different failure mode from interference but interacts with it.

---

## 2. Experimental Design Critique

### The central problem: You don't test your own novelty claim

Your claim is about **compounding, library-level degradation over time**. Your experimental design tests **static snapshots** of libraries at fixed sizes. There is no longitudinal dimension, no library growth over sessions, no measurement of how degradation accumulates.

This is the single biggest weakness. You need at least one condition where the library *grows incrementally* across experimental sessions, and you measure performance at multiple time points. Without this, you can claim interference exists but not that it *compounds*.

**Fix**: Add a longitudinal condition (Condition 5): "Growing library" — start with 10 clean skills, add 5 skills per session (mix of clean, conflicting, stale), run 12 sessions, measure performance at each session. Plot the degradation curve. This is where your paper's signature figure comes from.

### Other weaknesses

**Interference set conflates two distinct failure modes.** "Conflicting" and "near-duplicate" skills cause different problems:
- Near-duplicates cause *retrieval confusion* (which one do I pick?)
- Conflicting skills cause *execution failures* (I picked one but it has the wrong schema)

These should be separate conditions or at minimum separately tagged and analyzed. The current design can't disentangle them.

**No oracle retrieval condition.** You need a condition where the correct skill is always provided (no retrieval step), to separate retrieval failures from execution failures. Without this, when performance drops in the interference condition, you can't tell whether the agent retrieved the wrong skill or retrieved the right skill but used it incorrectly. Add Condition 0: "Oracle" — gold skill always provided.

**Circular task design.** 80 custom tasks "designed to trigger interference" will produce interference by construction. Reviewers will rightly ask: does this generalize? You need to show that interference hurts on *natural* tasks too, not just adversarially constructed ones. Suggestion: take the 200 ToolBench tasks and *tag* which ones happen to have potential interference in your library, then analyze natural vs. designed interference separately.

**Missing interaction effects.** Clean + interference and Clean + stale are tested separately, but real libraries have both simultaneously. Add a Condition 5 (or renumber): Clean + interference + stale. If effects are super-additive, that's a strong result.

**60 skills is too few for bloat.** Voyager's library grew to hundreds. To demonstrate bloat-driven degradation, you need to test at multiple library sizes. Run a scaling experiment: 30, 60, 120, 240 skills, measuring retrieval accuracy and task success at each size. This directly tests the fan-effect prediction.

**Power analysis is optimistic for calibration metrics.** ECE requires many more data points per condition than binary success rate. With 200 tasks × 3 seeds = 600 data points per condition, your ECE estimates will be noisy. Consider binning confidence scores and reporting reliability diagrams alongside ECE.

**No measurement of *when* memory helps vs. hurts.** The title says "When Memory Misleads" but you don't systematically vary task difficulty or novelty. Add a difficulty dimension: easy tasks (agent can solve without memory) vs. hard tasks (agent needs memory). The prediction: memory helps on hard tasks but hurts on easy tasks where retrieval adds noise. This is your crossover point.

**Error taxonomy is good but incomplete.** Add:
- *Parametric override*: Agent ignores retrieved skill and uses parametric knowledge (from Xie et al.)
- *Retrieval omission*: Agent should retrieve but doesn't (the Self-RAG finding)
- *Compositional failure*: Agent retrieves two correct skills but fails to compose them

### What's overkill

- **3 random seeds** is fine for a pilot but consider 5 for the full experiment. The marginal cost is low and it substantially improves CI estimates.
- **4 domains** is too many for 60 skills. See skill library critique below.

---

## 3. Skill Library Design Improvements

### Reduce domains, increase depth

60 skills across 4 domains = 15 per domain. That's too sparse to create meaningful within-domain interference. **Reduce to 2 domains, 30 skills each.** I suggest:

- **Web navigation** (search, browse, extract, fill forms): Rich potential for schema conflicts (multiple search APIs with different parameters) and staleness (websites change).
- **API calling** (calendar, email, file management, database): Rich potential for semantic conflicts (multiple calendar tools) and version conflicts (API v1 vs v2).

### Define a proper conflict taxonomy

Instead of "conflicting or near-duplicate," define four conflict types with equal representation:

| Conflict Type | Description | Example |
|---|---|---|
| **Schema conflict** | Same intent, different parameter structure | `search_web(query, limit)` vs `search_web(q, num_results, safe_mode)` |
| **Semantic conflict** | Same name, different behavior | `get_events(date)` returns JSON vs `get_events(date)` returns CSV |
| **Version conflict** | Same skill, different API version | Calendar API v2 requires auth token that v1 didn't |
| **Near-duplicate** | Different skills with high embedding overlap | `find_restaurants(city)` vs `find_dining(city)` — agent must choose |

Each type should have 5 instances (20 total interference skills), tagged with `conflict_type` in metadata.

### Graduated interference levels

To test the fan effect, you need tasks where the number of competing skills varies. Design tasks with 0, 1, 2, 4, and 8 competing alternatives. This lets you plot the degradation curve as a function of interference density — a signature result.

### Staleness as a continuous variable

Don't make staleness binary. Create skills with different "ages since API change": 1 day, 1 week, 1 month, 6 months. This lets you measure staleness as a continuous predictor and test whether degradation follows a predictable decay function.

### Add trap skills

Include 5 skills that are *plausible but subtly wrong* — e.g., a `send_email` skill that CCs a wrong address, or a `calculate_tax` skill with an off-by-one bracket error. These test whether agents can detect and reject retrieved-but-incorrect skills, which is the core of the reliability problem.

### Metadata you're missing

Add to each skill's JSON:
- `embedding_vector`: Pre-computed embedding for similarity analysis
- `parametric_overlap`: Does the base model already "know" this skill from training? (Test by asking the model without retrieval.)
- `conflict_type`: Schema / Semantic / Version / Near-duplicate
- `staleness_days`: Days since the simulated API change
- `fan_degree`: Number of competing skills for the same intent
- `trap`: Boolean, whether this skill is subtly wrong

---

## 4. Novelty Claim Assessment

### The claim is 70% there. Here's what's weak:

**"Compounding" is the key novelty word but the weakest link.** You claim compounding degradation but don't define what compounds with what. The three mechanisms (schema conflicts, staleness, bloat) are listed as parallel, but compounding implies they *interact*. The strongest version of your claim would be:

> "Schema conflicts and staleness interact with bloat: as libraries grow, stale skills become harder to identify (more noise), and schema conflicts become harder to resolve (more alternatives). This produces super-linear degradation."

This is a testable prediction. Make it.

**Schema conflicts alone aren't novel.** Gorilla already showed API confusion. Your novelty is in the *accumulation* and *interaction* of failure modes over time in a persistent library, not in the existence of any single failure mode. Sharpen the claim to emphasize the *system-level* nature of the problem.

**The "when" is underspecified.** Your title says "When Memory Misleads" but the claim doesn't identify boundary conditions. A stronger claim specifies the crossover:

> "We identify the crossover point where memory retrieval transitions from beneficial to harmful: when the library exceeds [X] skills with [Y]% interference, retrieval hurts more than it helps on tasks the agent could otherwise solve."

You don't have to know X and Y yet — that's what the experiment is for — but you should commit to finding and reporting them.

### Suggested revised claim

> "In persistent, multi-skill agent libraries, memory interference produces reliability degradation that compounds with library size and skill age. We identify three mechanisms — schema conflicts, temporal staleness, and retrieval noise from library bloat — and demonstrate that their effects interact: stale skills become harder to identify as libraries grow, and schema conflicts become harder to resolve with more competing alternatives. We quantify the crossover point where memory retrieval transitions from beneficial to harmful, and show that standard agent architectures lack the retrieval-gating and consistency-checking mechanisms needed to maintain reliability at scale. These failure modes are qualitatively distinct from single-shot RAG retrieval errors because they accumulate over the agent's lifetime and produce systematic rather than random degradation."

This is stronger because it: (a) specifies interaction between mechanisms, (b) commits to finding a crossover point, (c) distinguishes systematic from random degradation, and (d) emphasizes the lifetime accumulation dimension.

---

## 5. Concrete First-Week Plan

### Day 1–2: Skill library schema and conflict taxonomy

This is the foundation. Everything else depends on it.

```json
{
  "skill_id": "web_search_v2",
  "name": "search_web",
  "description": "Search the web for information",
  "domain": "web_navigation",
  "parameters": {"query": {"type": "string", "required": true}, "num_results": {"type": "integer", "required": false}},
  "return_format": "json_list",
  "success_rate": 0.92,
  "last_verified": "2026-06-01",
  "conflict_group_id": "search_web_group",
  "conflict_type": "schema_conflict",
  "fan_degree": 3,
  "staleness_days": 0,
  "is_trap": false,
  "parametric_overlap": true,
  "embedding_vector": [...]
}
```

Build the full 60-skill JSON. Write a validation script that checks: every conflict group has ≥2 members, fan degrees are consistent, staleness_days are set correctly.

### Day 2–3: Implement retrieval modes in ByteRover

Four retrieval modes:
1. **None**: No retrieval (baseline)
2. **Clean**: Retrieve only from clean set
3. **Full**: Retrieve from all skills (clean + interference + stale)
4. **Oracle**: Always return the gold skill (for separating retrieval vs. execution failures)

Each mode should be a simple filter on the skill library before the retrieval step. Don't over-engineer — this is a config flag, not a new system.

### Day 3–4: Instrumentation layer

Add logging to Hermes that records, per turn:
- `retrieved_skill_ids` (top-k)
- `retrieval_scores` (similarity scores)
- `agent_confidence` (verbalized or calibrated)
- `skill_used` (which skill the agent actually invoked)
- `execution_result` (success/failure/partial)
- `error_type` (from your taxonomy)

Output format: JSONL, one line per turn. This makes downstream analysis trivial.

### Day 4–5: Pilot run — 20 tasks, all conditions

Don't wait