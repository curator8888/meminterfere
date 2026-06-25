# Phase 5.5 Review: NLP/Systems Perspective on MemInterfere

**Reviewer:** GLM-5.1 (Research profile)  
**Date:** 2026-06-25  
**Scope:** 3 models x 80 tasks x 5 conditions x 3 temps = 3,600 runs  
**Key finding:** clean_memory=92.4% vs clean_interference=91.1% - a 1.3pp difference, not statistically significant

---

## 1. Ecological Validity

### 1.1 The Skill Selection Task Is Not Real Agent Behavior

The experimental setup asks models to **select a skill name** from a menu, not to **use a skill** in a multi-turn agent loop. This is a single-step classification task, not an agentic workflow. Real agents:

- **Retrieve** skills via semantic search, not by reading a full list
- **Execute** skills and observe results
- **Chain** skills across turns with error recovery
- **Verify** outcomes against expectations

The prompt format makes this a **forced-choice selection task** with zero execution cost. In real deployments, the cost of a wrong selection is asymmetric.

**Severity: HIGH.** The experiment measures **skill name disambiguation under list-level interference**, not **agent reliability degradation**. The framing should be revised accordingly.

### 1.2 Task Design Confounds

Three tasks (nat_web_001, nat_web_002, med_web_002) fail at **0% accuracy across ALL conditions including oracle**. These are **annotation disagreements**:

- nat_web_001: "weather in Tokyo" -> gold is search_web, but models pick get_weather (arguably correct)
- nat_web_002: "headlines about AI regulation" -> gold is search_web, but models pick search_news (arguably correct)
- med_web_002: "summarize 3 articles about climate policy" -> gold is search_web, but models pick search_news

This inflates the error rate by ~3.75pp.

### 1.3 93.8% of Tasks Contain Skill Verbs in Their Descriptions

The task is closer to **keyword matching** than genuine skill retrieval. The 3 consistently-failing tasks are ones where description keywords match a different skill than the gold label.

### 1.4 The Skill Library Has Unrealistic Interference Density

The 100-skill library has 22% near-duplicate/interference density, far higher than real agent ecosystems (typically <5%).

---

## 2. Near-Duplicate Taxonomy

### 2.1 Well-Defined but Under-Tested

Only **3 near-duplicate errors** out of 72 near-duplicate task-condition runs (4.2%). The gradient test pairs from the taxonomy are **not implemented** in the experiment.

### 2.2 Trap Acceptance Is Not 100%

Phase 5.5 data shows trap acceptance at **29.6%** (16/54 for one task), not 100%. The one task that fires (interfere_trap_001) has a description that explicitly cues the trap ("without waiting for full results" = fast). The other two trap tasks show **0% trap acceptance**.

### 2.3 Similarity Scores Are Not Empirically Validated

The weighting (0.3/0.4/0.3) is arbitrary and not fit to data.

---

## 3. Does the 5-Condition Design Adequately Isolate Retrieval vs. Selection vs. Invocation?

### 3.1 The Oracle Condition Does Not Provide Only the Gold Skill

The oracle condition shows the same 40 clean skills as clean_memory. GPT-4o-mini has identical accuracy in both (91.25%). The retrieval tracks (Track A/B/C from Phase 5.3) were **not implemented** in Phase 5.5.

**Severity: HIGH.** Without retrieval-track separation, the experiment cannot distinguish retrieval failures from selection failures from invocation failures.

### 3.2 The no_memory Condition Is Degenerate

Zero skills -> 0% accuracy is trivially impossible, not a meaningful baseline.

### 3.3 Staleness Is Confounded with Library Size

clean_interference has 90 skills, all_memory has 100. The marginal effect of staleness cannot be separated from library size effects.

### 3.4 Invocation Accuracy Is Not Measured

Only skill selection accuracy is measured. The "staleness corrupts invocation" claim from Phase 4 is untestable in the current setup.

---

## 4. Temperature Variations

Temperature (0.0, 0.3, 0.7) produces near-identical accuracy across all models. GPT-4o-mini is completely deterministic (73/80 every cell). The sweep adds 2,400 runs but zero information. Drop it.

---

## 5. Statistical Summary

| Comparison | Mean Diff | Cohen d | t(79) | p |
|-----------|-----------|---------|-------|---|
| clean_memory vs clean_interference | +1.25pp | 0.107 | 0.95 | 0.34 |
| clean_memory vs all_memory | +0.83pp | 0.074 | 0.66 | 0.51 |
| oracle vs clean_interference | +1.25pp | 0.108 | 0.96 | 0.34 |
| oracle vs all_memory | +0.83pp | 0.074 | 0.66 | 0.51 |

**None significant at p<0.05.** Effect sizes negligible (d < 0.11).

On oracle-solvable tasks only: oracle 100.0% -> clean_interference 97.6% (2.4pp drop), but post-hoc.

---

## 6. What Would Make This Paper Compelling at ACL/EMNLP

### Must-Have Additions

1. Fix oracle condition to provide only the gold skill
2. Remove or relabel 3 mislabeled tasks
3. Implement gradient test (3 similarity levels x 10 pairs x 3 models)
4. Implement Track A (gold retrieval) to separate retrieval from selection
5. Compute ECE and confidence calibration
6. Power analysis confirming adequate power

### Should-Have Additions

7. Measure invocation/parameter accuracy
8. Add adversarial interference conditions
9. Multi-step evaluation for cascade failures
10. Human annotation for task-skill alignment

### Reframe as

**"How Robust Are LLM Agents to Skill Library Interference? A Systematic Investigation"**

Contributions: (1) Formal taxonomy, (2) MemInterfere benchmark, (3) Null result: LLMs are robust to name-distinct interference, (4) Specific marginal failure modes.

---

## 7. Summary of Key Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| Task is skill selection, not agent execution | HIGH | Limits claim scope |
| 3 tasks have annotation errors | HIGH | +3.75pp error inflation |
| Oracle condition doesn't provide only gold skill | HIGH | Cannot separate retrieval from selection |
| Trap acceptance is 29.6%, not 100% | MEDIUM | Must revise Phase 4 claim |
| Near-duplicate effect is 4.2% (3/72) | MEDIUM | Underpowered; need gradient test |
| Temperature adds no information | LOW | Drop or justify |
| No parameter/invocation accuracy | MEDIUM | Cannot test invocation failure claim |
| Library density (22%) is unrealistic | MEDIUM | Acknowledge as limitation |

**Overall assessment:** The null result is informative, but the paper needs significant revisions to framing, task design, and analysis before ACL/EMNLP. The taxonomy is the strongest contribution; the gradient test and retrieval-track separation are the most critical missing pieces.
