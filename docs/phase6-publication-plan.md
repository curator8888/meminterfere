# MemInterfere Phase 6: Publication Plan

Based on peer reviews from Grok-3 (statistical methodology) and GLM-5.1 (NLP/systems).

## Current Status
- Phase 5.5: 4,800 runs complete, analysis done, paper drafted (6 pages, ACL format)
- Main finding: 1.19pp interference effect, NOT significant (p=0.26, d=0.056)
- TOST equivalence at ±3pp (p=0.035), confidence drop significant (p=0.003)
- Paper at: paper/paper.tex (compiled PDF exists)

---

## Category A: Statistical & Analysis Improvements (mostly DONE)
These were recommendations from the Grok-3 review. Most are already implemented.

| # | Recommendation | Status | Effort |
|---|---------------|--------|--------|
| A1 | TOST equivalence testing ±3pp and ±5pp | ✅ Done | — |
| A2 | Bayesian analysis (BF=0.038) | ✅ Done | — |
| A3 | Bootstrap 95% CIs | ✅ Done | — |
| A4 | Difficulty-stratified analysis (discriminating tasks) | ✅ Done | — |
| A5 | Confidence calibration analysis | ✅ Done | — |
| A6 | Token cost analysis (2.1-2.4x) | ✅ Done | — |
| A7 | Sensitivity analysis (exclude outlier, exclude GPT-4o-mini) | ✅ Done | — |
| A8 | MDE table | ✅ Done | — |
| A9 | Heterogeneity test (Cochran's Q, I²) | ✅ Done | — |
| A10 | GLMM instead of LMM (binary outcome) | ⬜ Not done | LOW |
| A11 | Exclude GPT-4o-mini as "robust model" subgroup | ✅ Done (sensitivity table) | — |
| A12 | Per-model subgroup analysis | ✅ Done | — |

---

## Category B: Paper Improvements (in progress)

| # | Recommendation | Status | Effort |
|---|---------------|--------|--------|
| B1 | Add figures (bar chart, forest plot, calibration plot) | ⬜ Not done | MEDIUM |
| B2 | Related work section | ⬜ Not done | MEDIUM |
| B3 | Tighter writing (cut filler, vary rhythm) | ⬜ Not done | LOW |
| B4 | Reframe title/abstract as null result | ⬜ Partially done | LOW |
| B5 | Discuss ceiling effect explicitly | ✅ Done | — |
| B6 | Fix 3 mislabeled tasks (exclude or correct) | ✅ Done (excluded) | — |
| B7 | Confidence as primary finding | ✅ Done | — |
| B8 | Token cost as practical significance | ✅ Done | — |

---

## Category C: New Experiments (HIGH impact, MEDIUM-HIGH effort)

These are the most impactful changes that would require re-running experiments.

| # | Recommendation | Impact | Effort | Priority |
|---|---------------|--------|--------|----------|
| C1 | **Harder task suite** (target 70-80% baseline, not 95%) | HIGH — addresses ceiling effect | MEDIUM (write new tasks, ~1 day) | **#1** |
| C2 | **Track A: Gold retrieval** (inject only gold skill) | HIGH — separates retrieval from selection | LOW (code exists from Phase 5.3, ~2 hours to run) | **#2** |
| C3 | **Track B: RAG retrieval** (top-K with sentence-transformers) | HIGH — measures retrieval precision | LOW (code exists from Phase 5.3, ~2 hours to run) | **#3** |
| C4 | **Gradient test** (3 similarity levels × 10 pairs) | MEDIUM — validates near-duplicate taxonomy | MEDIUM (task design + running, ~1 day) | **#4** |
| C5 | **Invocation/parameter accuracy** | MEDIUM — tests staleness claim | MEDIUM (new metric extraction, ~4 hours) | **#5** |
| C6 | **Multi-step evaluation** (30 tasks, cascade failures) | MEDIUM — tests compounding claim | LOW (code exists from Phase 5.4, ~2 hours to run) | **#6** |
| C7 | **Higher temperatures** (0.9, 1.0) | LOW — unlikely to change result | LOW (add 2 temp levels, ~4 hours API time) | **#7** |

---

## Category D: Design Fixes (address reviewer concerns)

| # | Recommendation | Impact | Effort | Priority |
|---|---------------|--------|--------|----------|
| D1 | **Fix oracle condition** (show only gold skill, not all 40 clean skills) | HIGH — current oracle is not a true oracle | LOW (modify prompt construction, ~1 hour) | **#1** |
| D2 | **Reduce library interference density** (from 22% to ~5%) | MEDIUM — addresses ecological validity | LOW (subsample skills, ~1 hour) | **#2** |
| D3 | **Human annotation for 3 ambiguous tasks** | LOW — already excluded | LOW (relabel or keep excluded) | **#3** |
| D4 | **Verify task-keyword overlap** (93.8% have skill verbs in description) | MEDIUM — addresses keyword-matching confound | MEDIUM (task redesign or keyword-stripped condition) | **#4** |

---

## Execution Plan

### Phase 6.1: Quick wins (1-2 days)
- [ ] A10: Run GLMM (mixed-effects logistic regression) alongside LMM
- [ ] B1: Generate 3 figures (bar chart, forest plot, calibration curve)
- [ ] B2: Write related work section
- [ ] B3: Polish writing (directness, rhythm, density)
- [ ] B4: Finalize title/abstract framing

### Phase 6.2: Critical experiments (2-3 days)
- [ ] D1: Fix oracle condition → rerun on all 4 models (Track A)
- [ ] C2: Run Track A (gold retrieval) on all 4 models
- [ ] C3: Run Track B (RAG retrieval, K=1,3,5,10) on all 4 models
- [ ] C6: Run multi-step evaluation (30 tasks) on all 4 models

### Phase 6.3: Harder task suite (3-5 days)
- [ ] C1: Design 40-60 harder tasks (target 70-80% baseline)
  - Tasks where skill selection is genuinely ambiguous
  - Tasks without skill-name keywords in description
  - Multi-hop tasks requiring skill composition
- [ ] Run harder task suite on all 4 models
- [ ] Re-analyze with harder tasks

### Phase 6.4: Gradient and invocation analysis (2-3 days)
- [ ] C4: Implement gradient test (3 similarity × 10 pairs × 4 models)
- [ ] C5: Extract invocation/parameter accuracy from existing responses
- [ ] C7: (Optional) Add higher temperatures

### Phase 6.5: Paper finalization (2-3 days)
- [ ] Integrate all new results
- [ ] Write complete discussion section
- [ ] Address all reviewer concerns
- [ ] Final proofreading and formatting
- [ ] Submit to arXiv preprint

---

## Cost Estimates

| Phase | API Calls | Est. Cost |
|-------|-----------|-----------|
| 6.1 (Analysis + Figures) | 0 | $0 |
| 6.2 (Track A/B + Multi-step) | ~4,800 | ~$12 |
| 6.3 (Harder task suite) | ~6,000-9,000 | ~$15-25 |
| 6.4 (Gradient + Invocation) | ~2,400 | ~$6 |
| **Total** | ~13,000-16,000 | ~$33-43 |

---

## Decision Points

1. **Do we run Track A/B/C experiments?** The retrieval code exists from Phase 5.3. This is the single most important gap — reviewers will ask "can you separate retrieval from selection?" Running Track A and Track B takes ~4 hours and addresses this directly.

2. **Do we build a harder task suite?** The ceiling effect (83% tasks identical CM/CI) is the biggest threat. Harder tasks would make interference effects detectable if they exist. But this requires genuine task design effort.

3. **Do we fix the oracle condition?** Currently "oracle" shows 40 clean skills (same as clean_memory). A true oracle shows only the gold skill. This is easy to fix but changes the condition meaning.

4. **What's the target venue?** ACL/EMNLP requires the most rigor. arXiv preprint is fine with current data. NeurIPS/ICML would focus more on the theoretical contribution.

5. **Timeline?** Phase 6.1-6.2 can be done in 3-4 days. Phase 6.3-6.4 adds another week. Total: ~2 weeks for a publication-ready paper.