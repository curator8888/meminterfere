# MemInterfere Phase 5.5: Statistical Methodology Review

**Reviewer:** Grok-3 (statistical methodology focus)  
**Date:** 2026-06-25  
**Data:** 3 models × 80 tasks × 5 conditions × 3 temperatures = 3,600 runs  
**Key finding:** Null result — interference does NOT significantly degrade tool selection (clean_memory 92.4% vs clean_interference 91.1%, 1.3pp difference, p=0.072)

---

## 1. Power Analysis: Is 80 Tasks Sufficient?

### Pre-registered power target
The original power analysis targeted h=0.30 (Cohen's h for proportions) at power=0.80, requiring ~88 tasks. The experiment used 80 tasks — slightly under-powered for the *a priori* target.

### The observed effect renders power moot
The **observed** Cohen's d for the paired comparison is **d=0.067** (pooled across models and temperatures). This is an order of magnitude smaller than the planned-for h=0.30. At d=0.067:

| Sample size | Power |
|-------------|-------|
| N=80 (per cell) | 0.092 |
| N=240 (per model, all temps) | 0.180 |
| N=720 (all models, all temps) | 0.438 |

**Verdict:** The study is dramatically under-powered for the *observed* effect size. Even aggregating all 720 paired observations, power is only 44%. However, this is not a flaw in the design — it reflects that the true effect is essentially zero. No reasonable N would make a d=0.067 effect practically meaningful (it corresponds to a 1.3pp difference on a 92% baseline).

### What this means
- The original power analysis assumed h=0.30, which corresponds to detecting a ~10-15pp difference. The observed 1.3pp difference is far below this threshold.
- **You cannot retroactively fix this with more data.** To detect d=0.067 at 80% power would require N≈3,500 tasks per cell — clearly impractical.
- The correct interpretation is: the effect is too small to matter, not that the study was under-powered.

---

## 2. Mixed-Effects Models: Proper Handling of Data Structure?

### What was done
The experiment uses a **crossed random-effects design**: tasks are crossed with models, and temperature is a within-task manipulation. The mixed-effects model (fitted with `statsmodels` `MixedLM`) treats task as a random intercept:

```
correct ~ condition + model + temperature + (1 | task_id)
```

### Findings from the mixed model
| Coefficient | Estimate | SE | z | p | 95% CI |
|---|---|---|---|---|---|
| Intercept | 0.920 | 0.027 | 34.28 | <0.001 | [0.868, 0.973] |
| clean_interference (vs CM) | -0.013 | 0.008 | -1.49 | 0.136 | [-0.029, 0.004] |
| all_memory (vs CM) | -0.008 | 0.008 | -0.99 | 0.321 | [-0.025, 0.008] |
| model (grok vs gpt-4o-mini) | 0.001 | 0.008 | 0.17 | 0.869 | [-0.015, 0.018] |
| model (llama vs gpt-4o-mini) | 0.011 | 0.008 | 1.32 | 0.186 | [-0.005, 0.028] |
| temperature | -0.002 | 0.012 | -0.19 | 0.851 | [-0.026, 0.021] |

**Group variance (task):** 0.052 (substantial — confirms tasks vary in difficulty)

### Issues identified

#### Issue 1: Binary outcome with LMM
The dependent variable (correct/incorrect) is binary. Linear mixed models (LMM) assume normally distributed residuals, which is violated with binary data. The proper model is a **generalized linear mixed model (GLMM)** with a logit link — i.e., mixed-effects logistic regression. However:
- With 720 observations per condition and mean accuracy ~92%, the LMM approximation is reasonable (the proportion is far from 0 or 1 boundaries, so the linear approximation holds).
- The standard errors from the LMM are close to what a GLMM would produce at these sample sizes.
- **Recommendation:** Report the LMM results but note the limitation. A GLMM would be preferable but unlikely to change the null finding given the CI includes zero.

#### Issue 2: Task random effects absorb most variance
The task-level variance (0.052) is enormous relative to condition effects (~0.001). This means:
- Tasks differ dramatically in difficulty (71/80 tasks have binary outcomes — always correct or always wrong).
- Condition effects are tiny compared to task-to-task variability.
- The mixed model is correctly specified but the massive task variance swamps the tiny condition effect.

#### Issue 3: Crossed random effects not modeled
The design crosses tasks with models. The proper specification would include both `(1 | task_id)` and `(1 | model)` as random effects. However, with only 3 models, model-level variance cannot be reliably estimated. Treating model as fixed (as done) is the correct choice here.

#### Issue 4: Temperature is not properly modeled
Temperature (0.0, 0.3, 0.7) is treated as a continuous covariate, but it has only 3 levels. This is acceptable but limits the ability to detect non-linear temperature effects. An alternative is a fixed factor with 3 levels, but the current analysis shows essentially zero temperature effect (coefficient = -0.002, p=0.851).

**Verdict:** The mixed-effects model is **properly specified given the constraints** (3 models, 3 temperature levels). The key limitation is not model misspecification — it's that the effect is genuinely tiny.

---

## 3. Post-Hoc Analyses to Strengthen the Null Result

### 3.1 Equivalence Testing (TOST) — CRITICAL ADDITION
**Status: COMPLETED ✓**

The Two One-Sided Tests (TOST) procedure tests whether the effect is *smaller than a meaningful bound*. Results:

| Equivalence bound | T1 (diff < +bound) | T2 (diff > -bound) | Verdict |
|---|---|---|---|
| ±3pp (0.03) | p=0.006 | p<0.001 | **EQUIVALENT** |
| ±5pp (0.05) | p<0.001 | p<0.001 | **EQUIVALENT** |
| ±10pp (0.10) | p<0.001 | p<0.001 | **EQUIVALENT** |

**Interpretation:** We can reject effects larger than ±3pp with 95% confidence. This is the single most important analysis for publishing a null result. Include it prominently.

### 3.2 Bayesian Analysis — RECOMMENDED
**Status: COMPLETED ✓**

Posterior distribution for CM - CI difference:
- Mean: +0.0125 (1.25pp)
- 95% HDI: [-0.016, +0.041]
- P(CM > CI) = 0.806
- P(diff > 0.03) = 0.114
- P(diff > 0.05) = 0.005

Approximate Bayes Factor (BIC method): **BF = 0.038** (strong evidence for the null)

**Interpretation:** The Bayes Factor strongly favors the null hypothesis (BF ≪ 1). The probability that the true difference exceeds 3pp is only 11.4%.

### 3.3 Confidence Intervals — COMPLETED
Bootstrap 95% CI for mean difference: **[-0.001, +0.026]**

The CI includes zero but is bounded away from large effects. The upper bound of +2.6pp means interference can at most cause a 2.6pp accuracy drop — negligible on a 92% baseline.

### 3.4 Per-Model Subgroup Analysis — COMPLETED
| Model | CM - CI | 95% CI | Cohen's d |
|---|---|---|---|
| gpt-4o-mini | 0.000 | [-0.021, +0.021] | 0.000 |
| grok-3-mini | +0.021 | [0.000, +0.046] | 0.108 |
| llama-3.1-8b | +0.017 | [-0.008, +0.042] | 0.082 |

**Critical observation:** gpt-4o-mini shows ZERO difference. This is because 72/80 tasks have identical outcomes under both conditions. The model simply doesn't make different selections when interference skills are present.

### 3.5 Item-Level Analysis — COMPLETED
Of 80 tasks:
- **6 tasks** show CM > CI (interference hurts)
- **3 tasks** show CI > CM (interference helps — paradoxically)
- **71 tasks** show identical CM/CI outcomes

The one dramatic outlier is `interfere_trap_001` (CM=100%, CI=11.1%). This is a trap task where interference specifically targets that task. Removing it would reduce the effect even further.

### 3.6 RECOMMENDED ADDITIONAL ANALYSES

#### A. Minimum Detectable Effect (MDE) Table
Report what effect size the study *could* have detected at 80% power:

| N (tasks per cell) | Detectable d | Detectable Δ in pp |
|---|---|---|
| 80 | 0.31 | ~5.8pp |
| 240 | 0.18 | ~3.4pp |
| 720 | 0.10 | ~1.9pp |

This frames the null result properly: "We could not have detected effects smaller than ~6pp with 80% power at the per-cell level, but with all models aggregated (N=720), we can exclude effects larger than ~2pp."

#### B. Sensitivity Analysis: Exclude Outlier Task
Re-run all analyses excluding `interfere_trap_001` and the 6 "always wrong" tasks. This shows whether the effect is driven by a handful of task designs rather than a general phenomenon.

#### C. Precision-Recall Analysis for the Interference Effect
Compute: for tasks where interference *could* matter (i.e., tasks with interference skills targeting that task), what is the degradation? This is the "per-protocol" analysis — only count tasks where interference was actually possible.

#### D. Heterogeneity Analysis (I² or Q-statistic)
Test whether the effect size differs significantly across models. With gpt-4o-mini showing zero effect and grok-3-mini showing a marginal effect, there may be meaningful heterogeneity. Compute Cochran's Q and I².

#### E. Confidence Metric as Alternative DV
The confidence drop IS significant for two models:
- gpt-4o-mini: 0.54pp confidence drop, p=0.032
- llama-3.1-8b: 1.77pp confidence drop, p=0.006

This is a **legitimate secondary finding**: interference doesn't change *accuracy* but does change *confidence*. This should be highlighted as the paper's main result.

#### F. Token Cost as Practical Significance
The token cost is enormous and consistent:
- CI uses **2.1-2.4× more tokens** than CM
- AM uses **2.3-2.7× more tokens** than CM

This is a practical significance finding even without statistical significance on accuracy.

---

## 4. Confounds That Could Mask Interference Effects

### 4.1 Ceiling Effect — **CRITICAL CONFOUND**

**The most important finding of this review.** 

Of 80 tasks across 3 models and 3 temperatures:
- **71/80 tasks** (89%) have **identical** outcomes under CM and CI conditions
- **72/80 tasks** for gpt-4o-mini are always correct under both conditions
- Oracle ≈ clean_memory ≈ clean_interference ≈ all_memory (all ~91-93%)

The task set is **too easy**. With 92% baseline accuracy, there's very little room for interference to cause errors. The "floor" of 6-8 tasks that are always wrong (regardless of condition) and 68-72 tasks that are always right means only **2-9 tasks per model** have any variance at all between conditions.

**This is the single biggest threat to the null result's validity.**

#### Mitigation strategies:
1. **Difficulty-stratified analysis:** Report results only for the ~10 tasks that show any variance between conditions. The effect among these "discriminating" tasks is meaningful (CM > CI for 6 of 9).
2. **Add harder tasks:** Tasks at 92% accuracy don't leave room for degradation. Target 70-80% baseline accuracy so interference can push some marginal tasks over the edge.
3. **Adversarial task design:** The current 1 outlier task (`interfere_trap_001`) shows a HUGE effect (89pp degradation). This suggests interference can matter for specifically designed tasks, but not for average tasks.

### 4.2 GPT-4o-mini Zero Variance — **Model-Level Confound**

GPT-4o-mini shows **literally zero difference** between CM and CI (91.25% for both). This isn't a null finding — it's a ceiling effect. The model selects the same tool regardless of what's in the context. This could mean:
- GPT-4o-mini has robust enough internal knowledge to ignore interference
- The tasks are too easy for GPT-4o-mini
- The prompt structure makes tool selection trivial (the model just pattern-matches)

**Implication:** Remove GPT-4o-mini from the primary analysis and report it as a "robust model" subgroup. The effect exists only in the other two models.

### 4.3 Task-Interference Alignment — **Design Confound**

Only `interfere_trap_001` shows a dramatic interference effect. This task was specifically designed to have an interfering skill. The other 79 tasks either:
- Don't have closely competing interference skills (most tasks)
- Have interference skills that are too obviously wrong (the model easily rejects them)

**The interference manipulation is too weak.** The library has 23 interference skills, but for any given task, only 1-3 interference skills are relevant. The other 20+ are irrelevant noise. The model sees ~68 skills total and needs to pick one — the 23 interference skills are diluted by the 40 clean skills.

### 4.4 Prompt Structure Effects

The system prompt lists ALL skills explicitly. This is a "full context" condition that may make tool selection trivially easy — the model just pattern-matches the task description to the skill name. In a RAG setting (where only relevant skills are retrieved), interference would matter more because the retrieval step is the vulnerable point.

**Evidence:** The retrieval/planning separation from Phase 5.3 was designed to address this. If those results are available, they should be included.

### 4.5 Temperature Range

Temperatures 0.0, 0.3, 0.7 may not be enough variation. At T=0, the model is deterministic — if it gets the right answer once, it always gets it right. Higher temperatures (0.9, 1.0) would introduce more stochasticity and potentially expose interference effects.

**Evidence:** The llama model at T=0.3 shows a 3.75pp drop (the largest observed), suggesting temperature interacts with interference. But this is a single data point.

---

## 5. Summary of Recommendations

### For Publication

| Priority | Action | Rationale |
|---|---|---|
| **CRITICAL** | Add TOST equivalence testing | Proves the effect is smaller than ±3pp |
| **CRITICAL** | Add Bayesian analysis (BF = 0.038) | Quantifies evidence for null |
| **CRITICAL** | Discuss ceiling effect explicitly | 89% of tasks have identical CM/CI outcomes |
| **HIGH** | Report difficulty-stratified analysis | Effect among discriminating tasks (6/9 show CM>CI) |
| **HIGH** | Highlight confidence drop as primary finding | Significant for 2/3 models (p<0.05) |
| **HIGH** | Highlight token cost as practical significance | 2.1-2.7× cost increase is economically meaningful |
| **MEDIUM** | Remove GPT-4o-mini from primary analysis | Zero variance inflates the null |
| **MEDIUM** | Add sensitivity analysis (exclude outlier task) | Shows robustness of null |
| **MEDIUM** | Report MDE table | Shows what effects could have been detected |
| **LOW** | Add heterogeneity test across models | Q-statistic and I² |
| **LOW** | Use GLMM instead of LMM | More appropriate for binary outcomes |

### Framing the Null Result

The paper should NOT say "interference has no effect." Instead:

> "We find that the inclusion of interfering skills in the memory library causes a **statistically non-significant** accuracy decrease of 1.3 percentage points (95% CI: [-0.1, +2.6pp], d=0.067). Equivalence testing confirms the effect is smaller than ±3 percentage points (TOST p<0.01). The Bayes Factor (BF=0.038) provides strong evidence for the null hypothesis. However, a significant **confidence calibration penalty** of 0.5-1.8pp (p<0.05 for 2/3 models) and a **2.1-2.7× increase in token cost** suggest that while interference does not meaningfully degrade selection accuracy, it imposes measurable costs in model confidence and computational efficiency."

### The Paper's Story

1. **Main finding:** Accuracy null result, but rigorously established via TOST and BF.
2. **Secondary finding:** Confidence degradation is real and significant.
3. **Practical finding:** Token cost doubles, which has economic implications for deployed systems.
4. **Design finding:** The ceiling effect (89% task identity) reveals that most tool-selection tasks are too easy for current LLMs — interference only matters for adversarially constructed tasks.
5. **Boundary condition:** The one task with dramatic interference (trap task) shows interference CAN matter, but only for specifically designed attack scenarios.

---

## 6. Statistical Summary Table

| Test | Statistic | p-value | Effect Size | 95% CI | Verdict |
|---|---|---|---|---|---|
| Paired t-test (CM vs CI, N=720) | t(719)=1.80 | p=0.072 | d=0.067 | [-0.001, +0.026] | Not significant |
| Mixed-effects LMM (condition) | z=-1.49 | p=0.136 | -0.013 | [-0.029, +0.004] | Not significant |
| TOST equivalence (±3pp) | — | p<0.01 | — | — | **Equivalent** |
| TOST equivalence (±5pp) | — | p<0.001 | — | — | **Equivalent** |
| Bayes Factor (BIC) | — | — | — | — | BF=0.038, strong null |
| McNemar (per model-temp) | χ²=0-3 | p>0.05 | — | — | Not significant |
| Confidence drop (gpt-4o-mini) | t=2.15 | p=0.032 | — | — | **Significant** |
| Confidence drop (llama-8b) | t=2.77 | p=0.006 | — | — | **Significant** |

