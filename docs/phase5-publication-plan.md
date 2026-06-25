# MemInterfere Publication Plan

**Goal:** Publish "The Anatomy of Tool Interference: Why LLM Agents Fail at the Margins" at a top venue (ACL/ICML/NeurIPS workshop or EMNLP main).

**Current state:** 60 valid runs, 1 model (Grok-3-mini), null result on main hypothesis, 3 peripheral findings.

**Publication requirements:** 300-500 runs, 3+ models, retrieval/planning separation, multi-step workflows, real-world libraries, formal taxonomy.

---

## Phase 5: Publication-Ready Experiment (6 phases)

### 5.1: Formal Near-Duplicate Taxonomy & Library Expansion

**Problem:** We have 5 near-duplicate pairs but no formal definition of "nearness."

**Actions:**
1. Define a **Semantic Similarity Spectrum** for skill names:
   - **Identical** (same name, different schema): `create_event` vs `create_event` (v1/v2)
   - **Near-identical** (synonym names, same function): `add_event` vs `create_event`
   - **Near-similar** (overlapping scope): `browse_website` vs `navigate_url`
   - **Distinct** (different names, different scope): `search_web` vs `send_email`

2. Expand library to **100 skills** with controlled near-duplicate density:
   - 40 clean skills (20 web, 20 API)
   - 20 near-identical duplicates (synonym pairs)
   - 15 near-similar duplicates (overlapping scope)
   - 10 schema conflicts (same name, different params)
   - 10 semantic conflicts (same name, different return format)
   - 5 version conflicts (v1/v2 of same API)

3. Create a **gradient test**: for each near-duplicate pair, vary the name similarity:
   - `create_event` vs `add_calendar_event` (high similarity)
   - `create_event` vs `schedule_meeting` (medium similarity)
   - `create_event` vs `book_appointment` (low similarity)
   This isolates the effect of name similarity on selection error rate.

4. Add **5 real-world skill libraries** sampled from:
   - Zapier API directory (public tool descriptions)
   - OpenAI function calling examples
   - Anthropic tool use examples
   - Home Assistant integrations
   - RapidAPI popular APIs
   Each provides 10-20 real tool definitions with genuine naming collisions.

**Deliverable:** `data/skills/expanded_library_100.json`, `docs/near-duplicate-taxonomy.md`

**Verification loop:**
- [ ] Every near-duplicate pair has a graded similarity score (0-1)
- [ ] Every conflict group has ≥2 members
- [ ] Real-world skills have source URLs documented
- [ ] Library validates with `python src/evaluate_agent.py --validate`

---

### 5.2: Multi-Model Evaluation Harness

**Problem:** Llama-3.1-8b failed format parsing. Need robust parsing + 4 models.

**Actions:**
1. Build `src/multi_model_runner.py`:
   - Support 4 model backends: xAI (Grok-3-mini), OpenRouter (Llama-3.1-70b, GPT-4o-mini), Anthropic (Claude-3.5-Haiku)
   - Robust response parsing: regex extraction for TOOL/CONFIDENCE/APPROACH lines + fallback extraction from freeform text
   - Temperature sweep: 0.0, 0.3, 0.7 (to measure stochastic effects)
   - Retry logic with exponential backoff
   - Structured JSON output with metadata (model, temp, latency, tokens)

2. Fix format compliance for smaller models:
   - Use few-shot prompting: include 3 examples in system prompt
   - Add `response_format: { type: "json_object" }` for models that support it
   - Fallback: parse any line starting with a skill name from the provided list
   - If parsing fails after 2 retries, mark as PARSE_ERROR and skip (not 0%)

3. Model selection (4 tiers):
   - **Small:** Llama-3.1-8b-instruct (free, tests weakest model)
   - **Medium:** Grok-3-mini ($0.30/M, our baseline)
   - **Capable:** GPT-4o-mini ($0.15/M, industry standard)
   - **Strong:** Claude-3.5-Haiku ($0.25/M, different architecture)

4. Cost estimation:
   - 500 tasks × 5 conditions × 3 temperatures × 4 models = 30,000 runs
   - But: conditions and temperatures are orthogonal, so 500 × 5 = 2,500 per model × 4 = 10,000 total runs
   - Temperature sweep: 10,000 × 3 = 30,000 total API calls
   - At ~500 tokens/call: ~15M tokens total
   - Cost: ~$3-5 across all providers

**Deliverable:** `src/multi_model_runner.py`, `src/response_parser.py`

**Verification loop:**
- [ ] Parse rate ≥95% across all 4 models (currently 0% for Llama)
- [ ] Each model produces correct tool names for ≥80% of clean-only tasks
- [ ] Latency per call <10s for all models
- [ ] Cost per run documented and under $0.01

---

### 5.3: Retrieval vs. Planning Separation

**Problem:** We don't know if failures are retrieval failures (wrong skill fetched) or planning failures (wrong skill chosen from correct context).

**Actions:**
1. Implement **Track A: Gold-Standard Retrieval**:
   - Inject the exact correct skill into the prompt (no retrieval step)
   - Test: given the correct skill, can the agent use it correctly?
   - This isolates planning failures from retrieval failures

2. Implement **Track B: RAG Retrieval**:
   - Use cosine similarity to retrieve top-K skills from the library
   - Test: can the agent select the right skill from the retrieved set?
   - This measures retrieval + planning combined

3. Implement **Track C: Full Context** (current design):
   - Show all skills in the library
   - Test: can the agent select from the full set?

4. For each track, measure:
   - **Selection accuracy:** correct tool chosen
   - **Invocation accuracy:** correct tool + correct parameters
   - **Confidence calibration:** ECE (Expected Calibration Error)
   - **Latency:** time to first token, total response time

5. Implement `src/retrieval_simulator.py`:
   - Cosine similarity over skill embeddings (use sentence-transformers)
   - Top-K retrieval with K ∈ {1, 3, 5, 10}
   - Measure retrieval precision (fraction of correct skills in top-K)

**Deliverable:** `src/retrieval_simulator.py`, updated `src/evaluate_agent.py` with Track A/B/C

**Verification loop:**
- [ ] Track A (gold standard) shows ≥95% selection accuracy (baseline)
- [ ] Track B (RAG) retrieval precision ≥80% at K=5
- [ ] Track C (full context) matches current experiment results
- [ ] Failure attribution: can we say "X% of errors are retrieval vs. planning"?

---

### 5.4: Multi-Step Agentic Workflows

**Problem:** Single-tool selection doesn't reflect real agent behavior. Real agents compose multiple tools in sequences.

**Actions:**
1. Design **30 multi-step tasks** requiring 2-5 tool calls:
   - "Search for the weather in London, then send an email with the forecast"
   - "List calendar events for this week, find conflicts, and reschedule"
   - "Scrape product prices from 3 sites, compare, and create a summary"

2. Each multi-step task has:
   - A required tool sequence (e.g., `search_web` → `summarize_page` → `send_email`)
   - Multiple interference points (conflicting skills at each step)
   - A success criterion (correct sequence + correct parameters)

3. Evaluate **compositional failure**:
   - Does step-1 interference cascade to step-2?
   - Does having the wrong skill at step-1 prevent step-2 from executing?
   - Does parameter corruption at step-1 propagate?

4. Add a **chain error metric**:
   - Position accuracy: correct tool at each position
   - Sequence accuracy: correct order of tools
   - Parameter accuracy: correct parameters at each step
   - Cascade rate: P(wrong step-2 | wrong step-1) vs P(wrong step-2 | correct step-1)

**Deliverable:** `data/tasks/multistep_tasks.json`, `src/multistep_evaluator.py`

**Verification loop:**
- [ ] Each multi-step task has a clear gold sequence
- [ ] Cascade rate >0% (interference propagates between steps)
- [ ] At least 5 tasks per difficulty level (easy/medium/hard)

---

### 5.5: Scaled Experiment Execution

**Problem:** Need 300-500 runs per model for statistical power.

**Actions:**
1. **Power analysis** (do this FIRST):
   - Current effect size: Cohen's h = 0.00 (no difference between conditions)
   - For the near-duplicate effect (10% error): Cohen's h ≈ 0.30 (small)
   - To detect h=0.30 at α=0.05, power=0.80: need ~175 runs per condition per model
   - 5 conditions × 175 runs = 875 runs per model × 4 models = 3,500 total runs
   - BUT: we can stratify and reuse tasks across conditions, so need 175 unique tasks × 5 conditions = 875 per model

2. Expand task set to **175 tasks**:
   - 50 easy (web 25, API 25)
   - 75 medium (web 35, API 35, multi-step 5)
   - 50 hard (web 20, API 20, multi-step 10)

3. Run the full experiment:
   - 4 models × 175 tasks × 5 conditions × 3 temperatures = 10,500 runs
   - Batch into 7 runs of ~1,500 each (to manage rate limits)
   - Each batch takes ~30 minutes (0.5s per call + rate limiting)
   - Total time: ~3.5 hours

4. Statistical analysis:
   - Paired t-tests for each model (task-level paired across conditions)
   - Mixed-effects ANOVA with model as random effect
   - Effect sizes (Cohen's d) for each comparison
   - Confidence intervals via bootstrap (10,000 iterations)
   - Bonferroni correction for multiple comparisons

**Deliverable:** `data/results/phase5_full_experiment.json`, `src/statistical_analysis.py`

**Verification loop:**
- [ ] Power analysis confirms 175 tasks is sufficient
- [ ] All 4 models have ≥95% parse rate
- [ ] Paired t-tests computed for all condition pairs
- [ ] Effect sizes and confidence intervals reported

---

### 5.6: Paper Writing & Submission

**Actions:**
1. Write the paper in LaTeX (ACL/EMNLP format):
   - **Title:** "The Anatomy of Tool Interference: Why LLM Agents Fail at the Margins"
   - **Abstract:** 250 words
   - **Sections:**
     1. Introduction (1 page)
     2. Related Work (1 page): tool use, skill retrieval, agent memory
     3. Interference Taxonomy (1.5 pages): 5 types with formal definitions
     4. MemInterfere Benchmark (1.5 pages): library design, task design, metrics
     5. Experiments (3 pages): 4 models, 5 conditions, 175 tasks, 3 tracks
     6. Results (2 pages): main findings, per-model analysis, failure analysis
     7. Discussion (1.5 pages): implications, limitations, future work
     8. Conclusion (0.5 pages)
   - **Appendix:** full skill library, task descriptions, statistical tables

2. Target venues (in order of preference):
   - **EMNLP 2027** (main conference, deadline ~June 2027)
   - **NeurIPS 2026** (workshop, deadline ~May 2026)
   - **ACL 2027** (main conference, deadline ~January 2027)
   - **ICML 2027** (main conference, deadline ~January 2027)
   - ArXiv preprint as fallback

3. Submission checklist:
   - [ ] All experiments reproducible (code + data on GitHub)
   - [ ] Statistical tests reported with p-values and effect sizes
   - [ ] Limitations section acknowledging single-domain library
   - [ ] Ethics statement on potential misuse
   - [ ] No PII in data, all skills are synthetic/derived from public APIs

**Deliverable:** `paper/` directory with LaTeX source, `paper/figures/` with all plots

---

## Verification Loop Additions (from peer review gap analysis)

### V1: Confidence Calibration (ECE)
Add Expected Calibration Error (ECE) metric to `src/statistical_analysis.py`:
- Bin predictions into 10 confidence buckets (0-10%, 10-20%, ..., 90-100%)
- For each bucket, compute |avg_confidence - accuracy|
- ECE = weighted average across buckets
- Report ECE per condition and per model
- This addresses GLM-5.1's concern about unexplored confidence drops

### V2: Verification Prompt Condition
Add a 6th experimental condition: **verification_prompt**
- System prompt includes: "Before using any tool, verify it is correct by checking: (1) Does the tool name exactly match what you need? (2) Are the parameters correct? (3) Is there a newer or more reliable version available?"
- Compare trap acceptance rate with and without verification
- This addresses the "trap acceptance is a design artifact" concern
- If verification reduces trap acceptance from 100% to <50%, it's a systems-level finding (verification loops work)
- If verification doesn't help, it's an LLM-level finding (skeptical prompting doesn't overcome deceptive descriptions)

### V3: Task Difficulty Filter
Before running scaled experiment, verify each task:
- Run each task in the NO_MEMORY condition
- If the model answers correctly without any tool library → task doesn't require memory
- Mark these tasks and exclude from primary analysis (report separately)
- Target: ≥50% of tasks should REQUIRE the skill library
- This addresses GLM-5.1's concern about easy tasks washing out interference effects

### V4: Per-Model Power Analysis
After pilot (5.2), run a mini power analysis per model:
- 20 tasks × 5 conditions = 100 runs per model
- Compute per-model effect sizes
- Adjust task count per model based on effect size
- Smaller/weaker models may need more tasks to detect effects

### V5: Statistical Rigor
In `src/statistical_analysis.py`:
- Bonferroni correction for multiple comparisons (5 conditions × 4 interference types = 20 comparisons)
- Bootstrap confidence intervals (10,000 iterations)
- Effect sizes (Cohen's d) for all pairwise comparisons
- Mixed-effects ANOVA with model as random effect
- Bayesian t-tests for robustness

### V6: Reproducibility Package
Create before submission:
- `Dockerfile` with all dependencies
- `requirements.txt` with pinned versions
- `run_experiment.sh` that reproduces all results
- `README.md` with step-by-step reproduction instructions
- `LICENSE` (Apache 2.0 or MIT)

---

## Execution Timeline

| Phase | Duration | Depends On | Estimated Cost | Verification Gate |
|-------|----------|-----------|---------------|-------------------|
| 5.1 Taxonomy & Library | 2 days | None | $0 | V3: ≥50% tasks require library |
| 5.2 Multi-Model Harness | 1 day | None | $0.50 | V4: ≥95% parse rate per model |
| 5.3 Retrieval/Planning | 2 days | 5.1 | $0 | Track A ≥95% baseline accuracy |
| 5.4 Multi-Step Tasks | 1 day | 5.1 | $0 | V7: cascade rate >0% |
| 5.5 Scaled Experiment | 1 day | 5.1-5.4 | $5.00 | V5: p<0.05 for near-duplicate effect |
| 5.6 Paper Writing | 5 days | 5.5 | $0 | V6: reproducibility package |
| **Total** | **~12 days** | | **~$5.50** | |

---

## Verification Checkpoints

After each phase, verify:

1. **5.1 → 5.2:** Library validates, taxonomy is formal, real-world skills included
2. **5.2 → 5.3:** Parse rate ≥95% across all models, baseline accuracy ≥80%
3. **5.3 → 5.4:** Track A/B/C separation produces distinct failure profiles
4. **5.4 → 5.5:** Multi-step tasks show cascade errors, not just selection errors
5. **5.5 → 5.6:** Statistical significance achieved for at least near-duplicate effect, confidence intervals narrow

**Go/No-Go after 5.5:**
- If near-duplicate effect size > Cohen's d=0.30 with p<0.05 → proceed to paper
- If effect sizes are all <0.20 → pivot to survey paper (still publishable, but different framing)
- If models disagree significantly → add model architecture analysis as contribution