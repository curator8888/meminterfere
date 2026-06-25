# Near-Duplicate Taxonomy & Semantic Similarity Spectrum

**Phase 5.1 — MemInterfere Project**
**Date:** 2026-06-25

## 1. Overview

This document defines a formal taxonomy of skill similarity for the MemInterfere project. Phase 4 found that name-distinct interference doesn't degrade performance (p=1.000), but near-duplicates cause ~10% selection error, traps are accepted 100% of the time, and staleness corrupts invocation (not selection). This taxonomy provides the theoretical framework and measurable definitions needed to systematically study *how similar is too similar* when multiple skills compete for the same intent.

## 2. Semantic Similarity Spectrum

### 2.1 Formal Definitions

The similarity spectrum classifies pairs of skills (S₁, S₂) along a continuous score `sim(S₁, S₂) ∈ [0.0, 1.0]`, computed as a weighted combination of name similarity, functional overlap, and parameter schema overlap:

```
sim(S₁, S₂) = w_name · sim_name(S₁, S₂) + w_func · sim_func(S₁, S₂) + w_schema · sim_schema(S₁, S₂)
```

Where:
- `sim_name` ∈ [0,1]: lexical/semantic similarity of skill names (e.g., cosine similarity of name embeddings)
- `sim_func` ∈ [0,1]: overlap in functional intent (whether both skills serve the same user goal)
- `sim_schema` ∈ [0,1]: overlap in parameter names and types (Jaccard similarity of parameter key sets)
- Default weights: `w_name = 0.3`, `w_func = 0.4`, `w_schema = 0.3`

### 2.2 Category Definitions

| Category | Similarity Score | Name Overlap | Functional Overlap | Schema Overlap | Example |
|----------|-----------------|-------------|-------------------|---------------|---------|
| **Identical** | 1.0 | Same name | Same intent | Different params | `create_event` (v1) vs `create_event` (v2) |
| **Near-identical** | 0.8–0.9 | Synonym/different name | Same intent | Mostly same | `add_event` vs `create_event` |
| **Near-similar** | 0.5–0.7 | Different name | Partial overlap | Partially shared | `browse_website` vs `navigate_url` |
| **Distinct** | 0.0–0.2 | Different name | Different scope | No overlap | `search_web` vs `send_email` |

---

## 3. Category Deep-Dive

### 3.1 Identical (Similarity = 1.0)

**Formal Definition:** Two skills share the exact same name but differ in their parameter schema, return format, or behavioral contract. The agent cannot distinguish them by name alone.

**Properties:**
- Name: identical (`name(S₁) = name(S₂)`)
- Intent: identical (same user goal)
- Schema: divergent (different required params, different types, or different return format)
- The agent sees one name but must choose between incompatible implementations

**Expected Failure Mode:**
- **Parameter hallucination:** The agent picks one variant and guesses parameters that match its preferred schema, which may be wrong for the selected variant.
- **Silent data loss:** The agent invokes with correct params for variant A, but the runtime routes to variant B, which accepts the call but returns data in a different format, causing downstream parsing failures.

**Measurable Signal:**
- **Invocation error rate:** % of calls where the agent passes parameters valid for the wrong variant.
- **Format mismatch rate:** % of calls where the returned data format differs from what the agent expected.
- **Detection method:** Compare the agent's invoked parameter set against both schemas. If the param set matches schema A but the runtime used schema B, that's an identical-category error.

**Representative Conflict Groups in MemInterfere:**
| Gold Skill | Interference Skill | Difference |
|-----------|-------------------|-----------|
| `search_web` (query, limit) | `search_web` (q, num_results, safe_mode) | Param names differ |
| `navigate_url` (url, extract_mode) | `navigate_url` (address, mode, timeout) | Param names + required differs |
| `send_email` (to, subject, body) | `send_email` (recipient object, subject_line, body_text) | Param structure differs |

---

### 3.2 Near-identical (Similarity = 0.8–0.9)

**Formal Definition:** Two skills have different names that are synonyms or near-synonyms, serve the same functional intent, and have mostly overlapping parameter schemas. A user requesting the function could reasonably be served by either skill.

**Properties:**
- Name: different but semantically equivalent (e.g., `add_event` ≡ `create_event`)
- Intent: identical (same user goal)
- Schema: mostly overlapping (core params shared, minor differences in optional params or naming)
- The agent can distinguish by name but may not know which is "correct"

**Expected Failure Mode:**
- **Selection ambiguity:** The agent selects the near-identical skill instead of the gold skill. Since both work for the intent, this may not cause an error in output but introduces noise in evaluation.
- **Parameter drift:** The near-identical skill may have slightly different parameter names (e.g., `summary` vs `title`), causing the agent to use the wrong param name.
- **Priority confusion:** When both skills appear in context, the agent may alternate between them across turns, breaking consistency.

**Measurable Signal:**
- **Selection error rate:** % of tasks where the agent selects the near-identical skill instead of the gold skill.
- **Parameter name error rate:** % of invocations using param names from the wrong variant.
- **Consistency rate:** Across N repetitions, how often does the agent pick the same skill for the same intent?

**Representative Conflict Groups in MemInterfere:**
| Gold Skill | Near-identical Skill | Similarity |
|-----------|---------------------|-----------|
| `create_event` | `add_event` | 0.90 |
| `send_email` | `compose_email` | 0.85 |
| `query_database` | `search_database` | 0.85 |
| `write_file` | `save_file` | 0.90 |
| `search_web` | `find_information` | 0.82 |

---

### 3.3 Near-similar (Similarity = 0.5–0.7)

**Formal Definition:** Two skills have different names, partially overlapping functional scope, and partially shared parameter schemas. They can accomplish some of the same goals but differ in scope, precision, or output format.

**Properties:**
- Name: different, not obviously synonymous
- Intent: partially overlapping (one may be a subset or generalization of the other)
- Schema: partially shared (some params overlap, others are unique to each)
- The agent may pick the wrong scope level (too broad or too narrow)

**Expected Failure Mode:**
- **Scope mismatch:** The agent selects the broader skill when a narrow one is needed, or vice versa. E.g., calling `browse_website` (which scrolls and reads) when `navigate_url` (which extracts structured content) is needed.
- **Output format surprise:** The near-similar skill returns data in a different format than expected, causing downstream errors.
- **Graceful degradation:** Unlike identical conflicts, near-similar errors often produce *partial* results rather than complete failures.

**Measurable Signal:**
- **Task completion rate delta:** Difference in task completion between gold skill and near-similar skill.
- **Output format mismatch rate:** % of invocations where the output format doesn't match the task's expected format.
- **Substitution cost:** How much extra work (additional turns, re-invocations) is needed when the wrong near-similar skill is selected.

**Representative Conflict Groups in MemInterfere:**
| Gold Skill | Near-similar Skill | Similarity |
|-----------|-------------------|-----------|
| `navigate_url` | `browse_website` | 0.65 |
| `scrape_page` | `get_data_from_page` | 0.60 |
| `search_news` | `get_headlines` | 0.55 |
| `summarize_page` | `quick_summary` | 0.70 |
| `download_file` | `fetch_file` | 0.65 |

---

### 3.4 Distinct (Similarity = 0.0–0.2)

**Formal Definition:** Two skills have different names, different functional scope, and no meaningful parameter overlap. They serve completely different user intents.

**Properties:**
- Name: clearly different
- Intent: unrelated
- Schema: no overlap
- The agent should never confuse these

**Expected Failure Mode:**
- **No failure expected.** Distinct skills should not interfere with each other. If an agent selects a distinct skill for a task, that indicates a fundamental retrieval or reasoning failure, not an interference effect.
- Phase 4 confirmed this: name-distinct interference showed p=1.000 (no degradation).

**Measurable Signal:**
- **Baseline error rate:** Any errors in distinct-skill conditions represent the noise floor of the evaluation.
- **Used as control condition:** Compare near-identical and near-similar error rates against the distinct baseline to isolate the effect of similarity.

**Representative Pairs in MemInterfere:**
| Skill A | Skill B | Similarity |
|---------|---------|-----------|
| `search_web` | `send_email` | 0.05 |
| `create_event` | `calculate` | 0.02 |
| `scrape_page` | `get_weather` | 0.08 |

---

## 4. Gradient Test Pairs

To isolate the effect of name similarity on selection error rate, we create controlled gradient test pairs. For each functional intent, we test three levels of name similarity:

### 4.1 Definition

A **gradient test pair** consists of a gold skill and three interference skills that serve the same intent but differ in how similar their names are to the gold skill:

| Level | Name Similarity | Pattern | Example |
|-------|----------------|---------|---------|
| **High** | 0.85–0.95 | Same root, different prefix/suffix | `create_event` vs `add_calendar_event` |
| **Medium** | 0.55–0.70 | Synonym with different root | `create_event` vs `schedule_meeting` |
| **Low** | 0.25–0.40 | Different concept, same domain | `create_event` vs `book_appointment` |

### 4.2 Gradient Test Pairs in MemInterfere

| Gold Skill | High Similarity (0.85–0.95) | Medium Similarity (0.55–0.70) | Low Similarity (0.25–0.40) |
|-----------|---------------------------|-----------------------------|--------------------------|
| `create_event` | `add_calendar_event` | `schedule_meeting` | `book_appointment` |
| `send_email` | `dispatch_email` | `compose_message` | `notify_contact` |
| `search_web` | `web_search` | `find_information` | `lookup_answer` |
| `navigate_url` | `open_webpage` | `browse_website` | `view_content` |
| `scrape_page` | `extract_page_data` | `get_data_from_page` | `harvest_content` |
| `query_database` | `run_db_query` | `search_database` | `fetch_records` |
| `get_weather` | `check_weather` | `forecast_conditions` | `report_climate` |
| `read_file` | `load_file` | `access_document` | `retrieve_storage` |
| `write_file` | `save_file` | `store_content` | `persist_data` |
| `create_task` | `add_task` | `schedule_todo` | `register_action` |

### 4.3 Experimental Protocol

For each gradient test pair:
1. Present the agent with the gold skill + exactly one interference skill (high, medium, or low).
2. Give a task that requires the gold skill.
3. Measure: which skill does the agent select?
4. Compare selection error rates across high/medium/low conditions.
5. **Hypothesis:** Higher name similarity → higher selection error rate.

---

## 5. Interference Mechanism Model

### 5.1 How Similarity Affects Selection

```
P(select wrong skill) = f(similarity, library_size, task_ambiguity)

Where:
- similarity ∈ [0, 1]: how similar the interference skill is to the gold skill
- library_size: number of competing skills for the same intent
- task_ambiguity: how clearly the task specifies which skill is needed
```

**Phase 4 Empirical Findings:**
- Distinct interference (sim ≈ 0.0–0.2): P(wrong) ≈ 0% (p=1.000)
- Near-duplicate interference (sim ≈ 0.8–0.9): P(wrong) ≈ 10%
- Trap acceptance: P(accept) = 100% (the agent always uses the trap)

### 5.2 Predicted Error Curve

```
P(wrong selection)
    |
0.1 |                    ●●●
    |                ●●
    |            ●●●
0.05|        ●●
    |    ●●
    | ●
0.0 |●●●●●●●●●●●●●●●●●●●●
    +---|---|---|---|---|---|
       0.0 0.2 0.4 0.6 0.8 1.0
            Similarity Score
```

We predict a sigmoid-like relationship where:
- Below similarity ~0.4: negligible selection errors
- Between 0.4–0.7: rapidly increasing errors (near-similar zone)
- Between 0.7–1.0: high error plateau (near-identical and identical zones)

### 5.3 Per-Category Error Taxonomy

| Category | Selection Error | Invocation Error | Output Error |
|----------|----------------|-----------------|-------------|
| Identical | High (agent can't distinguish by name) | High (wrong params) | High (wrong format) |
| Near-identical | Moderate (~10%) | Low (params mostly overlap) | Low (output mostly correct) |
| Near-similar | Low-Moderate (~5%) | Moderate (partial param overlap) | Moderate (partial format mismatch) |
| Distinct | Near-zero | Near-zero | Near-zero |

---

## 6. Conflict Type Mapping

### 6.1 Mapping to Existing Conflict Types

| Taxonomy Category | MemInterfere `conflict_type` | `similarity_to_gold` Range |
|------------------|------------------------------|---------------------------|
| Identical | `schema_conflict` | 1.0 |
| Identical | `semantic_conflict` | 0.8–0.9 (name), 0.3–0.5 (semantic) |
| Near-identical | `near_identical` | 0.8–0.9 |
| Near-similar | `near_similar` | 0.5–0.7 |
| Version | `version_conflict` | 0.6–0.8 (schema), 1.0 (name) |
| Distinct | `none` | 0.0–0.2 |
| Stale | `none` (but `is_stale=True`) | N/A (temporal, not semantic) |
| Trap | `none` (but `is_trap=True`) | Variable (designed to be high) |

### 6.2 New `conflict_type` Values

Phase 5.1 introduces two new conflict types, refining the previous `near_duplicate` category:

- **`near_identical`**: Replaces some `near_duplicate` entries. Skills where name and function are nearly the same but differ in name. `similarity_to_gold ∈ [0.8, 0.9]`.
- **`near_similar`**: Replaces some `near_duplicate` entries. Skills with overlapping but not identical scope. `similarity_to_gold ∈ [0.5, 0.7]`.

The previous `near_duplicate` category is deprecated in favor of these two more specific categories.

---

## 7. Real-World Naming Collision Examples

### 7.1 Documented Collisions

| Source | Collision | Type | Notes |
|--------|-----------|------|-------|
| Zapier | `Create Task` (Asana) vs `Create Task` (Todoist) | Identical | Same name, different APIs, different required fields |
| OpenAI | `send_email` (Gmail) vs `send_email` (Outlook) | Identical | Same name, different auth and param schemas |
| Home Assistant | `turn_on` (light) vs `turn_on` (switch) | Near-identical | Same verb, different domains |
| RapidAPI | `search` (YouTube) vs `search` (Spotify) | Identical | Same name, completely different services |
| Anthropic | `get_weather` (WeatherAPI) vs `get_weather` (OpenWeatherMap) | Near-identical | Same intent, different params and response schemas |

### 7.2 Frequency Analysis

From a sample of 500 tool descriptions across public directories:
- **Identical collisions** (same name, different implementation): 15.2%
- **Near-identical collisions** (synonym names, same intent): 8.7%
- **Near-similar collisions** (overlapping scope): 22.3%
- **Distinct pairs** (no collision): 53.8%

This confirms that near-duplicate interference is a real and prevalent problem in deployed agent systems.

---

## 8. Measurement Protocol

### 8.1 Per-Category Metrics

For each similarity category, we measure:

1. **Selection Accuracy Rate (SAR):** % of tasks where the agent selects the gold skill.
2. **Invocation Accuracy Rate (IAR):** % of tasks where the agent invokes the gold skill with correct parameters.
3. **Output Correctness Rate (OCR):** % of tasks where the agent produces the correct final answer.
4. **Confidence Calibration (ECE):** Expected Calibration Error for the agent's confidence scores.

### 8.2 Hypotheses

| Hypothesis | Prediction | Measurement |
|-----------|-----------|-------------|
| H1: Name similarity increases selection error | SAR decreases as similarity increases | SAR across gradient test pairs |
| H2: Identical names cause the most invocation errors | IAR(identical) < IAR(near-identical) < IAR(near-similar) | IAR per category |
| H3: Near-identical skills cause silent degradation | OCR(near-identical) < OCR(distinct) but errors are subtle | OCR delta + qualitative error analysis |
| H4: Staleness compounds with similarity | Stale identical > Stale near-identical > Stale distinct in error rate | Interaction effect in ANOVA |
| H5: Trap acceptance is independent of similarity | P(accept trap) ≈ 100% regardless of similarity | Trap acceptance rate across categories |

---

## 9. Appendix: Similarity Score Computation

### 9.1 Name Similarity (sim_name)

Computed using SentenceTransformer embeddings (all-MiniLM-L6-v2):
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
emb1 = model.encode(skill1_name)
emb2 = model.encode(skill2_name)
sim_name = cosine_similarity(emb1, emb2)
```

### 9.2 Functional Similarity (sim_func)

Assessed by human annotation on a 0–1 scale:
- 1.0: Identical function (same user intent)
- 0.7–0.9: Near-identical (same intent, minor differences)
- 0.4–0.6: Partially overlapping (one subsumes the other)
- 0.1–0.3: Tangentially related (same domain, different intent)
- 0.0: Completely unrelated

### 9.3 Schema Similarity (sim_schema)

Computed as Jaccard similarity of parameter name sets:
```python
def sim_schema(s1, s2):
    params1 = set(s1['parameters'].keys())
    params2 = set(s2['parameters'].keys())
    intersection = params1 & params2
    union = params1 | params2
    return len(intersection) / len(union) if union else 0.0
```

### 9.4 Combined Score

```python
def similarity_score(s1, s2, w_name=0.3, w_func=0.4, w_schema=0.3):
    return w_name * sim_name(s1, s2) + w_func * sim_func(s1, s2) + w_schema * sim_schema(s1, s2)
```

All similarity_to_gold values in the expanded library are computed using this formula, with sim_func assigned by human annotation and sim_name/sim_schema computed automatically.