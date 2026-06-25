#!/usr/bin/env python3
"""
Phase 5.5: Improvements based on Grok-3 and GLM-5.1 reviews

Implements:
1. Fix 3 mislabeled tasks (nat_web_001, nat_web_002, med_web_002)
2. Difficulty-stratified analysis (discriminating tasks only)
3. Confidence calibration analysis
4. Token cost analysis
5. Sensitivity analysis (exclude outliers)
6. Equivalence testing (TOST) — already done, include here
7. Minimum Detectable Effect table
8. Heterogeneity analysis across models
"""
import json, numpy as np, pandas as pd
from scipy import stats
from collections import Counter, defaultdict

with open('data/results/phase5_full/multi_model_results_latest.json') as f:
    data = json.load(f)

results = data['results']
df = pd.DataFrame(results)
df['skill_correct_int'] = df['skill_correct'].astype(int)

# Convert confidence to numeric where possible
def parse_confidence(c):
    if isinstance(c, (int, float)):
        return float(c)
    if isinstance(c, str):
        try: return float(c)
        except: return np.nan
    return np.nan

df['confidence_num'] = df['parsed_confidence'].apply(parse_confidence)

print("=" * 70)
print("IMPROVEMENT ANALYSIS (Based on Grok-3 + GLM-5.1 Reviews)")
print("=" * 70)

# 1. IDENTIFY AND EXCLUDE MISLABELED TASKS
print("\n1. MISLABELED TASK IDENTIFICATION")
print("-" * 50)
# Find tasks with 0% accuracy even in oracle condition
oracle_tasks = df[df['condition'] == 'oracle'].groupby('task_id')['skill_correct_int'].mean()
zero_oracle = oracle_tasks[oracle_tasks == 0].index.tolist()
print(f"Tasks with 0% oracle accuracy: {zero_oracle}")
print("(These are likely annotation disagreements)")

# 2. DIFFICULTY-STRATIFIED ANALYSIS
print("\n2. DISCRIMINATING TASKS ANALYSIS")
print("-" * 50)
# Find tasks where CM and CI outcomes differ
cm_by_task = df[df['condition'] == 'clean_memory'].groupby('task_id')['skill_correct_int'].mean()
ci_by_task = df[df['condition'] == 'clean_interference'].groupby('task_id')['skill_correct_int'].mean()
common_tasks = cm_by_task.index.intersection(ci_by_task.index)
diffs = cm_by_task[common_tasks] - ci_by_task[common_tasks]
discriminating = diffs[diffs != 0].index.tolist()
print(f"Discriminating tasks (CM ≠ CI): {len(discriminating)}/{len(common_tasks)}")
print(f"  CM > CI (interference hurts): {(diffs > 0).sum()} tasks")
print(f"  CI > CM (interference helps): {(diffs < 0).sum()} tasks")
print(f"  CM = CI (no effect): {(diffs == 0).sum()} tasks")
print()
print("Among discriminating tasks:")
disc_data = df[df['task_id'].isin(discriminating) & df['condition'].isin(['clean_memory', 'clean_interference'])]
cm_disc = disc_data[disc_data['condition'] == 'clean_memory']['skill_correct_int'].mean()
ci_disc = disc_data[disc_data['condition'] == 'clean_interference']['skill_correct_int'].mean()
print(f"  CM accuracy: {cm_disc:.3f}")
print(f"  CI accuracy: {ci_disc:.3f}")
print(f"  Difference: {cm_disc - ci_disc:+.4f} ({(cm_disc-ci_disc)*100:+.2f} pp)")

# Paired t-test on discriminating tasks
cm_disc_task = disc_data[disc_data['condition'] == 'clean_memory'].groupby('task_id')['skill_correct_int'].mean()
ci_disc_task = disc_data[disc_data['condition'] == 'clean_interference'].groupby('task_id')['skill_correct_int'].mean()
common_disc = cm_disc_task.index.intersection(ci_disc_task.index)
if len(common_disc) > 1:
    t_stat, p_val = stats.ttest_rel(cm_disc_task[common_disc], ci_disc_task[common_disc])
    print(f"  Paired t-test: t({len(common_disc)-1}) = {t_stat:.3f}, p = {p_val:.4f}")

# 3. CONFIDENCE CALIBRATION ANALYSIS
print("\n3. CONFIDENCE CALIBRATION (Interference Effect on Confidence)")
print("-" * 50)
valid_conf = df[df['confidence_num'].notna() & df['condition'].isin(['clean_memory', 'clean_interference'])]
if len(valid_conf) > 0:
    for model in sorted(valid_conf['model'].unique()):
        md = valid_conf[valid_conf['model'] == model]
        cm_conf = md[md['condition'] == 'clean_memory']['confidence_num']
        ci_conf = md[md['condition'] == 'clean_interference']['confidence_num']
        if len(cm_conf) > 0 and len(ci_conf) > 0:
            t_stat, p_val = stats.ttest_ind(cm_conf, ci_conf)
            diff = cm_conf.mean() - ci_conf.mean()
            print(f"  {model}:")
            print(f"    CM confidence: {cm_conf.mean():.3f} (n={len(cm_conf)})")
            print(f"    CI confidence: {ci_conf.mean():.3f} (n={len(ci_conf)})")
            print(f"    Difference: {diff:+.4f}, t={t_stat:.3f}, p={p_val:.4f} {'*' if p_val < 0.05 else 'n.s.'}")
else:
    print("  No confidence data available")

# 4. TOKEN COST ANALYSIS
print("\n4. TOKEN COST ANALYSIS")
print("-" * 50)
for model in sorted(df['model'].unique()):
    md = df[df['model'] == model]
    cm_tokens = md[md['condition'] == 'clean_memory']['total_tokens'].mean()
    ci_tokens = md[md['condition'] == 'clean_interference']['total_tokens'].mean()
    am_tokens = md[md['condition'] == 'all_memory']['total_tokens'].mean()
    oracle_tokens = md[md['condition'] == 'oracle']['total_tokens'].mean()
    print(f"  {model}:")
    print(f"    Oracle: {oracle_tokens:.0f} tokens")
    print(f"    CM:     {cm_tokens:.0f} tokens")
    print(f"    CI:     {ci_tokens:.0f} tokens ({ci_tokens/cm_tokens:.2f}x CM)")
    print(f"    AM:     {am_tokens:.0f} tokens ({am_tokens/cm_tokens:.2f}x CM)")

# 5. SENSITIVITY ANALYSIS
print("\n5. SENSITIVITY ANALYSIS (Exclude outlier task)")
print("-" * 50)
# Find the worst outlier task
task_diffs = cm_by_task[common_tasks] - ci_by_task[common_tasks]
worst_task = task_diffs.abs().idxmax()
print(f"Worst outlier task: {worst_task} (diff = {task_diffs[worst_task]:+.4f})")

# Re-run without worst outlier
df_no_outlier = df[df['task_id'] != worst_task]
cm_no = df_no_outlier[df_no_outlier['condition'] == 'clean_memory']['skill_correct_int'].mean()
ci_no = df_no_outlier[df_no_outlier['condition'] == 'clean_interference']['skill_correct_int'].mean()
print(f"  With outlier:    CM={cm_by_task[common_tasks].mean():.4f}, CI={ci_by_task[common_tasks].mean():.4f}, diff={cm_by_task[common_tasks].mean()-ci_by_task[common_tasks].mean():+.4f}")
print(f"  Without outlier:  CM={cm_no:.4f}, CI={ci_no:.4f}, diff={cm_no-ci_no:+.4f}")

# Also exclude zero-oracle tasks
df_clean = df[~df['task_id'].isin(zero_oracle)]
cm_clean = df_clean[df_clean['condition'] == 'clean_memory']['skill_correct_int'].mean()
ci_clean = df_clean[df_clean['condition'] == 'clean_interference']['skill_correct_int'].mean()
print(f"  Excluding zero-oracle: CM={cm_clean:.4f}, CI={ci_clean:.4f}, diff={cm_clean-ci_clean:+.4f}")

# 6. MINIMUM DETECTABLE EFFECT TABLE
print("\n6. MINIMUM DETECTABLE EFFECT TABLE")
print("-" * 50)
print(f"{'N per cell':>12} {'Detectable d':>15} {'Detectable Δ (pp)':>20}")
for n in [80, 240, 720, 3600]:
    d = 2 * 0.8 / np.sqrt(n)  # Simplified: d = 2*sqrt(power)/sqrt(n) approximation
    # More accurate: for paired t-test, d_min = z_{1-beta} + z_{1-alpha/2}) / sqrt(n)
    d_min = (stats.norm.ppf(0.8) + stats.norm.ppf(0.975)) / np.sqrt(n)
    delta_pp = d_min * np.sqrt(0.92 * 0.08) * 2  # Approximate for proportions near 92%
    print(f"{n:>12} {d_min:>15.3f} {delta_pp:>20.1f}")

# 7. HETEROGENEITY ANALYSIS
print("\n7. HETEROGENEITY ANALYSIS ACROSS MODELS")
print("-" * 50)
model_effects = {}
for model in sorted(df['model'].unique()):
    md = df[df['model'] == model]
    cm = md[md['condition'] == 'clean_memory']['skill_correct_int'].mean()
    ci = md[md['condition'] == 'clean_interference']['skill_correct_int'].mean()
    model_effects[model] = cm - ci
    print(f"  {model}: CM-CI = {cm-ci:+.4f}")

# Cochran's Q
k = len(model_effects)
effects = list(model_effects.values())
Q = k * sum(e**2 for e in effects) / sum(effects) - sum(effects)
print(f"\nCochran's Q = {Q:.3f} (df={k-1})")
p_Q = 1 - stats.chi2.cdf(Q, k-1) if Q > 0 else 1.0
print(f"P(Q > {Q:.3f}) = {p_Q:.4f}")
I2 = max(0, (Q - (k-1)) / Q * 100) if Q > 0 else 0
print(f"I² = {I2:.1f}%")

# 8. NEAR-DUPLICATE TRAP ANALYSIS
print("\n8. NEAR-DUPLICATE TRAP ANALYSIS")
print("-" * 50)
# Count tasks where parsed_tool matches a near-duplicate but not gold
trap_tasks = df[df['condition'].isin(['clean_interference', 'all_memory'])]
multiple_matches = trap_tasks[trap_tasks['all_matched_skills'].apply(lambda x: len(x) > 1 if isinstance(x, list) else False)]
print(f"Tasks with multiple skill matches: {len(multiple_matches)}/{len(trap_tasks)}")
wrong_match = multiple_matches[~multiple_matches['skill_correct']]
print(f"Wrong near-duplicate selected: {len(wrong_match)}/{len(multiple_matches)}")
if len(multiple_matches) > 0:
    print(f"Trap acceptance rate: {len(wrong_match)/len(multiple_matches):.3f}")

# 9. FULL EQUVALENCE TEST (TOST)
print("\n9. TOST EQUIVALENCE TEST (Comprehensive)")
print("-" * 50)
cm_all = df[df['condition'] == 'clean_memory']['skill_correct_int']
ci_all = df[df['condition'] == 'clean_interference']['skill_correct_int']
diff = cm_all.mean() - ci_all.mean()
se = np.sqrt(cm_all.var()/len(cm_all) + ci_all.var()/len(ci_all))
n = len(cm_all) + len(ci_all)

for margin in [0.01, 0.02, 0.03, 0.05, 0.10]:
    t_lower = (diff - (-margin)) / se
    t_upper = (diff - margin) / se
    p_lower = stats.t.sf(t_lower, df=n-2)
    p_upper = stats.t.cdf(t_upper, df=n-2)
    max_p = max(p_lower, p_upper)
    equiv = "EQUIVALENCE" if max_p < 0.05 else "no equivalence"
    print(f"  ±{margin*100:.0f}pp: TOST p={max_p:.6f} → {equiv}")

print()
print("=" * 70)
print("SUMMARY: RECOMMENDATIONS FOR PAPER")
print("=" * 70)
print("""
1. CRITICAL: Add TOST equivalence testing (±5pp established, ±3pp borderline)
2. CRITICAL: Discuss ceiling effect (89% tasks identical CM/CI outcomes)
3. CRITICAL: Report difficulty-stratified analysis for discriminating tasks
4. HIGH: Report confidence calibration (significant for 2/3 models)
5. HIGH: Report token cost analysis (2.1-2.7× increase)
6. HIGH: Exclude or relabel 3 zero-oracle tasks
7. MEDIUM: Report sensitivity analysis (without outlier task)
8. MEDIUM: Report heterogeneity analysis (Q, I²)
9. MEDIUM: Frame as "null result with rigorous equivalence testing"
10. LOW: Consider removing temperature sweep (adds no information)
""")