#!/usr/bin/env python3
"""
Phase 5.5: Comprehensive Analysis for Publication
Includes per-task analysis, near-duplicate trap analysis, and retrieval vs. selection breakdown.
"""
import json, numpy as np, pandas as pd
from scipy import stats
from collections import Counter, defaultdict

with open('data/results/phase5_full/multi_model_results_latest.json') as f:
    data = json.load(f)

results = data['results']
df = pd.DataFrame(results)
df['skill_correct_int'] = df['skill_correct'].astype(int)

print("=" * 70)
print("COMPREHENSIVE PUBLICATION-READY ANALYSIS")
print("=" * 70)

# 1. Per-task difficulty analysis
print("\n1. PER-TASK DIFFICULTY (ordered by accuracy)")
print("-" * 50)
task_acc = df.groupby('task_id')['skill_correct_int'].mean().sort_values()
print(f"Hardest task: {task_acc.index[0]} = {task_acc.values[0]:.3f}")
print(f"Easiest task: {task_acc.index[-1]} = {task_acc.values[-1]:.3f}")
print(f"Median task accuracy: {task_acc.median():.3f}")
print(f"Tasks at 100% accuracy: {(task_acc == 1.0).sum()}/{len(task_acc)}")
print(f"Tasks below 50% accuracy: {(task_acc < 0.5).sum()}/{len(task_acc)}")

# 2. Near-duplicate trap analysis
print("\n2. NEAR-DUPLICATE TRAP ANALYSIS")
print("-" * 50)
# Check if any tasks involve near-duplicate skills
trap_tasks = df[df['condition'].isin(['clean_interference', 'all_memory'])]
# Find tasks where parsed_tool matches a near-duplicate but not the gold skill
trap_accept = 0
trap_total = 0
for _, row in trap_tasks.iterrows():
    if row['all_matched_skills'] and len(row['all_matched_skills']) > 1:
        # Multiple skills matched - could be a trap
        trap_total += 1
        if not row['skill_correct']:
            trap_accept += 1

print(f"Tasks with multiple skill matches: {trap_total}")
print(f"Tasks where wrong near-duplicate was selected: {trap_accept}")
if trap_total > 0:
    print(f"Trap acceptance rate: {trap_accept/trap_total:.3f}")

# 3. Condition comparison (all pairs)
print("\n3. ALL CONDITION PAIRS (paired t-tests)")
print("-" * 50)
conditions = ['oracle', 'no_memory', 'clean_memory', 'clean_interference', 'all_memory']
for i, c1 in enumerate(conditions):
    for c2 in conditions[i+1:]:
        d1 = df[df['condition'] == c1].groupby('task_id')['skill_correct_int'].mean()
        d2 = df[df['condition'] == c2].groupby('task_id')['skill_correct_int'].mean()
        # Align by task_id
        common = d1.index.intersection(d2.index)
        if len(common) > 1:
            t_stat, p_val = stats.ttest_rel(d1[common], d2[common])
            diff = d1[common].mean() - d2[common].mean()
            print(f"  {c1} vs {c2}: diff={diff:+.4f}, t={t_stat:.3f}, p={p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'}")

# 4. Oracle ceiling analysis
print("\n4. ORACLE CEILING ANALYSIS")
print("-" * 50)
oracle_acc = df[df['condition'] == 'oracle'].groupby('task_id')['skill_correct_int'].mean()
cm_acc = df[df['condition'] == 'clean_memory'].groupby('task_id')['skill_correct_int'].mean()
ci_acc = df[df['condition'] == 'clean_interference'].groupby('task_id')['skill_correct_int'].mean()
common = oracle_acc.index.intersection(cm_acc.index).intersection(ci_acc.index)
print(f"Oracle accuracy: {oracle_acc[common].mean():.3f}")
print(f"Clean memory: {cm_acc[common].mean():.3f} (gap from oracle: {oracle_acc[common].mean() - cm_acc[common].mean():.4f})")
print(f"Clean interference: {ci_acc[common].mean():.3f} (gap from oracle: {oracle_acc[common].mean() - ci_acc[common].mean():.4f})")
print(f"Interference gap (CM-CI): {cm_acc[common].mean() - ci_acc[common].mean():.4f}")
print(f"Interference gap as % of oracle: {(cm_acc[common].mean() - ci_acc[common].mean()) / oracle_acc[common].mean() * 100:.2f}%")

# 5. Model × Condition interaction
print("\n5. MODEL × CONDITION INTERACTION")
print("-" * 50)
for model in sorted(df['model'].unique()):
    md = df[df['model'] == model]
    cm = md[md['condition'] == 'clean_memory']['skill_correct_int'].mean()
    ci = md[md['condition'] == 'clean_interference']['skill_correct_int'].mean()
    am = md[md['condition'] == 'all_memory']['skill_correct_int'].mean()
    print(f"  {model}:")
    print(f"    CM={cm:.3f}, CI={ci:.3f}, AM={am:.3f}")
    print(f"    CM-CI={cm-ci:+.4f}, CM-AM={cm-am:+.4f}")

# 6. Temperature interaction with interference
print("\n6. TEMPERATURE × INTERFERENCE INTERACTION")
print("-" * 50)
df_int = df[df['condition'].isin(['clean_memory', 'clean_interference'])].copy()
df_int['is_interference'] = (df_int['condition'] == 'clean_interference').astype(int)
for temp in sorted(df_int['temperature'].unique()):
    td = df_int[df_int['temperature'] == temp]
    cm = td[td['condition'] == 'clean_memory']['skill_correct_int'].mean()
    ci = td[td['condition'] == 'clean_interference']['skill_correct_int'].mean()
    print(f"  T={temp}: CM={cm:.3f}, CI={ci:.3f}, diff={cm-ci:+.4f}")

# 7. Effect size summary for paper
print("\n7. EFFECT SIZE SUMMARY (for paper)")
print("-" * 50)
cm_all = df[df['condition'] == 'clean_memory']['skill_correct_int']
ci_all = df[df['condition'] == 'clean_interference']['skill_correct_int']
diff = cm_all.mean() - ci_all.mean()
pooled_std = np.sqrt((cm_all.var() + ci_all.var()) / 2)
cohens_d = diff / pooled_std if pooled_std > 0 else 0
print(f"Mean CM accuracy: {cm_all.mean():.4f}")
print(f"Mean CI accuracy: {ci_all.mean():.4f}")
print(f"Raw difference: {diff:.4f} ({diff*100:.2f} pp)")
print(f"Pooled SD: {pooled_std:.4f}")
print(f"Cohen's d: {cohens_d:.4f}")
print(f"Interpretation: {'negligible' if abs(cohens_d) < 0.2 else 'small' if abs(cohens_d) < 0.5 else 'medium'}")

# 8. Binary: did interference HURT at all?
print("\n8. PER-TASK INTERFERENCE DIRECTION")
print("-" * 50)
cm_task = df[df['condition'] == 'clean_memory'].groupby('task_id')['skill_correct_int'].mean()
ci_task = df[df['condition'] == 'clean_interference'].groupby('task_id')['skill_correct_int'].mean()
common = cm_task.index.intersection(ci_task.index)
diffs = cm_task[common] - ci_task[common]
hurt = (diffs > 0).sum()
helped = (diffs < 0).sum()
no_change = (diffs == 0).sum()
print(f"Tasks where interference hurt: {hurt}/{len(common)} ({hurt/len(common)*100:.1f}%)")
print(f"Tasks where interference helped: {helped}/{len(common)} ({helped/len(common)*100:.1f}%)")
print(f"Tasks with no change: {no_change}/{len(common)} ({no_change/len(common)*100:.1f}%)")
print(f"Max interference harm: {diffs.max():.4f}")
print(f"Max interference benefit: {diffs.min():.4f}")