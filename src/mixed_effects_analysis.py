#!/usr/bin/env python3
"""Phase 5.5: Mixed-effects analysis and equivalence testing"""
import json, numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

with open('data/results/phase5_full/multi_model_results_latest.json') as f:
    data = json.load(f)

df = pd.DataFrame(data['results'])
df['skill_correct_int'] = df['skill_correct'].astype(int)
df['is_interference'] = (df['condition'] == 'clean_interference').astype(int)

df_int = df[df['condition'].isin(['clean_memory', 'clean_interference'])].copy()
df_int['skill_correct_int'] = df_int['skill_correct'].astype(int)
df_int['is_interference'] = (df_int['condition'] == 'clean_interference').astype(int)

print("=" * 70)
print("MIXED-EFFECTS MODEL 1: skill_correct ~ interference + temp + (1|model)")
print("=" * 70)

model1 = mixedlm('skill_correct_int ~ is_interference + temperature', 
                data=df_int, 
                groups=df_int['model'])
result1 = model1.fit()
print(result1.summary().tables[1])
print(f"\nInterference coef: {result1.params['is_interference']:.6f}")
print(f"Interference p:   {result1.pvalues['is_interference']:.6f}")

print()
print("=" * 70)
print("MIXED-EFFECTS MODEL 2: skill_correct ~ interference + C(model) + (1|task)")
print("=" * 70)

model2 = mixedlm('skill_correct_int ~ is_interference + temperature + C(model)', 
                data=df_int, 
                groups=df_int['task_id'])
result2 = model2.fit()
print(result2.summary().tables[1])
print(f"\nInterference coef: {result2.params['is_interference']:.6f}")
print(f"Interference p:   {result2.pvalues['is_interference']:.6f}")

# TOST equivalence test
cm = df_int[df_int['condition'] == 'clean_memory']['skill_correct_int']
ci = df_int[df_int['condition'] == 'clean_interference']['skill_correct_int']
diff = cm.mean() - ci.mean()
se = np.sqrt(cm.var()/len(cm) + ci.var()/len(ci))
n = len(cm) + len(ci)

print()
print("=" * 70)
print("EQUIVALENCE TEST (TOST)")
print("=" * 70)
print(f"Observed diff: {diff:.4f} ({diff*100:.2f} pp)")
print(f"SE: {se:.4f}")
print(f"95% CI: [{diff-1.96*se:.4f}, {diff+1.96*se:.4f}]")

for margin in [0.05, 0.03, 0.02]:
    t_lower = (diff - (-margin)) / se
    t_upper = (diff - margin) / se
    p_lower = stats.t.sf(t_lower, df=n-2)
    p_upper = stats.t.cdf(t_upper, df=n-2)
    max_p = max(p_lower, p_upper)
    equiv = "EQUIVALENCE" if max_p < 0.05 else "no equivalence"
    print(f"  ±{margin*100:.0f}pp: TOST p={max_p:.6f} → {equiv}")

# Per-model breakdown
print()
print("Per-model interference effects:")
for m in sorted(df_int['model'].unique()):
    md = df_int[df_int['model'] == m]
    cm_m = md[md['condition'] == 'clean_memory']['skill_correct_int'].mean()
    ci_m = md[md['condition'] == 'clean_interference']['skill_correct_int'].mean()
    print(f"  {m}: CM={cm_m:.3f}, CI={ci_m:.3f}, diff={cm_m-ci_m:.4f}")

# Effect by temperature
print()
print("Interference effect by temperature:")
for t in sorted(df_int['temperature'].unique()):
    td = df_int[df_int['temperature'] == t]
    cm_t = td[td['condition'] == 'clean_memory']['skill_correct_int'].mean()
    ci_t = td[td['condition'] == 'clean_interference']['skill_correct_int'].mean()
    print(f"  T={t}: CM={cm_t:.3f}, CI={ci_t:.3f}, diff={cm_t-ci_t:.4f}")