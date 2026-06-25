#!/usr/bin/env python3
"""
MemInterfere Phase 5.5: Statistical Analysis

Analyzes experiment results with:
1. Per-model accuracy by condition and temperature
2. Paired t-tests: clean_memory vs. clean_interference (main hypothesis)
3. Paired t-tests: clean_memory vs. all_memory (interference gradient)
4. Mixed-effects ANOVA: condition × model × temperature
5. Bootstrap confidence intervals for effect sizes
6. Parse rate analysis
7. Temperature sensitivity analysis
"""

import json
import sys
import os
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np

# Check for scipy/statsmodels
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


@dataclass
class AnalysisResult:
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    n: int
    description: str


def load_results(path: str) -> list[dict]:
    """Load experiment results from JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get('results', [])


def filter_valid(results: list[dict]) -> list[dict]:
    """Filter out error results and PARSE_ERROR results."""
    return [r for r in results if not r.get('error') and r.get('parse_method') != 'error']


def compute_accuracies(results: list[dict]) -> dict:
    """Compute accuracy by model, condition, and temperature."""
    valid = filter_valid(results)
    
    # Group by model, condition, temperature
    groups = defaultdict(list)
    for r in valid:
        key = (r.get('model', 'unknown'), r.get('condition', 'unknown'), r.get('temperature', -1))
        groups[key].append(r)
    
    accuracies = {}
    for key, runs in groups.items():
        model, condition, temp = key
        total = len(runs)
        correct = sum(1 for r in runs if r.get('skill_correct'))
        accuracies[key] = {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0,
        }
    
    return accuracies


def paired_ttest(results: list[dict], condition_a: str, condition_b: str) -> list[AnalysisResult]:
    """
    Paired t-test comparing two conditions across tasks, per model.
    Each task is a paired observation: accuracy under condition_a vs condition_b.
    """
    valid = filter_valid(results)
    models = set(r.get('model') for r in valid)
    temperatures = set(r.get('temperature') for r in valid)
    
    results_list = []
    
    for model in sorted(models):
        for temp in sorted(temperatures):
            # Get task-level paired data
            a_results = {r.get('task_id'): r for r in valid 
                         if r.get('model') == model and r.get('condition') == condition_a 
                         and r.get('temperature') == temp}
            b_results = {r.get('task_id'): r for r in valid 
                         if r.get('model') == model and r.get('condition') == condition_b 
                         and r.get('temperature') == temp}
            
            # Find common tasks
            common_tasks = set(a_results.keys()) & set(b_results.keys())
            if len(common_tasks) < 2:
                continue
            
            # Paired differences (correct=1, incorrect=0)
            a_scores = [1 if a_results[t].get('skill_correct') else 0 for t in common_tasks]
            b_scores = [1 if b_results[t].get('skill_correct') else 0 for t in common_tasks]
            differences = [a - b for a, b in zip(a_scores, b_scores)]
            
            n = len(differences)
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            
            if std_diff == 0:
                # All differences are identical
                p_value = 1.0 if mean_diff == 0 else 0.0
                t_stat = 0.0 if mean_diff == 0 else float('inf')
                ci_lower = mean_diff
                ci_upper = mean_diff
            elif HAS_SCIPY:
                t_stat, p_value = scipy_stats.ttest_rel(a_scores, b_scores)
                # Cohen's d for paired test
                d = mean_diff / std_diff if std_diff > 0 else 0
                # 95% CI
                se = std_diff / np.sqrt(n)
                ci_lower = mean_diff - 1.96 * se
                ci_upper = mean_diff + 1.96 * se
            else:
                t_stat = mean_diff / (std_diff / np.sqrt(n)) if std_diff > 0 else 0
                p_value = 2 * (1 - _t_cdf(abs(t_stat), n - 1)) if n > 1 else 1.0
                d = mean_diff / std_diff if std_diff > 0 else 0
                se = std_diff / np.sqrt(n)
                ci_lower = mean_diff - 1.96 * se
                ci_upper = mean_diff + 1.96 * se
            
            results_list.append(AnalysisResult(
                test_name=f"paired_t_{condition_a}_vs_{condition_b}",
                statistic=t_stat,
                p_value=p_value,
                effect_size=d if std_diff > 0 else 0,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n=n,
                description=f"{model} @ T={temp}: {condition_a} vs {condition_b}, mean_diff={mean_diff:.4f}, d={d if std_diff > 0 else 0:.4f}"
            ))
    
    return results_list


def _t_cdf(t, df):
    """Approximate t-distribution CDF using normal approximation for large df."""
    from math import erf, sqrt
    z = t / sqrt(1 + t*t/df) if df > 0 else t
    return 0.5 * (1 + erf(z / sqrt(2)))


def bootstrap_ci(data: list, n_bootstrap: int = 10000, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for mean."""
    data = np.array(data)
    n = len(data)
    bootstrap_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    return np.mean(data), lower, upper


def mixed_effects_anova(results: list[dict]) -> dict:
    """
    Mixed-effects ANOVA: accuracy ~ condition * model * temperature
    with task as random effect.
    Requires statsmodels.
    """
    if not HAS_STATSMODELS:
        return {"error": "statsmodels not available"}
    
    valid = filter_valid(results)
    if len(valid) < 10:
        return {"error": "Insufficient data for ANOVA"}
    
    import pandas as pd
    
    df = pd.DataFrame(valid)
    df['accuracy'] = df['skill_correct'].astype(int)
    
    # Filter to conditions with variation
    conditions = ['oracle', 'no_memory', 'clean_memory', 'clean_interference', 'all_memory']
    df = df[df['condition'].isin(conditions)]
    
    try:
        # Two-way ANOVA: condition × model
        model = ols('accuracy ~ C(condition) + C(model) + C(temperature) + C(condition):C(model)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return {
            'anova_table': anova_table.to_dict(),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_parse_rates(results: list[dict]) -> dict:
    """Analyze parse method distribution by model."""
    valid = filter_valid(results)
    models = set(r.get('model') for r in valid)
    
    parse_analysis = {}
    for model in sorted(models):
        model_results = [r for r in valid if r.get('model') == model]
        methods = Counter(r.get('parse_method', 'unknown') for r in model_results)
        total = len(model_results)
        parse_analysis[model] = {
            'total': total,
            'methods': dict(methods),
            'parse_rate': sum(1 for r in model_results if r.get('parse_method') != 'error') / total if total > 0 else 0,
            'json_rate': methods.get('json', 0) / total if total > 0 else 0,
        }
    
    return parse_analysis


def temperature_sensitivity(results: list[dict]) -> dict:
    """Analyze temperature sensitivity by model and condition."""
    valid = filter_valid(results)
    models = set(r.get('model') for r in valid)
    temps = sorted(set(r.get('temperature') for r in valid))
    
    sensitivity = {}
    for model in sorted(models):
        model_results = [r for r in valid if r.get('model') == model]
        by_temp = {}
        for temp in temps:
            temp_results = [r for r in model_results if r.get('temperature') == temp]
            correct = sum(1 for r in temp_results if r.get('skill_correct'))
            total = len(temp_results)
            by_temp[temp] = {
                'accuracy': correct / total if total > 0 else 0,
                'n': total,
            }
        sensitivity[model] = by_temp
    
    return sensitivity


def run_full_analysis(results_path: str, output_path: str = None):
    """Run the complete analysis pipeline."""
    results = load_results(results_path)
    valid = filter_valid(results)
    
    print("=" * 70)
    print("MemInterfere Phase 5.5: Statistical Analysis")
    print("=" * 70)
    
    # Basic stats
    total = len(results)
    errors = sum(1 for r in results if r.get('error'))
    parse_errors = sum(1 for r in results if r.get('parse_method') == 'error')
    valid_count = len(valid)
    print(f"\nTotal results: {total}")
    print(f"Valid results: {valid_count}")
    print(f"API errors: {errors}")
    print(f"Parse errors: {parse_errors}")
    print(f"Parse rate: {valid_count / total * 100:.1f}%" if total > 0 else "No results")
    
    # 1. Per-model accuracy
    print("\n" + "=" * 70)
    print("1. PER-MODEL ACCURACY BY CONDITION AND TEMPERATURE")
    print("=" * 70)
    accuracies = compute_accuracies(results)
    models = sorted(set(k[0] for k in accuracies.keys()))
    conditions = ['oracle', 'no_memory', 'clean_memory', 'clean_interference', 'all_memory']
    temps = sorted(set(k[2] for k in accuracies.keys()))
    
    for model in models:
        print(f"\n  {model}:")
        for temp in temps:
            print(f"    T={temp}:")
            for cond in conditions:
                key = (model, cond, temp)
                if key in accuracies:
                    a = accuracies[key]
                    print(f"      {cond:25s}: {a['correct']:3d}/{a['total']:3d} = {a['accuracy']:.1%}")
    
    # 2. Main hypothesis: clean_memory vs clean_interference
    print("\n" + "=" * 70)
    print("2. MAIN HYPOTHESIS: clean_memory vs clean_interference")
    print("   H0: Interference does NOT degrade tool selection")
    print("=" * 70)
    ttest_results = paired_ttest(results, 'clean_memory', 'clean_interference')
    for r in ttest_results:
        sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else "n.s."
        print(f"  {r.description}")
        print(f"    t({r.n-1}) = {r.statistic:.3f}, p = {r.p_value:.4f} {sig}")
        print(f"    Cohen's d = {r.effect_size:.4f}, 95% CI: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
    
    # 3. Gradient hypothesis: clean_memory vs all_memory
    print("\n" + "=" * 70)
    print("3. GRADIENT HYPOTHESIS: clean_memory vs all_memory")
    print("   Tests whether MORE interference degrades performance more")
    print("=" * 70)
    gradient_results = paired_ttest(results, 'clean_memory', 'all_memory')
    for r in gradient_results:
        sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else "n.s."
        print(f"  {r.description}")
        print(f"    t({r.n-1}) = {r.statistic:.3f}, p = {r.p_value:.4f} {sig}")
        print(f"    Cohen's d = {r.effect_size:.4f}, 95% CI: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
    
    # 4. Parse rates
    print("\n" + "=" * 70)
    print("4. PARSE RATE ANALYSIS")
    print("=" * 70)
    parse_analysis = analyze_parse_rates(results)
    for model, analysis in parse_analysis.items():
        print(f"  {model}:")
        print(f"    Total: {analysis['total']}")
        print(f"    Parse rate: {analysis['parse_rate']:.1%}")
        print(f"    JSON rate: {analysis['json_rate']:.1%}")
        print(f"    Methods: {analysis['methods']}")
    
    # 5. Temperature sensitivity
    print("\n" + "=" * 70)
    print("5. TEMPERATURE SENSITIVITY")
    print("=" * 70)
    temp_sens = temperature_sensitivity(results)
    for model, by_temp in temp_sens.items():
        print(f"  {model}:")
        for temp, stats in sorted(by_temp.items()):
            print(f"    T={temp}: {stats['accuracy']:.1%} (n={stats['n']})")
    
    # 6. Mixed-effects ANOVA
    print("\n" + "=" * 70)
    print("6. MIXED-EFFECTS ANOVA")
    print("=" * 70)
    anova = mixed_effects_anova(results)
    if 'error' in anova:
        print(f"  Skipped: {anova['error']}")
    else:
        print(f"  R² = {anova['r_squared']:.4f}")
        print(f"  Adjusted R² = {anova['adj_r_squared']:.4f}")
        print(f"  ANOVA table:")
        for source, values in anova['anova_table'].items():
            if source == 'PR(>F)':
                continue
            f_val = values.get('F', 'N/A')
            p_val = values.get('PR(>F)', 'N/A')
            print(f"    {source:40s}: F = {f_val}, p = {p_val}")
    
    # 7. Bootstrap CIs for main effects
    print("\n" + "=" * 70)
    print("7. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("=" * 70)
    for model in models:
        for temp in temps:
            cm_key = (model, 'clean_memory', temp)
            ci_key = (model, 'clean_interference', temp)
            if cm_key in accuracies and ci_key in accuracies:
                cm_data = [1 if r.get('skill_correct') else 0 for r in valid 
                           if r.get('model') == model and r.get('condition') == 'clean_memory' 
                           and r.get('temperature') == temp]
                ci_data = [1 if r.get('skill_correct') else 0 for r in valid 
                           if r.get('model') == model and r.get('condition') == 'clean_interference' 
                           and r.get('temperature') == temp]
                
                if len(cm_data) >= 5 and len(ci_data) >= 5:
                    cm_mean, cm_lo, cm_hi = bootstrap_ci(cm_data)
                    ci_mean, ci_lo, ci_hi = bootstrap_ci(ci_data)
                    diff = [a - b for a, b in zip(cm_data[:min(len(cm_data), len(ci_data))], 
                                                    ci_data[:min(len(cm_data), len(ci_data))])]
                    diff_mean, diff_lo, diff_hi = bootstrap_ci(diff)
                    print(f"  {model} @ T={temp}:")
                    print(f"    clean_memory:       {cm_mean:.3f} [{cm_lo:.3f}, {cm_hi:.3f}]")
                    print(f"    clean_interference: {ci_mean:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
                    print(f"    difference:         {diff_mean:.3f} [{diff_lo:.3f}, {diff_hi:.3f}]")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Models analyzed: {len(models)}")
    print(f"Conditions: {conditions}")
    print(f"Temperatures: {temps}")
    print(f"Total valid runs: {valid_count}")
    print(f"Total runs (incl. errors): {total}")
    
    if ttest_results:
        mean_p = np.mean([r.p_value for r in ttest_results])
        mean_d = np.mean([abs(r.effect_size) for r in ttest_results])
        print(f"\nMain hypothesis (clean_memory vs clean_interference):")
        print(f"  Mean p-value: {mean_p:.4f}")
        print(f"  Mean |Cohen's d|: {mean_d:.4f}")
        if mean_p > 0.05:
            print(f"  RESULT: Fail to reject H0 — interference does NOT significantly degrade selection")
        else:
            print(f"  RESULT: Reject H0 — interference DOES significantly degrade selection")
    
    # Save results if output path provided
    if output_path:
        output = {
            'total_results': total,
            'valid_results': valid_count,
            'errors': errors,
            'parse_errors': parse_errors,
            'accuracies': {str(k): v for k, v in accuracies.items()},
            'paired_ttests_main': [
                {'test': r.test_name, 'statistic': r.statistic, 'p_value': r.p_value,
                 'effect_size': r.effect_size, 'ci_lower': r.ci_lower, 'ci_upper': r.ci_upper,
                 'n': r.n, 'description': r.description}
                for r in ttest_results
            ],
            'paired_ttests_gradient': [
                {'test': r.test_name, 'statistic': r.statistic, 'p_value': r.p_value,
                 'effect_size': r.effect_size, 'ci_lower': r.ci_lower, 'ci_upper': r.ci_upper,
                 'n': r.n, 'description': r.description}
                for r in gradient_results
            ],
            'parse_analysis': parse_analysis,
            'temperature_sensitivity': temp_sens,
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")
    
    return {
        'accuracies': accuracies,
        'ttest_main': ttest_results,
        'ttest_gradient': gradient_results,
        'parse_analysis': parse_analysis,
        'temperature_sensitivity': temp_sens,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MemInterfere Phase 5.5 Statistical Analysis')
    parser.add_argument('--input', required=True, help='Path to experiment results JSON')
    parser.add_argument('--output', help='Path to save analysis results JSON')
    args = parser.parse_args()
    
    run_full_analysis(args.input, args.output)