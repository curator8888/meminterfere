#!/usr/bin/env python3
"""
Phase 6.1: GLMM analysis (mixed-effects logistic regression) + figure generation.

Addresses recommendation A10: Use GLMM for binary outcome instead of LMM.
"""
import json, os, sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'results', 'phase5_full_4model', 'multi_model_results_latest.json')

def run_glmm():
    """Run generalized linear mixed model (logistic regression) for binary outcome."""
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import pandas as pd
    except ImportError:
        print("ERROR: statsmodels required. Install with: pip install statsmodels")
        return None
    
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    
    results = data['results']
    
    # Build dataframe
    rows = []
    ambiguous = {'nat_web_001', 'nat_web_002', 'med_web_002'}
    for r in results:
        if r['task_id'] in ambiguous:
            continue
        rows.append({
            'correct': int(r.get('skill_correct', False)),
            'condition': r['condition'],
            'model': r['model'],
            'temperature': r['temperature'],
            'task_id': r['task_id'],
            'is_interference': 1 if r['condition'] == 'clean_interference' else 0,
            'is_clean_memory': 1 if r['condition'] == 'clean_memory' else 0,
        })
    
    df = pd.DataFrame(rows)
    print(f"GLMM dataset: {len(df)} observations, {df['correct'].mean():.3f} accuracy")
    print(f"  Models: {df['model'].unique()}")
    print(f"  Tasks: {df['task_id'].nunique()}")
    print()
    
    # Model 1: GLMM with task random effects (logistic regression)
    # correct ~ is_interference + temperature + (1 | task_id)
    print("=" * 70)
    print("GLMM: correct ~ is_interference + temperature + (1 | task_id)")
    print("=" * 70)
    
    # Binomial mixed model (approximation using GEE or random effects)
    # statsmodels doesn't have GLMM directly, use BinomialBayesMixedGLM
    try:
        # Use BinomialBayesMixedGLM for proper logistic mixed model
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        
        # Prepare data
        endog = df['correct'].values
        exog = sm.add_constant(df[['is_interference', 'temperature']].values)
        exog_vc = pd.get_dummies(df['task_id']).values.astype(float)
        
        model = BinomialBayesMixedGLM(
            endog=endog,
            exog=exog,
            exog_vc=exog_vc,
            vcp_p=1.0,
            vcp_s=1.0,
        )
        result = model.fit()
        print(result.summary())
        print()
        
        # Extract key results
        print("GLMM Key Results:")
        for i, name in enumerate(['Intercept', 'is_interference', 'temperature']):
            coef = result.fe_mean[i]
            se = result.fe_sd[i]
            z = coef / se if se > 0 else float('inf')
            p = 2 * (1 - sm.stats.norm.cdf(abs(z)))
            print(f"  {name}: coef={coef:.4f}, se={se:.4f}, z={z:.3f}, p={p:.4f}")
        
        # Odds ratio for interference
        or_interference = np.exp(result.fe_mean[1])
        or_ci_low = np.exp(result.fe_mean[1] - 1.96 * result.fe_sd[1])
        or_ci_high = np.exp(result.fe_mean[1] + 1.96 * result.fe_sd[1])
        print(f"\n  Odds Ratio for interference: {or_interference:.3f} [{or_ci_low:.3f}, {or_ci_high:.3f}]")
        
    except Exception as e:
        print(f"BinomialBayesMixedGLM failed: {e}")
        print("\nFalling back to standard logistic regression with clustered SE...")
        
        # Fallback: pooled logistic regression with cluster-robust SE
        X = sm.add_constant(df[['is_interference', 'temperature']])
        y = df['correct']
        model = sm.Logit(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['task_id']})
        print(model.summary())
        print()
        print("Pooled Logistic Regression Key Results (cluster-robust SE):")
        or_interference = np.exp(model.params['is_interference'])
        print(f"  Odds Ratio for interference: {or_interference:.3f}")
        print(f"  Pseudo R²: {model.prsquared:.4f}")
    
    return df

def generate_figures(df=None):
    """Generate publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("ERROR: matplotlib required. Install with: pip install matplotlib")
        return
    
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    
    results = data['results']
    ambiguous = {'nat_web_001', 'nat_web_002', 'med_web_002'}
    
    # Build dataframe
    if df is None:
        import pandas as pd
        rows = []
        for r in results:
            if r['task_id'] in ambiguous:
                continue
            rows.append({
                'correct': int(r.get('skill_correct', False)),
                'condition': r['condition'],
                'model': r['model'],
                'temperature': r['temperature'],
                'task_id': r['task_id'],
                'confidence': float(r.get('confidence', 0.0)),
                'tokens_in': int(r.get('tokens_in', 0)),
                'tokens_out': int(r.get('tokens_out', 0)),
            })
        df = pd.DataFrame(rows)
    
    # ── Figure 1: Per-condition accuracy bar chart ──────────────────────────
    print("\nGenerating Figure 1: Per-condition accuracy...")
    
    conditions = ['no_memory', 'oracle', 'clean_memory', 'clean_interference', 'all_memory']
    cond_labels = ['No Memory', 'Oracle', 'Clean\nMemory', 'Clean +\nInterference', 'All\nMemory']
    model_names = sorted(df['model'].unique())
    model_labels = {'claude-haiku-4.5': 'Claude Haiku 4.5', 'gpt-4o-mini': 'GPT-4o-mini', 
                    'grok-3-mini': 'Grok-3-mini', 'llama-3.1-8b-instruct': 'Llama-3.1-8B'}
    model_colors = {'claude-haiku-4.5': '#6366f1', 'gpt-4o-mini': '#10b981',
                    'grok-3-mini': '#f59e0b', 'llama-3.1-8b-instruct': '#ef4444'}
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(len(conditions))
    width = 0.18
    
    for i, model in enumerate(model_names):
        model_df = df[df['model'] == model]
        accs = []
        for cond in conditions:
            cond_df = model_df[model_df['condition'] == cond]
            accs.append(cond_df['correct'].mean() * 100)
        ax.bar(x + i * width, accs, width, label=model_labels.get(model, model),
               color=model_colors.get(model, '#666666'), alpha=0.85)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_title('Tool Selection Accuracy by Condition and Model', fontsize=13, fontweight='bold')
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.3, label='_nolegend_')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig1_condition_accuracy.pdf'), dpi=300)
    fig.savefig(os.path.join(fig_dir, 'fig1_condition_accuracy.png'), dpi=300)
    print(f"  Saved to {fig_dir}/fig1_condition_accuracy.{{pdf,png}}")
    
    # ── Figure 2: Forest plot of model-level effects ────────────────────────
    print("Generating Figure 2: Forest plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    y_pos = []
    labels = []
    effects = []
    ci_low = []
    ci_high = []
    
    for i, model in enumerate(model_names):
        model_df = df[df['model'] == model]
        cm_acc = model_df[model_df['condition'] == 'clean_memory']['correct'].mean()
        ci_acc = model_df[model_df['condition'] == 'clean_interference']['correct'].mean()
        diff = (cm_acc - ci_acc) * 100  # in pp
        
        # Bootstrap CI
        n_boot = 10000
        tasks = model_df['task_id'].unique()
        boot_diffs = []
        for _ in range(n_boot):
            boot_tasks = np.random.choice(tasks, size=len(tasks), replace=True)
            cm_boot = model_df[(model_df['condition'] == 'clean_memory') & (model_df['task_id'].isin(boot_tasks))]['correct'].mean()
            ci_boot = model_df[(model_df['condition'] == 'clean_interference') & (model_df['task_id'].isin(boot_tasks))]['correct'].mean()
            boot_diffs.append((cm_boot - ci_boot) * 100)
        
        ci_lo = np.percentile(boot_diffs, 2.5)
        ci_hi = np.percentile(boot_diffs, 97.5)
        
        y_pos.append(i)
        labels.append(model_labels.get(model, model))
        effects.append(diff)
        ci_low.append(ci_lo)
        ci_high.append(ci_hi)
    
    # Overall
    cm_all = df[df['condition'] == 'clean_memory']['correct'].mean()
    ci_all = df[df['condition'] == 'clean_interference']['correct'].mean()
    overall_diff = (cm_all - ci_all) * 100
    y_pos.append(len(model_names))
    labels.append('Overall')
    effects.append(overall_diff)
    
    # Bootstrap CI for overall
    n_boot = 10000
    tasks = df['task_id'].unique()
    boot_diffs = []
    for _ in range(n_boot):
        boot_tasks = np.random.choice(tasks, size=len(tasks), replace=True)
        cm_boot = df[(df['condition'] == 'clean_memory') & (df['task_id'].isin(boot_tasks))]['correct'].mean()
        ci_boot = df[(df['condition'] == 'clean_interference') & (df['task_id'].isin(boot_tasks))]['correct'].mean()
        boot_diffs.append((cm_boot - ci_boot) * 100)
    ci_low.append(np.percentile(boot_diffs, 2.5))
    ci_high.append(np.percentile(boot_diffs, 97.5))
    
    # Plot
    colors = [model_colors.get(m, '#333333') for m in model_names] + ['#000000']
    for i in range(len(y_pos)):
        ax.plot([ci_low[i], ci_high[i]], [y_pos[i], y_pos[i]], color=colors[i], linewidth=2, alpha=0.7)
        ax.plot(effects[i], y_pos[i], 'o', color=colors[i], markersize=8)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=3, color='red', linestyle=':', alpha=0.5, label='TOST bound (±3pp)')
    ax.axvline(x=-3, color='red', linestyle=':', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Accuracy Difference: Clean Memory − Clean Interference (pp)', fontsize=11)
    ax.set_title('Forest Plot: Interference Effect by Model', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig2_forest_plot.pdf'), dpi=300)
    fig.savefig(os.path.join(fig_dir, 'fig2_forest_plot.png'), dpi=300)
    print(f"  Saved to {fig_dir}/fig2_forest_plot.{{pdf,png}}")
    
    # ── Figure 3: Confidence calibration ───────────────────────────────────
    print("Generating Figure 3: Confidence calibration...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 3a: Confidence by condition
    ax = axes[0]
    conditions_short = ['clean_memory', 'clean_interference', 'all_memory']
    cond_short_labels = ['Clean Memory', 'Clean + Interference', 'All Memory']
    cond_short_colors = ['#10b981', '#f59e0b', '#ef4444']
    
    for j, (cond, label, color) in enumerate(zip(conditions_short, cond_short_labels, cond_short_colors)):
        cond_df = df[df['condition'] == cond]
        means = []
        for model in model_names:
            model_cond = cond_df[cond_df['model'] == model]
            means.append(model_cond['confidence'].mean())
        ax.bar([j], [np.mean(means)], width=0.6, color=color, alpha=0.7, label=label)
        ax.errorbar([j], [np.mean(means)], yerr=[np.std(means)], fmt='k_', capsize=5)
    
    ax.set_ylabel('Mean Confidence', fontsize=12)
    ax.set_xticks(range(len(cond_short_labels)))
    ax.set_xticklabels(cond_short_labels, fontsize=9)
    ax.set_title('Confidence by Condition', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 3b: Token cost by condition
    ax = axes[1]
    for j, (cond, label, color) in enumerate(zip(conditions_short, cond_short_labels, cond_short_colors)):
        cond_df = df[df['condition'] == cond]
        total_tokens = cond_df.groupby('model')['tokens_in'].mean() + cond_df.groupby('model')['tokens_out'].mean()
        ax.bar([j], [total_tokens.mean()], width=0.6, color=color, alpha=0.7)
        ax.errorbar([j], [total_tokens.mean()], yerr=[total_tokens.std()], fmt='k_', capsize=5)
    
    ax.set_ylabel('Mean Total Tokens', fontsize=12)
    ax.set_xticks(range(len(cond_short_labels)))
    ax.set_xticklabels(cond_short_labels, fontsize=9)
    ax.set_title('Token Cost by Condition', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig3_confidence_tokens.pdf'), dpi=300)
    fig.savefig(os.path.join(fig_dir, 'fig3_confidence_tokens.png'), dpi=300)
    print(f"  Saved to {fig_dir}/fig3_confidence_tokens.{{pdf,png}}")
    
    print("\nAll figures generated successfully!")
    return True

if __name__ == '__main__':
    print("Phase 6.1: GLMM Analysis + Figure Generation")
    print("=" * 70)
    
    df = run_glmm()
    success = generate_figures(df)
    
    if success:
        print("\nPhase 6.1 complete!")
    else:
        print("\nPhase 6.1 encountered errors.")