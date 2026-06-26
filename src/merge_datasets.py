#!/usr/bin/env python3
"""Merge Claude results into the full 4-model dataset and re-run analyses."""
import json, sys, os

# Load 3-model dataset
with open('data/results/phase5_full/multi_model_results_latest.json') as f:
    data_3model = json.load(f)

# Load Claude dataset
with open('data/results/phase5_claude/multi_model_results_latest.json') as f:
    data_claude = json.load(f)

results_3 = data_3model['results']
results_claude = data_claude['results']

print(f"3-model results: {len(results_3)}")
print(f"Claude results: {len(results_claude)}")

# Merge
all_results = results_3 + results_claude
print(f"Merged results: {len(all_results)}")

# Save merged dataset
output = {
    'metadata': {
        'description': 'MemInterfere Phase 5.5 full experiment - 4 models',
        'models': ['llama-3.1-8b-instruct', 'grok-3-mini', 'gpt-4o-mini', 'claude-haiku-4.5'],
        'conditions': ['oracle', 'no_memory', 'clean_memory', 'clean_interference', 'all_memory'],
        'temperatures': [0.0, 0.3, 0.7],
        'n_tasks': 80,
        'n_models': 4,
        'total_runs': len(all_results),
        'date': '2026-06-26'
    },
    'results': all_results
}

os.makedirs('data/results/phase5_full_4model', exist_ok=True)
with open('data/results/phase5_full_4model/multi_model_results_latest.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {len(all_results)} results to data/results/phase5_full_4model/")

# Quick summary
models = set(r['model'] for r in all_results)
for model in sorted(models):
    mr = [r for r in all_results if r['model'] == model]
    correct = sum(1 for r in mr if r['skill_correct'])
    print(f"  {model}: {len(mr)} runs, {correct/len(mr)*100:.1f}% accuracy")

for cond in ['oracle', 'no_memory', 'clean_memory', 'clean_interference', 'all_memory']:
    cr = [r for r in all_results if r['condition'] == cond]
    correct = sum(1 for r in cr if r['skill_correct'])
    print(f"  {cond}: {len(cr)} runs, {correct/len(cr)*100:.1f}% accuracy")