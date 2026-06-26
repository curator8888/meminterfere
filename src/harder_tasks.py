#!/usr/bin/env python3
"""
Phase 6.3: Harder Task Suite Loader

Loads the harder_tasks_40.json and converts to EvalTask objects.
These tasks have minimal keyword cues (7.3% vs 88.8% in original suite)
and specifically target near-duplicate interference scenarios.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import EvalTask

HARDER_TASKS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tasks', 'harder_tasks_40.json')


def load_harder_tasks(path: str = HARDER_TASKS_PATH) -> list[EvalTask]:
    """Load harder task suite from JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    tasks = []
    for t in data:
        tasks.append(EvalTask(
            task_id=t['task_id'],
            domain=t['domain'],
            difficulty=t['difficulty'],
            description=t['description'],
            expected_skill_ids=t['gold_skill_ids'],
            interference_potential=t['interference_potential'],
            natural_interference=t.get('natural_interference', []),
            requires_memory=t.get('requires_memory', False),
            expected_outcome='',
            subtasks=[],
        ))
    
    return tasks


# Make available for import
HARDER_TASKS = load_harder_tasks()


def validate_harder_tasks(tasks: list[EvalTask], skill_library) -> dict:
    """Validate that all gold skills exist in the skill library."""
    skill_names = {s.name for s in skill_library}
    missing_skills = set()
    ambiguous_tasks = []
    keyword_cue_tasks = []
    
    for t in tasks:
        # Check gold skill exists
        for sid in t.expected_skill_ids:
            if sid not in skill_names:
                missing_skills.add(sid)
        
        # Check keyword cues
        desc_lower = t.description.lower()
        for sid in t.expected_skill_ids:
            words = sid.replace('_', ' ').lower().split()
            if any(w in desc_lower for w in words if len(w) > 3):
                keyword_cue_tasks.append(t.task_id)
                break
    
    from collections import Counter
    diff_dist = Counter(t.difficulty for t in tasks)
    
    return {
        'valid': len(missing_skills) == 0,
        'n_tasks': len(tasks),
        'missing_skills': list(missing_skills),
        'keyword_cue_tasks': keyword_cue_tasks,
        'keyword_cue_pct': len(keyword_cue_tasks) / len(tasks) * 100,
        'domains': list(set(t.domain for t in tasks)),
        'difficulty_distribution': dict(diff_dist),
    }


if __name__ == '__main__':
    from evaluate_agent import _load_skills
    
    tasks = load_harder_tasks()
    print(f"Loaded {len(tasks)} harder tasks")
    
    skills = _load_skills()
    print(f"Loaded {len(skills)} skills")
    
    validation = validate_harder_tasks(tasks, skills)
    print(f"\nValidation:")
    print(f"  Valid: {validation['valid']}")
    print(f"  Missing skills: {validation['missing_skills']}")
    print(f"  Keyword cue tasks: {validation['keyword_cue_pct']:.1f}%")
    print(f"  Domains: {validation['domains']}")
    print(f"  Difficulty: {validation['difficulty_distribution']}")
    
    print(f"\nTask list:")
    for t in tasks:
        print(f"  {t.task_id}: {t.expected_skill_ids} [{t.interference_potential}]")