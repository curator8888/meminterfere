#!/usr/bin/env python3
"""
Phase 6.4: Similarity Gradient Test

Tests whether the interference effect scales with skill similarity.
Uses the actual taxonomy: near_identical (highest), version_conflict, 
schema_conflict, semantic_conflict, near_similar (lowest interference).

Each gradient condition shows ONLY clean skills + interference at one similarity level.
This isolates the dose-response curve of interference similarity.

Design:
- 5 gradient conditions + 1 baseline
- 4 models × 6 conditions × 80 tasks × 1 temp (0.0) = 1920 runs
- Est cost: ~$3
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from evaluate_agent import _load_skills, get_skills_by_type
from collections import Counter

GRADIENT_CONDITIONS = {
    "clean_memory": "Baseline: only clean skills (40 skills)",
    "grad_near_identical": "Clean + near-identical interference (40+17=57 skills)",
    "grad_version_conflict": "Clean + version conflicts (40+6=46 skills)",
    "grad_schema_conflict": "Clean + schema conflicts (40+7=47 skills)",
    "grad_semantic_conflict": "Clean + semantic conflicts (40+5=45 skills)",
    "grad_near_similar": "Clean + near-similar interference (40+10=50 skills)",
}

# Map from gradient condition name -> conflict_type filter
GRADIENT_TYPE_MAP = {
    "grad_near_identical": "near_identical",
    "grad_version_conflict": "version_conflict",
    "grad_schema_conflict": "schema_conflict",
    "grad_semantic_conflict": "semantic_conflict",
    "grad_near_similar": "near_similar",
}

def get_library_for_gradient_condition(condition_name: str) -> list:
    """Build a skill library for a gradient condition."""
    clean = get_skills_by_type("clean")
    all_skills = _load_skills()
    
    if condition_name == "clean_memory":
        return clean
    
    conflict_type = GRADIENT_TYPE_MAP.get(condition_name)
    if conflict_type is None:
        raise ValueError(f"Unknown gradient condition: {condition_name}")
    
    interfering = [s for s in all_skills if s.conflict_type == conflict_type]
    return clean + interfering

def validate_gradient():
    """Validate the gradient test setup."""
    all_skills = _load_skills()
    clean = get_skills_by_type("clean")
    types = Counter(s.conflict_type for s in all_skills)
    
    print("Skill library composition:")
    print(f"  Clean (none): {types.get('none', 0)}")
    print(f"  Near-identical: {types.get('near_identical', 0)}")
    print(f"  Version conflict: {types.get('version_conflict', 0)}")
    print(f"  Schema conflict: {types.get('schema_conflict', 0)}")
    print(f"  Semantic conflict: {types.get('semantic_conflict', 0)}")
    print(f"  Near-similar: {types.get('near_similar', 0)}")
    print()
    
    for cond, desc in GRADIENT_CONDITIONS.items():
        lib = get_library_for_gradient_condition(cond)
        print(f"  {cond}: {len(lib)} skills — {desc}")
    
    # Verify gradient ordering: near_identical should cause most interference
    ordering = ["near_identical", "version_conflict", "schema_conflict", 
                "semantic_conflict", "near_similar"]
    print(f"\nExpected interference gradient (most → least):")
    print(f"  {' > '.join(ordering)}")
    
    return {
        "condition_sizes": {cond: len(get_library_for_gradient_condition(cond)) 
                          for cond in GRADIENT_CONDITIONS},
    }

if __name__ == "__main__":
    validate_gradient()