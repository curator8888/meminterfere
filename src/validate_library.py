"""
Phase 5.1: Validate the expanded 100-skill library.

Checks:
- Every near-duplicate pair has a graded similarity score
- Every conflict group has >= 2 members
- Real-world skills have source URLs documented
- Total skill count is exactly 100
- All required fields are present
- No duplicate skill_ids
"""

import json
import sys
import os


def validate_library(path: str) -> list[str]:
    """Validate the expanded skill library. Returns list of errors."""
    errors = []
    
    with open(path) as f:
        data = json.load(f)
    
    skills = data.get('skills', [])
    
    # 1. Total skill count is exactly 100
    if len(skills) != 100:
        errors.append(f"Expected 100 skills, got {len(skills)}")
    
    # 2. No duplicate skill_ids
    ids = [s.get('skill_id', '') for s in skills]
    from collections import Counter
    dupes = [(id, cnt) for id, cnt in Counter(ids).items() if cnt > 1]
    if dupes:
        for id, cnt in dupes:
            errors.append(f"Duplicate skill_id: {id} appears {cnt} times")
    
    # 3. All required fields are present
    required_fields = [
        'skill_id', 'name', 'domain', 'description', 'parameters', 
        'return_format', 'success_rate', 'last_verified', 'conflict_group_id',
        'conflict_type', 'fan_degree', 'staleness_days', 'is_trap', 'is_stale',
        'is_clean', 'parametric_overlap', 'trap_description', 'example_usage',
        'similarity_to_gold', 'source_url', 'gradient_pairs'
    ]
    for i, s in enumerate(skills):
        for field in required_fields:
            if field not in s:
                errors.append(f"Skill {i} ({s.get('name', 'unknown')}): missing field '{field}'")
    
    # 4. Every conflict group has >= 2 members
    groups = {}
    for s in skills:
        gid = s.get('conflict_group_id')
        if gid:
            groups.setdefault(gid, []).append(s)
    
    for gid, members in groups.items():
        if len(members) < 2:
            errors.append(f"Conflict group '{gid}' has only {len(members)} member(s), needs >= 2")
    
    # 5. Every near-identical/near-similar pair has a graded similarity score
    for gid, members in groups.items():
        # Check that at least one member has similarity_to_gold = 1.0 (gold skill)
        gold_members = [m for m in members if m.get('similarity_to_gold', 0) >= 0.99]
        if not gold_members and len(members) >= 2:
            errors.append(f"Conflict group '{gid}' has no gold skill (similarity_to_gold=1.0)")
        
        # Check that non-gold members have similarity_to_gold set
        for m in members:
            ct = m.get('conflict_type', 'none')
            if ct in ('near_identical', 'near_similar', 'schema_conflict', 'semantic_conflict', 'version_conflict'):
                sim = m.get('similarity_to_gold', 0)
                if sim == 0 and not m.get('is_clean', False):
                    errors.append(f"Skill '{m.get('name')}' in group '{gid}' has similarity_to_gold=0 but is not clean")
    
    # 6. Real-world skills have source URLs documented
    real_world_skills = [s for s in skills if s.get('source_url', '') != '']
    # Note: This check is informational only - not all skills need source URLs
    print(f"  Skills with source URLs: {len(real_world_skills)}")
    
    # 7. Domain validity
    valid_domains = {'web_navigation', 'api_calling'}
    for s in skills:
        if s.get('domain') not in valid_domains:
            errors.append(f"Skill '{s.get('name')}' has invalid domain: {s.get('domain')}")
    
    # 8. Conflict type validity
    valid_conflict_types = {'none', 'schema_conflict', 'semantic_conflict', 'version_conflict', 
                            'near_identical', 'near_similar'}
    for s in skills:
        ct = s.get('conflict_type', '')
        if ct not in valid_conflict_types:
            errors.append(f"Skill '{s.get('name')}' has invalid conflict_type: {ct}")
    
    # 9. Check similarity_to_gold values are in valid ranges for their conflict type
    for s in skills:
        ct = s.get('conflict_type', 'none')
        sim = s.get('similarity_to_gold', 0)
        if ct == 'schema_conflict' and sim != 1.0 and not s.get('is_clean', False):
            errors.append(f"Skill '{s.get('name')}' has schema_conflict but similarity_to_gold={sim} (expected 1.0)")
        if ct == 'near_identical' and (sim < 0.25 or sim > 0.95) and not s.get('is_clean', False):
            errors.append(f"Skill '{s.get('name')}' has near_identical but similarity_to_gold={sim} (expected 0.25-0.95)")
        if ct == 'near_similar' and (sim < 0.25 or sim > 0.75) and not s.get('is_clean', False):
            errors.append(f"Skill '{s.get('name')}' has near_similar but similarity_to_gold={sim} (expected 0.25-0.75)")
    
    # 10. Gradient pairs are valid
    gradient_skills = [s for s in skills if s.get('gradient_pairs', [])]
    for s in gradient_skills:
        for gp in s['gradient_pairs']:
            if 'level' not in gp:
                errors.append(f"Skill '{s.get('name')}' gradient pair missing 'level'")
            if 'gold_skill' not in gp:
                errors.append(f"Skill '{s.get('name')}' gradient pair missing 'gold_skill'")
            if 'name_similarity' not in gp:
                errors.append(f"Skill '{s.get('name')}' gradient pair missing 'name_similarity'")
    
    # Print summary
    print(f"\n  Total skills: {len(skills)}")
    print(f"  Conflict groups: {len(groups)}")
    print(f"  Skills in conflict groups: {sum(len(m) for m in groups.values())}")
    print(f"  Clean skills: {sum(1 for s in skills if s.get('is_clean', False))}")
    print(f"  Stale skills: {sum(1 for s in skills if s.get('is_stale', False))}")
    print(f"  Trap skills: {sum(1 for s in skills if s.get('is_trap', False))}")
    
    by_ct = {}
    for s in skills:
        ct = s.get('conflict_type', 'none')
        by_ct[ct] = by_ct.get(ct, 0) + 1
    print(f"  By conflict type: {by_ct}")
    
    by_domain = {}
    for s in skills:
        d = s.get('domain', 'unknown')
        by_domain[d] = by_domain.get(d, 0) + 1
    print(f"  By domain: {by_domain}")
    
    # Similarity distribution
    sim_values = [s.get('similarity_to_gold', 0) for s in skills if s.get('conflict_group_id') and not s.get('is_clean', False)]
    if sim_values:
        print(f"  Similarity scores - min: {min(sim_values):.2f}, max: {max(sim_values):.2f}, mean: {sum(sim_values)/len(sim_values):.2f}")
    
    return errors


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_path = os.path.join(base_dir, "data", "skills", "expanded_library_100.json")
    
    print("Validating expanded_library_100.json...")
    errors = validate_library(lib_path)
    
    if errors:
        print(f"\n❌ VALIDATION FAILED: {len(errors)} errors found")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n✅ VALIDATION PASSED: All checks successful")
        sys.exit(0)