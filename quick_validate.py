"""Quick validation script for expanded_library_100.json"""
import json
from collections import Counter

with open('/home/openhermes/meminterfere/data/skills/expanded_library_100.json') as f:
    data = json.load(f)

skills = data['skills']
print(f"Total skills: {len(skills)}")

ids = [s['skill_id'] for s in skills]
print(f"Unique skill_ids: {len(set(ids))}")

dupes = [(id, cnt) for id, cnt in Counter(ids).items() if cnt > 1]
if dupes:
    print(f"DUPLICATE skill_ids: {dupes}")
else:
    print("No duplicate skill_ids")

groups = {}
for s in skills:
    gid = s.get('conflict_group_id')
    if gid:
        groups.setdefault(gid, []).append(s['name'])

print(f"\nConflict groups ({len(groups)}):")
for gid, members in sorted(groups.items()):
    print(f"  {gid}: {members}")

has_sim = sum(1 for s in skills if s.get('similarity_to_gold', 0) > 0 and s.get('conflict_group_id'))
print(f"\nSkills with similarity_to_gold > 0 in conflict group: {has_sim}")
print(f"Skills with gradient_pairs: {sum(1 for s in skills if s.get('gradient_pairs', []))}")

# Check all required fields
required_fields = ['skill_id', 'name', 'domain', 'description', 'parameters', 'return_format',
                   'success_rate', 'last_verified', 'conflict_group_id', 'conflict_type',
                   'fan_degree', 'staleness_days', 'is_trap', 'is_stale', 'is_clean',
                   'parametric_overlap', 'trap_description', 'example_usage', 'similarity_to_gold',
                   'source_url', 'gradient_pairs']

missing = []
for i, s in enumerate(skills):
    for field in required_fields:
        if field not in s:
            missing.append(f"Skill {i} ({s.get('name', 'unknown')}): missing {field}")

if missing:
    print(f"\nMissing fields ({len(missing)}):")
    for m in missing[:10]:
        print(f"  {m}")
else:
    print("\nAll required fields present!")

# Conflict group sizes
small_groups = [(gid, len(m)) for gid, m in groups.items() if len(m) < 2]
if small_groups:
    print(f"\nConflict groups with < 2 members: {small_groups}")
else:
    print("\nAll conflict groups have >= 2 members")