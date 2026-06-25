"""Diagnose conflict groups in the expanded library."""
import json
from collections import defaultdict

with open('/home/openhermes/meminterfere/data/skills/expanded_library_100.json') as f:
    data = json.load(f)

skills = data['skills']

# Find which groups are missing their gold skill
groups = defaultdict(list)
for s in skills:
    gid = s.get('conflict_group_id')
    if gid:
        groups[gid].append(s)

# Check if each group has at least one gold skill (is_clean or similarity_to_gold == 1.0)
for gid, members in sorted(groups.items()):
    has_gold = any(m.get('is_clean') or m.get('similarity_to_gold', 0) == 1.0 for m in members)
    if not has_gold:
        print(f"Group {gid} has NO gold skill! Members: {[m['name'] for m in members]}")
    else:
        gold_names = [m['name'] for m in members if m.get('is_clean') or m.get('similarity_to_gold', 0) == 1.0]
        other_names = [m['name'] for m in members if not (m.get('is_clean') or m.get('similarity_to_gold', 0) == 1.0)]
        print(f"Group {gid}: gold={gold_names}, others={other_names}")

# Also check: clean skills with no conflict group but that are referenced
clean_with_group = [s['name'] for s in skills if s.get('is_clean') and s.get('conflict_group_id')]
print(f"\nClean skills with conflict_group_id: {len(clean_with_group)}")
clean_no_group = [s['name'] for s in skills if s.get('is_clean') and not s.get('conflict_group_id')]
print(f"Clean skills with NO conflict_group_id: {len(clean_no_group)}")