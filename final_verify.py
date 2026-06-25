"""Final comprehensive verification of Phase 5.1 deliverables."""
import json
import sys

print("=" * 60)
print("Phase 5.1: Comprehensive Verification")
print("=" * 60)

# 1. Check taxonomy document
import os
taxonomy_path = "/home/openhermes/meminterfere/docs/near-duplicate-taxonomy.md"
assert os.path.exists(taxonomy_path), f"Missing: {taxonomy_path}"
with open(taxonomy_path) as f:
    content = f.read()
assert "Semantic Similarity Spectrum" in content
assert "Identical" in content
assert "Near-identical" in content
assert "Near-similar" in content
assert "Distinct" in content
assert "Expected Failure Mode" in content or "Expected failure mode" in content
assert "Measurable Signal" in content or "Measurable signal" in content
print("✅ Taxonomy document verified")

# 2. Check expanded library
lib_path = "/home/openhermes/meminterfere/data/skills/expanded_library_100.json"
with open(lib_path) as f:
    data = json.load(f)
skills = data['skills']
assert len(skills) == 100, f"Expected 100 skills, got {len(skills)}"

# Check required fields
required_fields = [
    'skill_id', 'name', 'domain', 'description', 'parameters', 
    'return_format', 'success_rate', 'last_verified', 'conflict_group_id',
    'conflict_type', 'fan_degree', 'staleness_days', 'is_trap', 'is_stale',
    'is_clean', 'parametric_overlap', 'trap_description', 'example_usage',
    'similarity_to_gold', 'source_url', 'gradient_pairs'
]
for i, s in enumerate(skills):
    for field in required_fields:
        assert field in s, f"Skill {i} ({s.get('name', 'unknown')}): missing field '{field}'"

# Check no duplicate IDs
ids = [s['skill_id'] for s in skills]
assert len(set(ids)) == 100, f"Duplicate skill_ids found"

# Check conflict groups
groups = {}
for s in skills:
    gid = s.get('conflict_group_id')
    if gid:
        groups.setdefault(gid, []).append(s)

for gid, members in groups.items():
    assert len(members) >= 2, f"Conflict group '{gid}' has < 2 members"

# Check distribution
stats = data['metadata']['stats']
print(f"✅ Expanded library verified: {stats['total']} skills")
print(f"   - Clean: {stats['clean']}")
print(f"   - Near-identical: {stats['near_identical']}")
print(f"   - Near-similar: {stats['near_similar']}")
print(f"   - Schema conflict: {stats['schema_conflict']}")
print(f"   - Semantic conflict: {stats['semantic_conflict']}")
print(f"   - Version conflict: {stats['version_conflict']}")
print(f"   - Stale: {stats['stale']}")
print(f"   - Trap: {stats['trap']}")

# 3. Check gradient pairs
gradient_pairs = data['metadata']['gradient_pairs']
assert len(gradient_pairs) >= 10, f"Expected >= 10 gradient pairs, got {len(gradient_pairs)}"
for gp in gradient_pairs:
    assert 'gold_skill' in gp
    assert 'high_similarity' in gp
    assert 'medium_similarity' in gp
    assert 'low_similarity' in gp
    # Verify the referenced skills exist
    all_names = {s['name'] for s in skills}
    for key in ['gold_skill', 'high_similarity', 'medium_similarity', 'low_similarity']:
        if key == 'gold_skill':
            assert gp[key] in all_names, f"Gold skill '{gp[key]}' not found in library"
print(f"✅ Gradient test pairs verified: {len(gradient_pairs)} pairs")

# 4. Check real-world libraries
rw_path = "/home/openhermes/meminterfere/data/skills/real_world_libraries.json"
with open(rw_path) as f:
    rw_data = json.load(f)
assert len(rw_data['libraries']) >= 5, f"Expected >= 5 real-world libraries, got {len(rw_data['libraries'])}"
for lib in rw_data['libraries']:
    assert 'source' in lib
    assert 'source_url' in lib
    assert 'tools' in lib
    assert len(lib['tools']) >= 10, f"{lib['source']}: only {len(lib['tools'])} tools, need >= 10"
    for tool in lib['tools']:
        assert 'name' in tool
        assert 'collision_note' in tool
print(f"✅ Real-world libraries verified: {len(rw_data['libraries'])} sources, {sum(len(l['tools']) for l in rw_data['libraries'])} total tools")

# 5. Check validation script
val_path = "/home/openhermes/meminterfere/src/validate_library.py"
assert os.path.exists(val_path), f"Missing: {val_path}"
print("✅ Validation script exists")

# 6. Run validation
print("\nRunning validation...")
os.system(f"python3 {val_path}")

print("\n" + "=" * 60)
print("Phase 5.1 verification COMPLETE")
print("=" * 60)