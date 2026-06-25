"""Verify real_world_libraries.json"""
import json
with open('/home/openhermes/meminterfere/data/skills/real_world_libraries.json') as f:
    data = json.load(f)
for lib in data['libraries']:
    print(f'{lib["source"]}: {len(lib["tools"])} tools, collision_type={lib["collision_type"]}')
print(f'Total tools across all libraries: {sum(len(lib["tools"]) for lib in data["libraries"])}')