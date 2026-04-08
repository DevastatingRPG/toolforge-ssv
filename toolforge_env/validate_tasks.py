import sys

# Validate tools import
try:
    from server.tools import build_atomic_tools
    print("1. tools.py loads cleanly: True")
except Exception as e:
    print(f"1. tools.py loads cleanly: False ({e})")
    sys.exit(1)

# Validate tasks import
try:
    from server.inputs.simulated.tasks import TASKS
    print("2. tasks.py imports cleanly: True")
except Exception as e:
    print(f"2. tasks.py imports cleanly: False ({e})")
    sys.exit(1)

# Validate slots import
try:
    from server.slots import DEVOPS_SLOTS
    print("Slots loaded cleanly.")
except Exception as e:
    print(f"Slots failed to load: {e}")
    sys.exit(1)

# Verify Easy Groups
easy_groups = [g for g in TASKS if g['task_id'].startswith('easy')]
all_easy_ok = True
for g in easy_groups:
    if len(g['tasks']) != 15:
        all_easy_ok = False
        print(f"FAILED: Easy group '{g['task_id']}' has {len(g['tasks'])} tasks (expected 15)")
print(f"3. Every easy group has exactly 15 tasks: {all_easy_ok}")

# Verify Medium Groups
medium_groups = [g for g in TASKS if g['task_id'].startswith('medium')]
all_medium_ok = True
for g in medium_groups:
    if len(g['tasks']) != 20:
        all_medium_ok = False
        print(f"FAILED: Medium group '{g['task_id']}' has {len(g['tasks'])} tasks (expected 20)")
print(f"4. Every medium group has exactly 20 tasks: {all_medium_ok}")

# Check slots and baselines
all_slots = set(DEVOPS_SLOTS.keys())
slots_ok = True
baselines_ok = True
for g in TASKS:
    for t in g['tasks']:
        
        # Verify required_slots
        for s in t.required_slots:
            if s not in all_slots:
                slots_ok = False
                print(f"FAILED: Task '{t.id}' has an invalid slot '{s}'")
                
        # Verify baseline
        if t.baseline_call_count is None or t.baseline_call_count <= 0 or t.baseline_call_count > 10:
            baselines_ok = False
            print(f"FAILED: Task '{t.id}' has unreasonable baseline {t.baseline_call_count}")

print(f"5. All required_slots exist in slot definitions: {slots_ok}")
print(f"6. All baseline_call_count values are plausible: {baselines_ok}")

tools_list = [t.name for t in build_atomic_tools()]
print(f"Total tools length is: {len(tools_list)}")
print(f"Tools: {tools_list}")
