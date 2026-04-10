import sys
import os

# Ensure the server modules are in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.evaluation.plan_evaluator import compute_macro_miss_penalty
from models import ToolCall, Tool

def test_macro_miss_penalty():
    print("Testing compute_macro_miss_penalty...")
    
    macroAB = Tool(
        name="macro_AB",
        description="Macro AB",
        is_macro=True,
        steps=[ToolCall(tool_name="A"), ToolCall(tool_name="B")]
    )
    
    accepted_macros = [macroAB]
    
    # 1. slot_ratio = 1.0, 1 miss
    # Expected: base penalty = MAX (0.05). miss = 1 -> -0.05
    plan_miss_once = [ToolCall(tool_name="A"), ToolCall(tool_name="B")]
    res = compute_macro_miss_penalty(plan_miss_once, accepted_macros, slot_ratio=1.0)
    print(f"1. slot_ratio=1.0, 1 miss -> Penalty: {res:.3f} (Expected: -0.050)")
    assert abs(res - (-0.05)) < 1e-4

    # 2. slot_ratio = 0.65, 1 miss
    # Expected: base penalty = MIN (0.02). miss = 1 -> -0.02
    res = compute_macro_miss_penalty(plan_miss_once, accepted_macros, slot_ratio=0.65)
    print(f"2. slot_ratio=0.65, 1 miss -> Penalty: {res:.3f} (Expected: -0.020)")
    assert abs(res - (-0.02)) < 1e-4

    # 3. slot_ratio = 0.5, 1 miss
    # Expected: < 0.65 threshold -> 0.0
    res = compute_macro_miss_penalty(plan_miss_once, accepted_macros, slot_ratio=0.5)
    print(f"3. slot_ratio=0.50, 1 miss -> Penalty: {res:.3f} (Expected: 0.000)")
    assert abs(res) < 1e-4
    
    # 4. slot_ratio = 1.0, 2 misses, non-overlapping
    # Base penalty = 0.05
    # Total = (0.05) + (0.05 + 0.01) = 0.11 -> Cap is 0.10, so -0.10
    plan_miss_twice = [ToolCall(tool_name="A"), ToolCall(tool_name="B"), ToolCall(tool_name="A"), ToolCall(tool_name="B")]
    res = compute_macro_miss_penalty(plan_miss_twice, accepted_macros, slot_ratio=1.0)
    print(f"4. slot_ratio=1.0, 2 misses -> Penalty: {res:.3f} (Expected: -0.100 due to cap)")
    assert abs(res - (-0.10)) < 1e-4
    
    # 5. slot_ratio = 0.65, 2 misses, non-overlapping
    # Base penalty = 0.02
    # Total = (0.02) + (0.02 + 0.01) = 0.05 -> -0.05
    res = compute_macro_miss_penalty(plan_miss_twice, accepted_macros, slot_ratio=0.65)
    print(f"5. slot_ratio=0.65, 2 misses -> Penalty: {res:.3f} (Expected: -0.050)")
    assert abs(res - (-0.05)) < 1e-4

    # 6. Used macro instead of atomics
    # Expected: miss_count = 0 -> 0.0
    plan_hit = [ToolCall(tool_name="macro_AB")]
    res = compute_macro_miss_penalty(plan_hit, accepted_macros, slot_ratio=1.0)
    print(f"6. used macro directly -> Penalty: {res:.3f} (Expected: 0.000)")
    assert abs(res) < 1e-4
    
    # 7. overlapping sequence check
    macroABA = Tool(
        name="macro_ABA",
        description="Macro ABA",
        is_macro=True,
        steps=[ToolCall(tool_name="A"), ToolCall(tool_name="B"), ToolCall(tool_name="A")]
    )
    # Plan: A B A B A
    # If we greedily take A B A, we are left with B A. The second A B A is not complete.
    # Total misses = 1
    plan_overlap = [ToolCall(tool_name="A"), ToolCall(tool_name="B"), ToolCall(tool_name="A"), ToolCall(tool_name="B"), ToolCall(tool_name="A")]
    res = compute_macro_miss_penalty(plan_overlap, [macroABA], slot_ratio=1.0)
    print(f"7. overlapping sequences -> Penalty: {res:.3f} (Expected: -0.050, 1 miss only)")
    assert abs(res - (-0.05)) < 1e-4

    print("All tests passed!")

if __name__ == "__main__":
    test_macro_miss_penalty()
