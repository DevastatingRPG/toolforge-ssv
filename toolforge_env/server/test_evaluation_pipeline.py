"""
Test file for the Evaluation Pipeline.

Covers:
A. Structural failure
B. Harmful semantic plan
C. Perfect atomic correct plan
D. Correct plan with harmless unnecessary call
E. Macro-efficient plan
"""

import logging

from toolforge_env.models import Task, Tool, ToolCall
from toolforge_env.server.evaluation_pipeline import run_evaluation_pipeline

logging.basicConfig(level=logging.INFO)

def run_tests():
    # Setup dummy data
    task = Task(
        id="dummy-test-task",
        prompt="Execute tool_a then tool_b",
        difficulty="easy",
        required_steps=["tool_a", "tool_b"],
        core_steps=["tool_a"],
        required_slots=["TEST_SLOT_A", "TEST_SLOT_B"],
        baseline_token_cost=5,
    )

    tool_a = Tool(
        name="tool_a",
        description="does A",
        params_schema={"type": "object", "properties": {"param_a": {}}, "required": ["param_a"]},
        token_cost=2,
    )
    tool_b = Tool(
        name="tool_b",
        description="does B",
        params_schema={"type": "object", "properties": {"param_b": {}}, "required": ["param_b"]},
        token_cost=3,
    )
    tool_delete = Tool(
        name="delete_db",
        description="deletes things",
        params_schema={"type": "object", "properties": {}, "required": []},
        token_cost=5,
    )
    macro_tool = Tool(
        name="macro_ab",
        description="does A and B cheaply",
        params_schema={"type": "object", "properties": {"param_a": {}, "param_b": {}}, "required": ["param_a", "param_b"]},
        token_cost=2, # Cheap macro
        is_macro=True
    )

    available_tools = {
        "tool_a": tool_a,
        "tool_b": tool_b,
        "delete_db": tool_delete,
        "macro_ab": macro_tool
    }

    # A. Structural Failure
    print("\n--- A. Test: Structural Failure (Empty Plan) ---")
    res1 = run_evaluation_pipeline([], task, available_tools, [], 5)
    print(f"Passed Validation: {res1.passed_validation}")
    print(f"Final Reward: {res1.reward:.3f} (Expected: -1.0)")

    print("\n--- A. Test: Structural Failure (Missing Params) ---")
    res3 = run_evaluation_pipeline(
        [ToolCall(tool_name="tool_a", params={}, token_cost=2)], 
        task, available_tools, [], 5
    )
    print(f"Passed Validation: {res3.passed_validation}")
    print(f"Reason: {res3.validation.reason if res3.validation else 'N/A'}")

    # B. Harmful semantic plan
    print("\n--- B. Test: Harmful Semantic Plan ---")
    res4 = run_evaluation_pipeline([
        ToolCall(tool_name="tool_a", params={"param_a": "val"}, token_cost=2),
        ToolCall(tool_name="delete_db", params={}, token_cost=5)
    ], task, available_tools, [], 5)
    print(f"Passed Validation: {res4.passed_validation}")
    print(f"Harmful Calls Present: {res4.slot_judgment.harmful_calls_present}")
    print(f"Final Reward: {res4.reward:.3f} (Expected: -1.0)")

    # C. Perfect atomic correct plan
    print("\n--- C. Test: Perfect Atomic Correct Plan ---")
    res5 = run_evaluation_pipeline([
        ToolCall(tool_name="tool_a", params={"param_a": "val"}, token_cost=2),
        ToolCall(tool_name="tool_b", params={"param_b": "val"}, token_cost=3)
    ], task, available_tools, [], 5)
    print(f"Passed Validation: {res5.passed_validation}")
    print(f"Stage 3 (Plan Accuracy): {res5.plan_accuracy.score:.3f} (Expected: 0.0)")
    print(f"Stage 4 (Token Cost Efficiency): {res5.token_cost.efficiency_score:.3f} (Expected: 0.0)")
    print(f"Final Reward: {res5.reward:.3f} (Expected: 0.0)")

    # D. Correct plan with harmless unnecessary call
    print("\n--- D. Test: Correct Plan with Harmless Unnecessary Call ---")
    res6 = run_evaluation_pipeline([
        ToolCall(tool_name="tool_a", params={"param_a": "val"}, token_cost=2),
        ToolCall(tool_name="tool_b", params={"param_b": "val"}, token_cost=3),
        ToolCall(tool_name="tool_a", params={"param_a": "val2"}, token_cost=2),  # Unnecessary duplicate
    ], task, available_tools, [], 5)
    print(f"Passed Validation: {res6.passed_validation}")
    print(f"Stage 3 (Plan Accuracy): {res6.plan_accuracy.score:.3f}")
    print(f"Stage 4 (Token Cost Efficiency): {res6.token_cost.efficiency_score:.3f}")
    print(f"Final Reward: {res6.reward:.3f} (Expected: < 0.0 but > -1.0)")

    # E. Macro-efficient plan
    print("\n--- E. Test: Macro-Efficient Plan ---")
    res7 = run_evaluation_pipeline([
        ToolCall(tool_name="macro_ab", params={"param_a": "val", "param_b": "val"}, token_cost=2),
    ], task, available_tools, accepted_macros=[macro_tool], baseline_token_cost=5)
    print(f"Passed Validation: {res7.passed_validation}")
    print(f"Tokens Used: {res7.token_cost.tokens_used}, Baseline: {res7.token_cost.baseline_tokens}")
    print(f"Stage 4 (Token Cost Efficiency): {res7.token_cost.efficiency_score:.3f} (Expected: > 0.0)")
    print(f"Final Reward: {res7.reward:.3f} (Expected: > 0.0)")


if __name__ == "__main__":
    run_tests()
    print("\n=== Pipeline evaluation tests complete ===")
