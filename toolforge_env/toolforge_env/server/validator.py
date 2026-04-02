"""Stage-1 algorithmic plan validator.

Performs pure deterministic structural checks on a proposed plan before
it reaches the semantic slot judge. No LLM calls, no side effects.

Validation order:
  1. Empty plan            → EMPTY_PLAN    (penalty -1.0)
  2. Unknown tool name     → INVALID_TOOL  (penalty -0.8)
  3. Missing required param→ MISSING_PARAM (penalty -0.6)
  4. Extra unknown param   → EXTRA_PARAM   (penalty -0.3)
  5. All pass              → VALID         (penalty  0.0)
"""

from typing import Dict, List

from toolforge_env.models import Tool, ToolCall, ValidationResult


def validate_plan(
    plan: List[ToolCall],
    available_tools: Dict[str, Tool],
) -> ValidationResult:
    """Validate a plan against the available toolbox.

    Args:
        plan: Ordered list of tool calls the agent wants to execute.
        available_tools: Mapping of tool name → Tool definition.

    Returns:
        A ValidationResult indicating pass/fail with a reason code.
    """

    # 1. Empty plan
    if plan is None or len(plan) == 0:
        return ValidationResult(
            valid=False,
            reason="EMPTY_PLAN",
            penalty=-1.0,
            detail="Plan contains no tool calls.",
        )

    # 2. Tool existence
    for call in plan:
        if call.tool_name not in available_tools:
            return ValidationResult(
                valid=False,
                reason="INVALID_TOOL",
                penalty=-0.8,
                detail=f"Tool '{call.tool_name}' does not exist in toolbox.",
            )

    # 3 & 4. Parameter validation per call
    for call in plan:
        tool = available_tools[call.tool_name]

        # params_schema is JSON-Schema-like with "properties" and "required"
        allowed_params = set(tool.params_schema.get("properties", {}).keys())
        required_params = set(tool.params_schema.get("required", []))
        provided_params = set(call.params.keys())

        # Missing required parameters
        missing = required_params - provided_params
        if missing:
            return ValidationResult(
                valid=False,
                reason="MISSING_PARAM",
                penalty=-0.6,
                detail=(
                    f"Tool '{call.tool_name}' is missing required "
                    f"parameter(s): {sorted(missing)}."
                ),
            )

        # Extra parameters not in schema
        extra = provided_params - allowed_params
        if extra:
            return ValidationResult(
                valid=False,
                reason="EXTRA_PARAM",
                penalty=-0.3,
                detail=(
                    f"Tool '{call.tool_name}' received unexpected "
                    f"parameter(s): {sorted(extra)}."
                ),
            )

    # All checks passed
    return ValidationResult(
        valid=True,
        reason="VALID",
        penalty=0.0,
    )
