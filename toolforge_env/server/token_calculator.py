"""Stage-4 token-cost calculator (placeholder).

Computes an efficiency score comparing the agent's token usage against
the task's baseline cost. Currently returns hardcoded placeholder values.
Real efficiency logic (macro savings, ratio curves) will be added later.

Public API:
    calculate_token_cost(plan, accepted_macros, baseline_token_cost)
        -> TokenCostResult
"""

from typing import List

from toolforge_env.models import TokenCostResult, Tool, ToolCall


def calculate_token_cost(
    plan: List[ToolCall],
    accepted_macros: List[Tool],
    baseline_token_cost: int,
) -> TokenCostResult:
    """Compute a token-efficiency score for a plan.

    Args:
        plan:                The list of tool calls in the agent's plan.
        accepted_macros:     Macros the agent has had approved so far.
        baseline_token_cost: Naive atomic cost for the current task.

    Returns:
        TokenCostResult with placeholder scores.
    """
    return TokenCostResult(
        tokens_used=10,
        baseline_tokens=10,
        efficiency_ratio=1.0,
        efficiency_score=1.0,
        macro_savings=0,
    )
