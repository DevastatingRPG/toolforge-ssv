"""Judge pipeline orchestrator.

This is the single entry point the environment calls to evaluate an
agent's proposed plan. It chains the four judge stages in order:

    1. Structural validation  (validator.py)
    2. Semantic slot judgment  (slot_judge.py)
    3. Reward calculation      (reward_calculator.py)
    4. Token-cost calculation  (token_calculator.py)

If Stage 1 fails the pipeline short-circuits and returns a partial
PipelineResult with only the validation stage populated.

Public API:
    run_judge_pipeline(plan, task, available_tools,
                       accepted_macros, baseline_token_cost) -> PipelineResult
"""

from typing import Dict, List

from toolforge_env.models import (
    PipelineResult,
    Task,
    Tool,
    ToolCall,
)
from toolforge_env.server.reward_calculator import calculate_reward
from toolforge_env.server.slot_judge import judge_plan
from toolforge_env.server.slots import DEVOPS_SLOTS
from toolforge_env.server.token_calculator import calculate_token_cost
from toolforge_env.server.validator import validate_plan


def get_relevant_slots(required_slots: List[str]) -> Dict[str, str]:
    """Return the subset of DEVOPS_SLOTS matching *required_slots*.

    Slot names that do not appear in the global library are silently
    omitted rather than raising an error.
    """
    return {
        name: DEVOPS_SLOTS[name]
        for name in required_slots
        if name in DEVOPS_SLOTS
    }


def run_judge_pipeline(
    plan: List[ToolCall],
    task: Task,
    available_tools: Dict[str, Tool],
    accepted_macros: List[Tool],
    baseline_token_cost: int,
) -> PipelineResult:
    """Run the full four-stage judge pipeline on a proposed plan.

    Args:
        plan:                The agent's proposed tool-call sequence.
        task:                The task definition being solved.
        available_tools:     Dict of tool-name → Tool currently available.
        accepted_macros:     Macros approved earlier in the episode.
        baseline_token_cost: Naive atomic cost for this task.

    Returns:
        PipelineResult aggregating all stage outputs and a final score.
    """

    # Stage 1 — structural validation
    validation = validate_plan(plan, available_tools)

    if not validation.valid:
        return PipelineResult(
            validation=validation,
            slot_judgment=None,
            reward=None,
            token_cost=None,
            final_score=max(0.0, 1.0 + validation.penalty),
            passed_validation=False,
            summary=f"Validation failed: {validation.reason}",
        )

    # Stage 2 — semantic slot judgment
    slot_judgment = judge_plan(
        task_prompt=task.prompt,
        required_slots=task.required_slots,
        slot_definitions=get_relevant_slots(task.required_slots),
        available_tools=list(available_tools.values()),
        plan=plan,
    )

    # Stage 3 — reward calculation
    reward = calculate_reward(validation, slot_judgment, task)

    # Stage 4 — token-cost calculation
    token_cost = calculate_token_cost(
        plan=plan,
        accepted_macros=accepted_macros,
        baseline_token_cost=baseline_token_cost,
    )

    # Final blended score: 70% correctness, 30% efficiency
    final_score = (0.7 * reward.raw_score) + (0.3 * token_cost.efficiency_score)
    final_score = max(0.0, min(1.0, final_score))

    return PipelineResult(
        validation=validation,
        slot_judgment=slot_judgment,
        reward=reward,
        token_cost=token_cost,
        final_score=final_score,
        passed_validation=True,
        summary="Plan validated, slots judged, placeholder reward applied.",
    )
