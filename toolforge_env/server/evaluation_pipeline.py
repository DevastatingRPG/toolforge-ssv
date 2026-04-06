"""
Evaluation Pipeline Orchestrator

Thin orchestration layer that chains the plan evaluation logic.

Pipeline flow:
1. run_sanity_validation(...)
2. run_slot_judgment(...)
3. plan_accuracy_score(...)
4. run_token_calculation(...)
5. reward_calculation(...)

Short-circuits if validation fails, or if harmful calls are present.
"""

from typing import Dict, List, Optional

from toolforge_env.models import MacroProposal, PipelineResult, Task, Tool, ToolCall
from toolforge_env.server.plan_evaluator import (
    get_relevant_slots,
    plan_accuracy_score,
    reward_calculation,
    run_sanity_validation,
    run_slot_judgment,
    run_token_calculation,
)


def run_evaluation_pipeline(
    plan: List[ToolCall],
    task: Task,
    available_tools: Dict[str, Tool],
    accepted_macros: List[Tool],
    baseline_token_cost: int,
    goal_achieved: bool = False,
    sequence_counts: Optional[Dict[str, int]] = None,
    macro_definitions: Optional[Dict[str, List[str]]] = None,
    macro_proposal: Optional[Tool] = None,
) -> PipelineResult:
    """Run the evaluation pipeline on a proposed plan."""
    
    # 1. run_sanity_validation(...)
    validation = run_sanity_validation(plan, available_tools)

    if not validation.valid:
        # short-circuit immediately
        return PipelineResult(
            validation=validation,
            slot_judgment=None,
            plan_accuracy=None,
            token_cost=None,
            reward=validation.penalty,
            passed_validation=False,
            summary=f"Validation failed: {validation.reason}",
        )

    # 2. run_slot_judgment(...)
    slot_judgment = run_slot_judgment(
        task_prompt=task.prompt,
        required_slots=task.required_slots,
        slot_definitions=get_relevant_slots(task.required_slots),
        available_tools=list(available_tools.values()),
        plan=plan,
    )
    
    if slot_judgment.harmful_calls_present:
        # short-circuit immediately
        return PipelineResult(
            validation=validation,
            slot_judgment=slot_judgment,
            plan_accuracy=None,
            token_cost=None,
            reward=-1.0,
            passed_validation=True,
            summary="Harmful tool call detected.",
        )

    # 3. plan_accuracy_score(...)
    plan_accuracy = plan_accuracy_score(
        validation_result=validation,
        slot_judgment=slot_judgment,
        task=task,
        goal_achieved=goal_achieved,
    )

    # 4. run_token_calculation(...)
    token_cost = run_token_calculation(
        plan=plan,
        accepted_macros=accepted_macros,
        task=task,
        available_tools=available_tools,
        sequence_counts=sequence_counts,
        macro_definitions=macro_definitions,
        macro_proposal=macro_proposal,
    )

    # 5. reward_calculation(...)
    final_reward = reward_calculation(plan_accuracy, token_cost)

    return PipelineResult(
        validation=validation,
        slot_judgment=slot_judgment,
        plan_accuracy=plan_accuracy,
        token_cost=token_cost,
        reward=final_reward,
        passed_validation=True,
        summary="Plan validated, slots judged, final reward applied.",
    )
