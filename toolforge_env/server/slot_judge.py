"""Stage-2 semantic slot judge (placeholder).

This module will eventually call an LLM to determine which semantic slots
each tool call in a plan fills. For now it returns a hardcoded
success-shaped result so downstream integration can be built and tested
without requiring an API key.

Public API:
    judge_plan(task_prompt, required_slots, slot_definitions,
               available_tools, plan) -> SlotJudgmentResult
"""

from typing import Dict, List

from toolforge_env.models import (
    SlotJudgmentResult,
    Tool,
    ToolCall,
    ToolEvaluation,
)


def judge_plan(
    task_prompt: str,
    required_slots: List[str],
    slot_definitions: Dict[str, str],
    available_tools: List[Tool],
    plan: List[ToolCall],
) -> SlotJudgmentResult:
    """Evaluate a validated plan against the task's semantic slots.

    In production this will call an LLM to determine per-call slot
    assignments. The current implementation returns a placeholder
    all-success result.

    Args:
        task_prompt:      The human-readable task description.
        required_slots:   Slot names the task requires.
        slot_definitions: Mapping of slot name → description.
        available_tools:  Tools the agent can use.
        plan:             The validated list of tool calls.

    Returns:
        SlotJudgmentResult with one evaluation per tool call and
        all required slots marked as filled.
    """
    return _build_placeholder_result(plan, required_slots)


def _build_placeholder_result(
    plan: List[ToolCall],
    required_slots: List[str],
) -> SlotJudgmentResult:
    """Build a success-shaped placeholder result.

    Each tool call is assigned to a required slot in order (if available).
    All required slots are reported as filled.
    """
    evaluations: List[ToolEvaluation] = []
    for idx, call in enumerate(plan):
        # Assign a slot from required_slots if we haven't exhausted them
        slot = required_slots[idx] if idx < len(required_slots) else None
        evaluations.append(
            ToolEvaluation(
                tool_call_index=idx,
                tool_name=call.tool_name,
                fills_slot=slot,
                correct=True,
                reason="Placeholder: accepted by stub judge.",
            )
        )

    return SlotJudgmentResult(
        evaluations=evaluations,
        slots_filled=list(required_slots),
        slots_missing=[],
        task_complete=True,
    )
