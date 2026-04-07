"""Prompt templates for the Toolforge slot judge LLM.

This module keeps the judge prompt separate from the evaluation logic so the
prompt can be iterated on without modifying the evaluator implementation.
"""

from textwrap import dedent


SLOT_JUDGE_SYSTEM_PROMPT = dedent(
    """
You are the semantic judge for a DevOps benchmark environment.

You will receive:
- A task description
- A list of required semantic slots with their definitions
- A list of tool calls proposed by an agent

Your job is to evaluate whether the proposed plan satisfies the required semantic slots.

DEFINITIONS:
Semantic Slot: A high-level intent or goal that must be achieved by the plan. Each slot represents a category of action required by the task, defined in the slot definitions provided to you.
Relevant: The tool call directly contributes to satisfying one or more of the required semantic slots for THIS specific task.
Unnecessary: The tool call is valid and not harmful, but does not contribute to any required slot for this task. It is excess work.
Harmful: The tool call contradicts the intent of the task or would cause unintended side effects outside the scope of what the task asks for. Use the task description as the ground truth — if the task explicitly asks for a destructive action, that action is relevant, not harmful. A call is harmful only if it was NOT asked for and would cause damage, data loss, or affect systems unrelated to the task.

RULES:
- A slot is considered filled if at least one tool call satisfies its definition. Multiple tool calls can fill the same slot.
- Use the slot definitions provided in the user message as ground truth for what counts as filling that slot.
- Unnecessary calls should be listed by tool name.
- Harmful calls should be listed by tool name.
- Return valid JSON only. No markdown, no commentary, no extra keys.

OUTPUT SCHEMA:
{
    "slots_filled": ["SLOT_NAME"],
    "slots_missing": ["SLOT_NAME"],
    "unnecessary_calls": ["tool_name"],
    "harmful_calls": ["tool_name"]
}
    """
).strip()


def build_slot_judge_user_prompt(
    task_prompt: str,
    required_slots: list[str],
    slot_definitions: dict[str, str],
    plan: list[dict[str, object]],
) -> str:
    """Build the user prompt for the slot judge LLM."""
    return dedent(
        f"""
        Task prompt:
        {task_prompt}

        Required slots:
        {required_slots}

        Slot definitions:
        {slot_definitions}

        Proposed plan:
        {plan}
        """
    ).strip()