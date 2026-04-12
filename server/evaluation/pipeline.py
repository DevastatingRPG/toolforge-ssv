"""
Evaluation Pipeline Orchestrator

Thin orchestration layer that chains the plan evaluation logic.

Pipeline flow:
1. run_sanity_validation(...)
2. run_slot_judgment(...)
3. derive reward inputs and delegate reward policy to rubrics.py

Short-circuits if validation fails, or if harmful calls are present.
Final reward is bounded to [-0.2, 1.0].
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from models import PipelineResult, Task, Tool, ToolCall
from .plan_evaluator import (
    calculate_dynamic_baseline_tokens,
    count_prior_sequence_occurrences,
    get_relevant_slots,
    run_sanity_validation,
    run_slot_judgment,
    VALIDATION_PENALTY,
    FINAL_REWARD_MIN,
)
from ..rubrics import compute_toolforge_reward_breakdown


logger = logging.getLogger(__name__)


def _count_macro_misses(plan: List[ToolCall], accepted_macros: List[Tool]) -> int:
    """Count atomic sequences in the plan that match accepted macros."""
    if not accepted_macros:
        return 0

    plan_tool_names = [call.tool_name for call in plan]
    miss_count = 0

    for macro in accepted_macros:
        if not macro.steps:
            continue

        macro_seq = [call.tool_name for call in macro.steps]
        seq_len = len(macro_seq)
        if seq_len < 2 or seq_len > len(plan_tool_names):
            continue

        i = 0
        while i <= len(plan_tool_names) - seq_len:
            if plan_tool_names[i:i + seq_len] == macro_seq:
                miss_count += 1
                i += seq_len
            else:
                i += 1

    return miss_count


def _ensure_reward_file_logger() -> None:
    """Attach a file handler once so pipeline logs are persisted under reward_logs/."""
    log_dir = Path(__file__).resolve().parents[2] / "reward_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Reuse any existing pipeline reward file handler.
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.get_name() == "pipeline_reward_file":
            return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_reward_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.set_name("pipeline_reward_file")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.propagate = True


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
    _ensure_reward_file_logger()

    logger.info(
        "Pipeline start | task_id=%s | required_slots=%s | plan_len=%d",
        task.id,
        task.required_slots,
        len(plan),
    )
    logger.info(
        "Plan tools=%s",
        [call.tool_name for call in plan],
    )
    
    # Temporarily add proposed macro to available tools for evaluation purposes.
    if macro_proposal:
        available_tools[macro_proposal.name] = macro_proposal

    # 1. run_sanity_validation(...)
    validation = run_sanity_validation(plan, available_tools)
    logger.info(
        "Stage1 validation | valid=%s | reason=%s | penalty=%.3f",
        validation.valid,
        validation.reason,
        validation.penalty,
    )

    if not validation.valid:
        logger.info("Pipeline end (validation short-circuit) | reward=%.3f", VALIDATION_PENALTY)
        return PipelineResult(
            validation=validation,
            slot_judgment=None,
            plan_accuracy=None,
            token_cost=None,
            reward=VALIDATION_PENALTY,
            passed_validation=False,
            summary=f"Validation failed: {validation.reason}",
            step_slot_ratio=0.0,
            step_task_complete=False,
            step_harmful=False,
            step_macro_prior_count=None,
            step_macro_used=False,
            step_macro_miss_count=0,
            step_baseline_calls=task.baseline_call_count,
            step_actual_calls=len(plan),
            step_macro_creation_bonus=0.0,
            step_macro_usage_bonus=0.0,
            step_macro_miss_penalty=0.0,
            step_efficiency_score=0.0,
        )

    # 2. run_slot_judgment(...)
    slot_judgment = run_slot_judgment(
        task_prompt=task.prompt,
        required_slots=task.required_slots,
        slot_definitions=get_relevant_slots(task.required_slots),
        available_tools=list(available_tools.values()),
        plan=plan,
    )
    logger.info(
        "Stage2 slots | filled=%s | missing=%s | complete=%s | harmful=%s",
        slot_judgment.slots_filled,
        slot_judgment.slots_missing,
        slot_judgment.task_complete,
        slot_judgment.harmful_calls_present,
    )
    
    if slot_judgment.harmful_calls_present:
        logger.info("Pipeline end (harmful short-circuit) | reward=%.3f", FINAL_REWARD_MIN)
        return PipelineResult(
            validation=validation,
            slot_judgment=slot_judgment,
            plan_accuracy=None,
            token_cost=None,
            reward=FINAL_REWARD_MIN,
            passed_validation=True,
            summary="Harmful tool call detected.",
            step_slot_ratio=0.0,
            step_task_complete=False,
            step_harmful=True,
            step_macro_prior_count=None,
            step_macro_used=False,
            step_macro_miss_count=0,
            step_baseline_calls=task.baseline_call_count,
            step_actual_calls=len(plan),
            step_macro_creation_bonus=0.0,
            step_macro_usage_bonus=0.0,
            step_macro_miss_penalty=0.0,
            step_efficiency_score=0.0,
        )

    if getattr(slot_judgment, "judge_failed", False):
        logger.error("Pipeline end (judge failed short-circuit) | reward=0.000")
        return PipelineResult(
            validation=validation,
            slot_judgment=slot_judgment,
            plan_accuracy=None,
            token_cost=None,
            reward=0.0,
            passed_validation=True,
            summary="Semantic judge unavailable / LLM failure",
            step_slot_ratio=0.0,
            step_task_complete=False,
            step_harmful=False,
            step_macro_prior_count=None,
            step_macro_used=False,
            step_macro_miss_count=0,
            step_baseline_calls=task.baseline_call_count,
            step_actual_calls=len(plan),
            step_macro_creation_bonus=0.0,
            step_macro_usage_bonus=0.0,
            step_macro_miss_penalty=0.0,
            step_efficiency_score=0.0,
        )

    n_required = len(task.required_slots)
    n_filled = len(slot_judgment.slots_filled)
    slot_ratio = n_filled / n_required if n_required > 0 else 1.0

    macro_prior_count = None
    if (
        macro_proposal is not None
        and sequence_counts is not None
        and macro_proposal.steps
        and len(macro_proposal.steps) >= 2
    ):
        proposed_sequence = tuple(call.tool_name for call in macro_proposal.steps)
        macro_prior_count = count_prior_sequence_occurrences(proposed_sequence, sequence_counts)

    macro_names = {macro.name for macro in accepted_macros}
    macro_used = any(call.tool_name in macro_names for call in plan)
    macro_miss_count = _count_macro_misses(plan, accepted_macros)
    baseline_calls = calculate_dynamic_baseline_tokens(task, available_tools)
    actual_calls = len(plan)

    reward_breakdown = compute_toolforge_reward_breakdown(
        {
            "validation_result": validation.model_dump(),
            "slot_ratio": slot_ratio,
            "harmful_calls_present": slot_judgment.harmful_calls_present,
            "judge_failed": bool(getattr(slot_judgment, "judge_failed", False)),
            "macro_prior_count": macro_prior_count,
            "macro_used": macro_used,
            "macro_miss_count": macro_miss_count,
            "baseline_calls": baseline_calls,
            "actual_calls": actual_calls,
        }
    )

    slot_score = reward_breakdown["slot_score"]
    macro_creation = reward_breakdown["macro_creation"]
    macro_usage = reward_breakdown["macro_usage"]
    macro_miss_penalty = reward_breakdown["macro_miss_penalty"]
    efficiency_score = reward_breakdown["efficiency_score"]
    final_reward = reward_breakdown["final_reward"]

    logger.info(
        "Stage2 slot_score | slot_ratio=%.3f | slot_score=%.3f",
        slot_ratio,
        slot_score,
    )
    logger.info(
        "Stage3 macro_bonuses | macro_creation=%.3f | macro_usage=%.3f | macro_miss_penalty=%.3f",
        macro_creation,
        macro_usage,
        macro_miss_penalty,
    )
    if slot_ratio >= 1.0:
        logger.info(
            "Stage4 efficiency | used=%d | baseline=%d | efficiency_score=%.3f",
            len(plan),
            task.baseline_call_count,
            efficiency_score,
        )
    else:
        logger.info("Stage4 efficiency | skipped (slot_ratio=%.3f < 1.0)", slot_ratio)

    logger.info("Stage5 final_reward | final_reward=%.3f", final_reward)

    return PipelineResult(
        validation=validation,
        slot_judgment=slot_judgment,
        plan_accuracy=None,
        token_cost=None,
        reward=final_reward,
        passed_validation=True,
        summary="Plan validated, slots judged, final reward applied.",
        step_slot_ratio=slot_ratio,
        step_task_complete=slot_judgment.task_complete,
        step_harmful=False,
        step_macro_prior_count=macro_prior_count,
        step_macro_used=macro_used,
        step_macro_miss_count=macro_miss_count,
        step_baseline_calls=baseline_calls,
        step_actual_calls=actual_calls,
        step_macro_creation_bonus=macro_creation,
        step_macro_usage_bonus=macro_usage,
        step_macro_miss_penalty=macro_miss_penalty,
        step_efficiency_score=efficiency_score,
    )
