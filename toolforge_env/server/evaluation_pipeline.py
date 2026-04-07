"""
Evaluation Pipeline Orchestrator

Thin orchestration layer that chains the plan evaluation logic.

Pipeline flow:
1. run_sanity_validation(...)
2. run_slot_judgment(...)
3. compute_step_reward(...)  — slot score + macro bonuses + efficiency

Short-circuits if validation fails, or if harmful calls are present.
Final reward is bounded to [-0.2, 1.0].
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from models import PipelineResult, Task, Tool, ToolCall
from server.plan_evaluator import (
    get_relevant_slots,
    run_sanity_validation,
    run_slot_judgment,
    compute_step_reward,
    VALIDATION_PENALTY,
    FINAL_REWARD_MIN,
)


logger = logging.getLogger(__name__)


def _ensure_reward_file_logger() -> None:
    """Attach a file handler once so pipeline logs are persisted under reward_logs/."""
    log_dir = Path(__file__).resolve().parents[1] / "reward_logs"
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
        # short-circuit immediately
        return PipelineResult(
            validation=validation,
            slot_judgment=None,
            plan_accuracy=None,
            token_cost=None,
            reward=VALIDATION_PENALTY,
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
    logger.info(
        "Stage2 slots | filled=%s | missing=%s | complete=%s | harmful=%s",
        slot_judgment.slots_filled,
        slot_judgment.slots_missing,
        slot_judgment.task_complete,
        slot_judgment.harmful_calls_present,
    )
    
    if slot_judgment.harmful_calls_present:
        logger.info("Pipeline end (harmful short-circuit) | reward=%.3f", FINAL_REWARD_MIN)
        # short-circuit immediately
        return PipelineResult(
            validation=validation,
            slot_judgment=slot_judgment,
            plan_accuracy=None,
            token_cost=None,
            reward=FINAL_REWARD_MIN,
            passed_validation=True,
            summary="Harmful tool call detected.",
        )

    # 3. Compute step reward (slot score + macro bonuses + efficiency)
    reward_breakdown = compute_step_reward(
        slot_judgment=slot_judgment,
        task=task,
        plan=plan,
        available_tools=available_tools,
        accepted_macros=accepted_macros,
        macro_proposal=macro_proposal,
        sequence_counts=sequence_counts,
    )

    slot_ratio = reward_breakdown["slot_ratio"]
    slot_score = reward_breakdown["slot_score"]
    macro_creation = reward_breakdown["macro_creation"]
    macro_usage = reward_breakdown["macro_usage"]
    efficiency_score = reward_breakdown["efficiency_score"]
    final_reward = reward_breakdown["final_reward"]

    logger.info(
        "Stage2 slot_score | slot_ratio=%.3f | slot_score=%.3f",
        slot_ratio,
        slot_score,
    )
    logger.info(
        "Stage3 macro_bonuses | macro_creation=%.3f | macro_usage=%.3f",
        macro_creation,
        macro_usage,
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
    )
