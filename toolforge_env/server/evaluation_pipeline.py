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

import logging
from datetime import datetime
from pathlib import Path
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
        logger.info("Pipeline end (validation short-circuit) | reward=%.3f", validation.penalty)
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
    logger.info(
        "Stage2 slots | filled=%s | missing=%s | complete=%s | harmful=%s",
        slot_judgment.slots_filled,
        slot_judgment.slots_missing,
        slot_judgment.task_complete,
        slot_judgment.harmful_calls_present,
    )
    
    if slot_judgment.harmful_calls_present:
        logger.info("Pipeline end (harmful short-circuit) | reward=-1.000")
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
    logger.info(
        "Stage3 plan_accuracy | ratio=%.3f | slot_score=%.3f | unnecessary_penalty=%.3f | score=%.3f",
        plan_accuracy.slot_completion_ratio,
        plan_accuracy.slot_score,
        plan_accuracy.unnecessary_penalty,
        plan_accuracy.score,
    )

    # 4. run_token_calculation(...)
    token_cost = None
    if slot_judgment.task_complete:
        token_cost = run_token_calculation(
            plan=plan,
            accepted_macros=accepted_macros,
            task=task,
            available_tools=available_tools,
            sequence_counts=sequence_counts,
            macro_definitions=macro_definitions,
            macro_proposal=macro_proposal,
        )
        logger.info(
            "Stage4 token | used=%d | baseline=%d | efficiency_ratio=%.3f | efficiency_score=%.3f | macro_bonus=%.3f",
            token_cost.tokens_used,
            token_cost.baseline_tokens,
            token_cost.efficiency_ratio,
            token_cost.efficiency_score,
            token_cost.macro_bonus,
        )
    else:
        logger.info("Stage4 token | skipped due to incomplete slots")

    # 5. reward_calculation(...)
    final_reward = reward_calculation(plan_accuracy, token_cost)
    logger.info("Stage5 reward | final_reward=%.3f", final_reward)

    return PipelineResult(
        validation=validation,
        slot_judgment=slot_judgment,
        plan_accuracy=plan_accuracy,
        token_cost=token_cost,
        reward=final_reward,
        passed_validation=True,
        summary="Plan validated, slots judged, final reward applied.",
    )
