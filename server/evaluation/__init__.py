"""
Evaluation package for the ToolForge environment.

Re-exports key functions so consumers can import from either:
    from server.evaluation import run_evaluation_pipeline, update_sequence_counts
    from server.evaluation.pipeline import run_evaluation_pipeline
    from server.evaluation.plan_evaluator import update_sequence_counts
"""

from server.evaluation.pipeline import run_evaluation_pipeline
from server.evaluation.plan_evaluator import (
    # Constants
    VALIDATION_PENALTY,
    FINAL_REWARD_MIN,
    FINAL_REWARD_MAX,
    SLOT_THRESHOLD,
    # Public APIs
    get_relevant_slots,
    run_sanity_validation,
    run_slot_judgment,
    compute_step_reward,
    update_sequence_counts,
    # LLM config (used by diagnose_llm.py)
    API_BASE_URL,
    MODEL_NAME,
    HF_TOKEN,
    # Internal helpers exposed for diagnostics
    _call_llm_slot_judgment,
)

__all__ = [
    "run_evaluation_pipeline",
    "get_relevant_slots",
    "run_sanity_validation",
    "run_slot_judgment",
    "compute_step_reward",
    "update_sequence_counts",
    "VALIDATION_PENALTY",
    "FINAL_REWARD_MIN",
    "FINAL_REWARD_MAX",
    "SLOT_THRESHOLD",
    "API_BASE_URL",
    "MODEL_NAME",
    "HF_TOKEN",
    "_call_llm_slot_judgment",
]
