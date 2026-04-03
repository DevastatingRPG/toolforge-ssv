"""Stage-3 reward calculator (placeholder).

Computes a reward score from the validation and slot judgment results.
Currently returns hardcoded placeholder values. Real reward shaping
(partial credit, difficulty scaling, penalty curves) will replace the
placeholder logic in a future pass.

Public API:
    calculate_reward(validation_result, slot_judgment, task) -> RewardResult
"""

from toolforge_env.models import (
    RewardResult,
    SlotJudgmentResult,
    Task,
    ValidationResult,
)


def calculate_reward(
    validation_result: ValidationResult,
    slot_judgment: SlotJudgmentResult,
    task: Task,
) -> RewardResult:
    """Compute a reward score for a validated and slot-judged plan.

    Args:
        validation_result: Output of Stage-1 validator (expected valid here).
        slot_judgment:     Output of Stage-2 semantic slot judge.
        task:              The task definition being evaluated against.

    Returns:
        RewardResult with placeholder scores.
    """
    return RewardResult(
        correctness_score=1.0,
        slot_score=1.0,
        penalty=0.0,
        raw_score=1.0,
        breakdown={"placeholder": 1.0},
    )
