"""
Rubrics for the ToolForge environment.

Follows the OpenEnv Rubric system (RFC 004) to provide composable,
outcome-based rewards suitable for RL training (GRPO, etc.).

The key insight from DSPy GRPO and Daytona RL guides: the RLM is a pure
inference engine. Reward computation is external — it compares the final
answer against ground truth. The environment provides the reward via rubrics;
the training framework consumes it.
"""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics.base import Rubric

FINAL_REWARD_MIN = -0.2
FINAL_REWARD_MAX = 1.0
VALIDATION_PENALTY = -0.2

SLOT_THRESHOLD = 0.65
SLOT_SCORE_MIN = -0.15
SLOT_SCORE_MAX = 0.25

MACRO_CREATION_MAX = 0.20
MACRO_CREATION_DECAY_FLOOR = 0.05
MACRO_CREATION_THRESHOLD = 2
MACRO_CREATION_FULL_RANGE = {2, 3}
MACRO_USAGE_PARTIAL = 0.03
MACRO_USAGE_FULL = 0.05
MACRO_MISS_PENALTY_MIN = 0.02
MACRO_MISS_PENALTY_MAX = 0.05
MACRO_MISS_PENALTY_CAP = 0.10

EFFICIENCY_SCORE_BASELINE = 0.2
EFFICIENCY_SCORE_MIN = 0.0
EFFICIENCY_SCORE_MAX = 0.5
EFFICIENCY_SCALE = 0.3


def compute_slot_score(slot_ratio: float) -> float:
    """Piecewise linear slot score bounded to [-0.15, 0.25]."""
    if slot_ratio < SLOT_THRESHOLD:
        if SLOT_THRESHOLD == 0:
            return SLOT_SCORE_MIN
        return SLOT_SCORE_MIN * (1.0 - slot_ratio / SLOT_THRESHOLD)
    if SLOT_THRESHOLD == 1.0:
        return SLOT_SCORE_MAX
    return SLOT_SCORE_MAX * ((slot_ratio - SLOT_THRESHOLD) / (1.0 - SLOT_THRESHOLD))


def compute_macro_creation_bonus(slot_ratio: float, prior_count: int | None) -> float:
    """Bonus for creating a macro after a repeated sequence has been observed."""
    if slot_ratio < SLOT_THRESHOLD or prior_count is None:
        return 0.0
    if prior_count < MACRO_CREATION_THRESHOLD:
        return 0.0
    if prior_count in MACRO_CREATION_FULL_RANGE:
        return MACRO_CREATION_MAX
    return max(
        MACRO_CREATION_DECAY_FLOOR,
        MACRO_CREATION_MAX * (max(MACRO_CREATION_FULL_RANGE) / prior_count),
    )


def compute_macro_usage_bonus(slot_ratio: float, macro_used: bool) -> float:
    """Bonus for using a macro when the plan is sufficiently correct."""
    if slot_ratio < SLOT_THRESHOLD or not macro_used:
        return 0.0
    if slot_ratio == 1.0:
        return MACRO_USAGE_FULL
    return MACRO_USAGE_PARTIAL


def compute_macro_miss_penalty(slot_ratio: float, miss_count: int) -> float:
    """Penalty for expanding an available macro into atomic calls."""
    if slot_ratio < SLOT_THRESHOLD or miss_count <= 0:
        return 0.0

    base_penalty = MACRO_MISS_PENALTY_MIN + (
        (MACRO_MISS_PENALTY_MAX - MACRO_MISS_PENALTY_MIN)
        * ((slot_ratio - SLOT_THRESHOLD) / (1.0 - SLOT_THRESHOLD))
    )
    total_penalty = sum(base_penalty + idx * 0.01 for idx in range(miss_count))
    return -min(MACRO_MISS_PENALTY_CAP, total_penalty)


def compute_efficiency_score(slot_ratio: float, baseline_calls: int | None, actual_calls: int | None) -> float:
    """Reward efficient plans, but only when all slots are filled."""
    if slot_ratio < 1.0:
        return 0.0
    if baseline_calls is None or actual_calls is None or baseline_calls <= 0:
        return EFFICIENCY_SCORE_BASELINE

    efficiency_ratio = (baseline_calls - actual_calls) / baseline_calls
    score = EFFICIENCY_SCORE_BASELINE + EFFICIENCY_SCALE * efficiency_ratio
    return max(EFFICIENCY_SCORE_MIN, min(EFFICIENCY_SCORE_MAX, score))


def compute_toolforge_reward_breakdown(metadata: dict[str, Any]) -> dict[str, float]:
    """Compute the ToolForge reward and return all component contributions."""
    validation_result = metadata.get("validation_result") or {}
    if validation_result and not validation_result.get("valid", True):
        return {
            "slot_score": 0.0,
            "macro_creation": 0.0,
            "macro_usage": 0.0,
            "macro_miss_penalty": 0.0,
            "efficiency_score": 0.0,
            "final_reward": VALIDATION_PENALTY,
        }

    if metadata.get("judge_failed", False):
        return {
            "slot_score": 0.0,
            "macro_creation": 0.0,
            "macro_usage": 0.0,
            "macro_miss_penalty": 0.0,
            "efficiency_score": 0.0,
            "final_reward": 0.0,
        }

    if metadata.get("harmful_calls_present", False):
        return {
            "slot_score": 0.0,
            "macro_creation": 0.0,
            "macro_usage": 0.0,
            "macro_miss_penalty": 0.0,
            "efficiency_score": 0.0,
            "final_reward": FINAL_REWARD_MIN,
        }

    slot_ratio = float(metadata.get("slot_ratio", 0.0))
    macro_prior_count = metadata.get("macro_prior_count")
    macro_used = bool(metadata.get("macro_used", False))
    macro_miss_count = int(metadata.get("macro_miss_count", 0))
    baseline_calls = metadata.get("baseline_calls")
    actual_calls = metadata.get("actual_calls")

    slot_score = compute_slot_score(slot_ratio)
    macro_creation = compute_macro_creation_bonus(slot_ratio, macro_prior_count)
    macro_usage = compute_macro_usage_bonus(slot_ratio, macro_used)
    macro_miss_penalty = compute_macro_miss_penalty(slot_ratio, macro_miss_count)
    efficiency_score = compute_efficiency_score(slot_ratio, baseline_calls, actual_calls)

    if slot_ratio < SLOT_THRESHOLD:
        final_raw = slot_score
        efficiency_score = 0.0
    elif slot_ratio < 1.0:
        final_raw = slot_score + macro_creation + macro_usage + macro_miss_penalty
        efficiency_score = 0.0
    else:
        final_raw = (
            slot_score
            + macro_creation
            + macro_usage
            + macro_miss_penalty
            + efficiency_score
        )

    final_reward = max(FINAL_REWARD_MIN, min(FINAL_REWARD_MAX, final_raw))
    return {
        "slot_score": slot_score,
        "macro_creation": macro_creation,
        "macro_usage": macro_usage,
        "macro_miss_penalty": macro_miss_penalty,
        "efficiency_score": efficiency_score,
        "final_reward": final_reward,
    }


class SlotRatioRubric(Rubric):
    """Process rubric: reward based on slot filling ratio.

    Provides a per-step signal to encourage plans that fill more slots.
    Returns the current slot ratio (0.0 to 1.0) as the reward.
    """

    def forward(self, action: Any, observation: Any) -> float:
        metadata = getattr(observation, "metadata", {})
        return float(metadata.get("slot_ratio", 0.0))


class PlanValidationRubric(Rubric):
    """Process rubric: penalty for invalid plans."""

    def __init__(self, penalty: float = -0.2) -> None:
        super().__init__()
        self.penalty = penalty

    def forward(self, action: Any, observation: Any) -> float:
        validation_result = getattr(observation, "metadata", {}).get("validation_result")
        if validation_result and not validation_result.get("valid", True):
            return self.penalty
        return 0.0


class SlotScoreRubric(Rubric):
    """Process rubric: piecewise linear score based on slot ratio."""

    def __init__(
        self,
        threshold: float = 0.65,
        min_score: float = -0.15,
        max_score: float = 0.25,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.min_score = min_score
        self.max_score = max_score

    def forward(self, action: Any, observation: Any) -> float:
        slot_ratio = float(getattr(observation, "metadata", {}).get("slot_ratio", 0.0))
        if slot_ratio < self.threshold:
            if self.threshold == 0:
                return self.min_score
            return self.min_score * (1.0 - slot_ratio / self.threshold)
        else:
            if self.threshold == 1.0:
                return self.max_score
            return self.max_score * ((slot_ratio - self.threshold) / (1.0 - self.threshold))


class MacroCreationRubric(Rubric):
    """Process rubric: bonus for creating useful macros."""

    def __init__(
        self,
        threshold: float = 0.65,
        max_bonus: float = 0.20,
        decay_floor: float = 0.05,
        creation_threshold: int = 2,
        full_range: set[int] | None = None,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.max_bonus = max_bonus
        self.decay_floor = decay_floor
        self.creation_threshold = creation_threshold
        self.full_range = full_range or {2, 3}

    def forward(self, action: Any, observation: Any) -> float:
        slot_ratio = float(getattr(observation, "metadata", {}).get("slot_ratio", 0.0))
        if slot_ratio < self.threshold:
            return 0.0

        prior_count = getattr(observation, "metadata", {}).get("macro_prior_count")
        if prior_count is None:
            return 0.0

        if prior_count < self.creation_threshold:
            return 0.0
        if prior_count in self.full_range:
            return self.max_bonus
        return max(self.decay_floor, self.max_bonus * (max(self.full_range) / prior_count))


class MacroUsageRubric(Rubric):
    """Process rubric: bonus for using existing macros."""

    def __init__(
        self,
        threshold: float = 0.65,
        partial_bonus: float = 0.03,
        full_bonus: float = 0.05,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.partial_bonus = partial_bonus
        self.full_bonus = full_bonus

    def forward(self, action: Any, observation: Any) -> float:
        slot_ratio = float(getattr(observation, "metadata", {}).get("slot_ratio", 0.0))
        if slot_ratio < self.threshold:
            return 0.0

        macro_used = getattr(observation, "metadata", {}).get("macro_used", False)
        if not macro_used:
            return 0.0

        if slot_ratio == 1.0:
            return self.full_bonus
        return self.partial_bonus


class EfficiencyRubric(Rubric):
    """Process rubric: reward for efficient plans (at 100% slot fill)."""

    def __init__(
        self,
        baseline_reward: float = 0.2,
        min_reward: float = 0.0,
        max_reward: float = 0.5,
        scale: float = 0.3,
    ) -> None:
        super().__init__()
        self.baseline_reward = baseline_reward
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.scale = scale

    def forward(self, action: Any, observation: Any) -> float:
        slot_ratio = float(getattr(observation, "metadata", {}).get("slot_ratio", 0.0))
        if slot_ratio < 1.0:
            return 0.0

        baseline_calls = getattr(observation, "metadata", {}).get("baseline_calls")
        actual_calls = getattr(observation, "metadata", {}).get("actual_calls")

        if baseline_calls is None or actual_calls is None or baseline_calls <= 0:
            return self.baseline_reward

        efficiency_ratio = (baseline_calls - actual_calls) / baseline_calls
        score = self.baseline_reward + self.scale * efficiency_ratio
        return max(self.min_reward, min(self.max_reward, score))


class MacroMissPenaltyRubric(Rubric):
    """Process rubric: penalty when an available macro was not used."""

    def __init__(
        self,
        threshold: float = SLOT_THRESHOLD,
        min_penalty: float = MACRO_MISS_PENALTY_MIN,
        max_penalty: float = MACRO_MISS_PENALTY_MAX,
        cap: float = MACRO_MISS_PENALTY_CAP,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.cap = cap

    def forward(self, action: Any, observation: Any) -> float:
        metadata = getattr(observation, "metadata", {})
        slot_ratio = float(metadata.get("slot_ratio", 0.0))
        miss_count = int(metadata.get("macro_miss_count", 0))
        if slot_ratio < self.threshold or miss_count <= 0:
            return 0.0

        base_penalty = self.min_penalty + (
            (self.max_penalty - self.min_penalty)
            * ((slot_ratio - self.threshold) / (1.0 - self.threshold))
        )
        total_penalty = sum(base_penalty + idx * 0.01 for idx in range(miss_count))
        return -min(self.cap, total_penalty)


class ToolforgeRubric(Rubric):
    """Top-level rubric that composes the ToolForge step reward."""

    def __init__(self) -> None:
        super().__init__()
        self.validation = PlanValidationRubric(penalty=VALIDATION_PENALTY)
        self.slot_score = SlotScoreRubric(
            threshold=SLOT_THRESHOLD,
            min_score=SLOT_SCORE_MIN,
            max_score=SLOT_SCORE_MAX,
        )
        self.macro_creation = MacroCreationRubric(
            threshold=SLOT_THRESHOLD,
            max_bonus=MACRO_CREATION_MAX,
            decay_floor=MACRO_CREATION_DECAY_FLOOR,
            creation_threshold=MACRO_CREATION_THRESHOLD,
            full_range=MACRO_CREATION_FULL_RANGE,
        )
        self.macro_usage = MacroUsageRubric(
            threshold=SLOT_THRESHOLD,
            partial_bonus=MACRO_USAGE_PARTIAL,
            full_bonus=MACRO_USAGE_FULL,
        )
        self.macro_miss_penalty = MacroMissPenaltyRubric()
        self.efficiency = EfficiencyRubric(
            baseline_reward=EFFICIENCY_SCORE_BASELINE,
            min_reward=EFFICIENCY_SCORE_MIN,
            max_reward=EFFICIENCY_SCORE_MAX,
            scale=EFFICIENCY_SCALE,
        )

    def forward(self, action: Any, observation: Any) -> float:
        metadata = getattr(observation, "metadata", {})
        breakdown = compute_toolforge_reward_breakdown(metadata)
        return float(breakdown["final_reward"])


