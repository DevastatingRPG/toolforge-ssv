"""
Grader functions for the ToolForge environment.

Each grader receives the complete environment state object and returns
a benchmark score in [0.01, 0.99]. All three difficulty levels use the
same EpisodeGrader class — difficulty differences are reflected by the
episode data itself, not by separate grading logic.

Referenced by openenv.yaml as 'server.graders.grade_easy' etc.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EpisodeGrader:
    """Reads the episode grading state and computes a normalized [0.01, 0.99] score.

    Sub-scores:
        accuracy            — correct_plan_count / episode_steps
        token_optimization  — avg efficiency over fully-correct opportunities
        macro_creation      — macro_creation_correct / macro_creation_attempts
        macro_usage         — macro_usage_correct / macro_usage_attempts

    Weights (accuracy > token_opt > macro_creation > macro_usage):
        0.40 / 0.30 / 0.20 / 0.10

    Final score clamped to [0.01, 0.99].
    """

    WEIGHTS = {
        "accuracy": 0.40,
        "token_optimization": 0.30,
        "macro_creation": 0.20,
        "macro_usage": 0.10,
    }

    def grade(self, state: Any) -> float:
        """Compute normalized episode score from the environment state.

        Args:
            state: The ToolForgeState object for the completed episode.

        Returns:
            Score clamped to [0.01, 0.99].
        """
        # Extract the grading accumulator from state
        g = getattr(state, "grading", None)
        if g is None:
            logger.warning("EpisodeGrader: no grading state found, returning 0.01")
            return 0.01

        # --- Sub-score: accuracy ---
        if g.episode_steps > 0:
            accuracy = g.correct_plan_count / g.episode_steps
        else:
            accuracy = 0.0

        # --- Sub-score: token optimization ---
        if g.fully_correct_efficiency_opportunities > 0:
            token_opt = g.sum_efficiency_score / g.fully_correct_efficiency_opportunities
        else:
            token_opt = 0.0

        # --- Sub-score: macro creation ---
        if g.macro_creation_attempts > 0:
            macro_create = g.macro_creation_correct / g.macro_creation_attempts
        else:
            macro_create = 0.0

        # --- Sub-score: macro usage ---
        if g.macro_usage_attempts > 0:
            macro_use = g.macro_usage_correct / g.macro_usage_attempts
        else:
            macro_use = 0.0

        raw = (
            self.WEIGHTS["accuracy"] * accuracy
            + self.WEIGHTS["token_optimization"] * token_opt
            + self.WEIGHTS["macro_creation"] * macro_create
            + self.WEIGHTS["macro_usage"] * macro_use
        )

        final = max(0.01, min(0.99, raw))

        logger.info(
            "EpisodeGrader | accuracy=%.3f token_opt=%.3f macro_create=%.3f macro_use=%.3f | raw=%.4f final=%.4f",
            accuracy, token_opt, macro_create, macro_use, raw, final,
        )

        return final


# Shared grader instance
_grader = EpisodeGrader()


def grade_easy(*args, **kwargs) -> float:
    """Grade an easy-tier episode."""
    # The framework passes the state as the first positional argument
    state = args[0] if args else kwargs.get("state")
    return _grader.grade(state)


def grade_medium(*args, **kwargs) -> float:
    """Grade a medium-tier episode."""
    state = args[0] if args else kwargs.get("state")
    return _grader.grade(state)


def grade_hard(*args, **kwargs) -> float:
    """Grade a hard-tier episode."""
    state = args[0] if args else kwargs.get("state")
    return _grader.grade(state)
