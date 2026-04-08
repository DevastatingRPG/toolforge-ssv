"""
Grader functions for the ToolForge environment.

Each grader receives the episode trajectory and returns a score in [0, 1].
These are referenced by openenv.yaml as 'server.graders.grade_easy' etc.
"""

import random


def _base_grade(*args, **kwargs) -> float:
    """Placeholder grader that returns a random score.
    
    Will be replaced with real grading logic that inspects
    the episode trajectory once the external grader spec is finalised.
    """
    return random.uniform(0.01, 0.99)


def grade_easy(*args, **kwargs) -> float:
    """Grade an easy-tier episode."""
    return _base_grade(*args, **kwargs)


def grade_medium(*args, **kwargs) -> float:
    """Grade a medium-tier episode."""
    return _base_grade(*args, **kwargs)


def grade_hard(*args, **kwargs) -> float:
    """Grade a hard-tier episode."""
    return _base_grade(*args, **kwargs)
