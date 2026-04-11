# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional
try:
    from ...models import (
        ToolForgeState,
        Task,
    )
    from ..inputs.base import InputProvider
except ImportError:
    from models import (
        ToolForgeState,
        Task,
    )
    from server.inputs.base import InputProvider


logger = logging.getLogger(__name__)

def get_next_task_from_generator(input_provider: Optional[InputProvider]) -> Task:
    """Return next task from generator supporting get() or get_input()."""
    print("Prompt change")
    if input_provider is None:
        raise RuntimeError("Task data generator not initialized. Call reset() first.")

    return input_provider.get_input()

def advance_to_next_task(state: ToolForgeState, input_provider: Optional[InputProvider]) -> bool:
    """Advance from current task to the next queued task."""

    completed_task_id = state.current_task.id

    if input_provider is None or input_provider.is_done():
        state.done = True
        logger.info(
            "Episode complete. Final task '%s' finished; no tasks remain.",
            completed_task_id,
        )
        return False

    next_task = get_next_task_from_generator(input_provider)
    state.current_task = next_task
    logger.info(
        "Task advanced from '%s' to '%s'.",
        completed_task_id,
        next_task.id,
    )
    logger.debug("Next task prompt: %s", next_task.prompt)
    return True
