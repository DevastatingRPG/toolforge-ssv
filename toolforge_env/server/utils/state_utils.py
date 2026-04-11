# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from uuid import uuid4
try:
    from ...models import (
        ToolForgeState,
        Task,
        ToolForgeObservation,
    )
except ImportError:
    from models import (
        ToolForgeState,
        Task,
        ToolForgeObservation,
    )

from server.tools import build_atomic_tools

def create_default_state() -> ToolForgeState:
    """
    Create a default ToolForgeState with basic parameters.

    Returns:
        ToolForgeState with default values
    """
    default_task = Task(
        id="default-task",
        prompt="Default task",
        difficulty="easy",
        required_slots=[],
        baseline_token_cost=0,
    )

    return ToolForgeState(
        episode_id=str(uuid4()),
        step_count=0,
        current_task=default_task,
        available_tools=build_atomic_tools(),
        accepted_macros=[],
        rejected_macro_count=0,
        call_history=[],
        tokens_used=0,
        done=False,
    )

def create_default_observation(state: ToolForgeState, available_tools_to_prompt_specs) -> ToolForgeObservation:
    """
    Create a default ToolForgeObservation with basic parameters.

    Returns:
        ToolForgeObservation with default values
    """
    return ToolForgeObservation(
        current_task=state.current_task,
        available_tools=available_tools_to_prompt_specs(state.available_tools),
        grading = state.grading,
        done= False,
        reward = 0.0,
        metadata={},
    )
