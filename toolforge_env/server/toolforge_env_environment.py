# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Toolforge Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from typing import Optional

try:
    from ..models import ToolforgeAction, ToolforgeObservation, ToolForgeState, Task
except ImportError:
    from models import ToolforgeAction, ToolforgeObservation, ToolForgeState, Task


class ToolforgeEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = ToolforgeEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Toolforge Env environment ready!"
        >>>
        >>> obs = env.step(ToolforgeAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, input_provider=None):
        """Initialize the toolforge_env environment."""
        self._state = self._create_default_state()
        self._reset_count = 0
        self._input_provider = input_provider

    def _create_default_state(self) -> ToolForgeState:
        """
        Create a default ToolForgeState with basic parameters.

        Returns:
            ToolForgeState with default values
        """
        default_task = Task(
            id="default-task",
            prompt="Default task",
            difficulty="easy",
            required_steps=[],
            core_steps=[],
        )

        return ToolForgeState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task=default_task,
            task_queue=[],
            completed_tasks=[],
            available_tools=[],
            accepted_macros=[],
            rejected_macro_count=0,
            call_history=[],
            tokens_used=0,
            done=False,
        )

    def _create_default_observation(self) -> ToolforgeObservation:
        """
        Create a default ToolforgeObservation with basic parameters.

        Returns:
            ToolforgeObservation with default values
        """
        return ToolforgeObservation(
            current_task=self._state.current_task,
            available_tools=[]
        )

    def reset(        
            self, 
            seed: Optional[int] = None, 
            episode_id: Optional[str] = None, 
            task_id: Optional[str] = None, 
            **kwargs
        ) -> ToolforgeObservation:
        """
        Reset the environment.

        Returns:
            ToolforgeObservation with a ready message
        """
        # Use provided episode_id or generate a new one
        ep_id = episode_id if episode_id is not None else str(uuid4())
        self._state = self._create_default_state()
        self._state.episode_id = ep_id
        self._reset_count += 1

        return self._create_default_observation()

    def step(self, action: ToolforgeAction) -> ToolforgeObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: ToolforgeAction containing the message to echo

        Returns:
            ToolforgeObservation with the echoed message and its length
        """
        self._state.step_count += 1

        

        # Simple reward: longer messages get higher rewards
        reward = 1

        return ToolforgeObservation(
            current_task=self._state.current_task,
            available_tools=self._state.available_tools,
            done=False,
            reward=reward,
            metadata={"step": self._state.step_count},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
