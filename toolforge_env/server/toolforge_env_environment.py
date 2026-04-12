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

import logging
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State
from typing import Any, Dict, List, Optional

try:
    from .inputs.base import InputProvider
    from .inputs.simulated.task_selector import TaskSelector
    from .inputs.factory import create_input_provider
except ImportError:
    from server.inputs.base import InputProvider
    from server.inputs.simulated.task_selector import TaskSelector
    from server.inputs.factory import create_input_provider

try:
    from ..models import (
        ToolForgeAction,
        ToolForgeObservation,
        ToolForgeState,
        EpisodeGradingState,
    )
except ImportError:
    from models import (
        ToolForgeAction,
        ToolForgeObservation,
        ToolForgeState,
        EpisodeGradingState,
    )

try:
    from server.tools import build_atomic_tools
    from server.evaluation.pipeline import run_evaluation_pipeline
    from server.evaluation.plan_evaluator import update_sequence_counts
    from rubrics import ToolforgeRubric
    from .utils.state_utils import create_default_state, create_default_observation
    from .utils.tool_utils import available_tools_to_prompt_specs, analyze_plan
    from .utils.task_utils import get_next_task_from_generator, advance_to_next_task
    from .utils.macro_utils import process_macro_proposal
    from .utils.grading_utils import update_grading_state
except ImportError:
    from .tools import build_atomic_tools
    from .evaluation.pipeline import run_evaluation_pipeline
    from .evaluation.plan_evaluator import update_sequence_counts
    from ..rubrics import ToolforgeRubric
    from server.utils.state_utils import create_default_state, create_default_observation
    from server.utils.tool_utils import available_tools_to_prompt_specs, analyze_plan
    from server.utils.task_utils import get_next_task_from_generator, advance_to_next_task
    from server.utils.macro_utils import process_macro_proposal
    from server.utils.grading_utils import update_grading_state


logger = logging.getLogger(__name__)


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
        >>> obs = env.step(ToolForgeAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Hard guardrail to ensure episodes terminate deterministically.
    MAX_EPISODE_STEPS: int = 100

    def __init__(self):
        """Initialize the toolforge_env environment."""
        super().__init__(transform=None, rubric=ToolforgeRubric())
        self.rubric = ToolforgeRubric()
        self._state = create_default_state()
        self._reset_count = 0

        self._task_selector = TaskSelector()
        self._input_provider_factory = create_input_provider
        self._input_provider: Optional[InputProvider] = None

        # persistent config
        self.mode = None
        self.difficulty: str = "easy"
        self.initialized = False

    def reset(        
            self, 
            seed: Optional[int] = None, 
            episode_id: Optional[str] = None, 
            task_id: Optional[str] = None, 
            **kwargs
        ) -> ToolForgeObservation:
        """
        Reset the environment.

        Returns:
            ToolForgeObservation with a ready message
        """
        ep_id = episode_id if episode_id is not None else str(uuid4())

        self._state = create_default_state()
        self._state.episode_id = ep_id
        self._reset_count += 1
        resolved_task_id = task_id or kwargs.get("task_id", "easy")

        # subsequent resets ignore incoming values
        task_list = self._task_selector.next_task_list(resolved_task_id)

        self._input_provider = self._input_provider_factory(task_list)

        # ------------------------------------------------------------------
        # Capture episode length BEFORE consuming the first task.
        # This value is forwarded to the UI via observation metadata so the
        # Gradio demo tab can track progress independently of env.done.
        # (The stateless HTTP /reset + /step endpoints each create a fresh
        # env instance, so env.done is unreliable for navigation; we use
        # total_tasks as the authoritative episode-length signal instead.)
        # ------------------------------------------------------------------
        total_tasks: int = (
            self._input_provider.task_count()
            if self._input_provider is not None
            else 0
        )

        # Only fetch the first task when a real input provider exists.
        # If input_provider is None (e.g. future human-interaction mode where
        # tasks are supplied externally), we leave current_task as the
        # default placeholder and let the caller push tasks in via step().
        if self._input_provider is not None:
            first_task = get_next_task_from_generator(self._input_provider)
            self._state.current_task = first_task

        if hasattr(self.rubric, "reset"):
            self.rubric.reset()

        obs = create_default_observation(self._state, available_tools_to_prompt_specs)
        # Surface episode length so the UI can navigate without relying on env.done
        obs.metadata = {"total_tasks": total_tasks}
        return obs

    def step(self, action: ToolForgeAction) -> ToolForgeObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: ToolForgeAction containing the message to echo

        Returns:
            ToolForgeObservation with the echoed message and its length
        """
        if self._is_done():
            obs_terminal = create_default_observation(self._state, available_tools_to_prompt_specs)
            obs_terminal.done = True
            obs_terminal.reward = 0.0
            obs_terminal.metadata = {
                "summary": "Episode already completed.",
                "terminal": True,
            }
            return obs_terminal

        if not isinstance(action, ToolForgeAction):
            logger.warning(
                "Malformed action rejected. Expected ToolForgeAction, got %s",
                type(action).__name__,
            )
            self._state.step_count += 1

            # Track malformed actions in grading state
            self._state.grading.episode_steps += 1
            self._state.grading.validation_failures += 1

            progression = "advanced_to_next_task_malformed_action"
            if self._state.step_count >= self.MAX_EPISODE_STEPS:
                self._state.done = True
                progression = "episode_terminated_max_steps"
            else:
                advanced = advance_to_next_task(self._state, self._input_provider)
                progression = "advanced_to_next_task" if advanced else "episode_completed"

            obs_malformed = create_default_observation(self._state, available_tools_to_prompt_specs)
            obs_malformed.done = self._is_done()
            obs_malformed.reward = 0.0
            obs_malformed.metadata = {
                "summary": "Malformed action treated as failed attempt.",
                "malformed_action": True,
                "plan_accepted": False,
                "task_prompt": self._state.current_task.prompt,
                "task_level": self._state.current_task.difficulty,
                "progression": progression,
            }
            return obs_malformed

        self._state.step_count += 1

        plan_accounting = analyze_plan(action.plan, self._state.available_tools)
        step_call_count = plan_accounting["step_call_count"]
        
        self._state.call_history.extend(action.plan)
        self._state.tokens_used += step_call_count

        available_tools_by_name = {
            tool.name: tool for tool in self._state.available_tools
        }
        pipeline_result = run_evaluation_pipeline(
            plan=action.plan,
            task=self._state.current_task,
            available_tools=available_tools_by_name,
            accepted_macros=self._state.accepted_macros,
            baseline_token_cost=self._state.current_task.baseline_token_cost,
            sequence_counts=self._state.sequence_counts,
            macro_definitions=self._state.macro_definitions,
            macro_proposal=action.macro_proposal,
        )

        # Update sequence counts AFTER pipeline evaluation so recognition uses prior counts only
        update_sequence_counts(action.plan, self._state.sequence_counts)

        # Update macro usage counts for any macro tools used in this plan
        macro_names = {m.name for m in self._state.accepted_macros}
        for call in action.plan:
            if call.tool_name in macro_names:
                self._state.macro_usage_counts[call.tool_name] = (
                    self._state.macro_usage_counts.get(call.tool_name, 0) + 1
                )

        progression = "advanced_to_next_task"
        if self._state.step_count >= self.MAX_EPISODE_STEPS:
            self._state.done = True
            progression = "episode_terminated_max_steps"
        else:
            advanced = advance_to_next_task(self._state, self._input_provider)
            progression = "advanced_to_next_task" if advanced else "episode_completed"

        macro_result = process_macro_proposal(
            action=action,
            state=self._state,
            can_accept=bool(pipeline_result.passed_validation),
            reject_reason="plan_not_accepted",
        )

        # Note: the macro is added to self._state.available_tools inside _process_macro_proposal()

        observation_metadata = {
            "summary": pipeline_result.summary,
            "validation_result": pipeline_result.validation.model_dump(),
            "slot_ratio": pipeline_result.step_slot_ratio or 0.0,
            "task_complete": bool(pipeline_result.step_task_complete),
            "harmful_calls_present": bool(pipeline_result.step_harmful),
            "judge_failed": bool(
                pipeline_result.slot_judgment
                and getattr(pipeline_result.slot_judgment, "judge_failed", False)
            ),
            "macro_prior_count": pipeline_result.step_macro_prior_count,
            "macro_used": bool(pipeline_result.step_macro_used),
            "macro_miss_count": pipeline_result.step_macro_miss_count or 0,
            "baseline_calls": pipeline_result.step_baseline_calls,
            "actual_calls": pipeline_result.step_actual_calls or step_call_count,
            "progression": progression,
            "macro_result": macro_result,
        }

        observation = ToolForgeObservation(
            current_task=self._state.current_task,
            available_tools=available_tools_to_prompt_specs(self._state.available_tools),
            done=self._is_done(),
            reward=0.0,
            grading=self._state.grading,
            metadata=observation_metadata,
        )

        reward = float(self._apply_rubric(action, observation))
        observation.reward = reward

        # Accumulate grading state from this step
        update_grading_state(self._state, pipeline_result, macro_result, action)

        print(self._state.current_task.prompt, reward, progression, macro_result)
        print(f"Done: {self._state.done}{self._is_done()} | Step {self._state.step_count} | Task '{self._state.current_task.id}' | Reward: {reward:.3f} | Progression: {progression} | Macro Proposal: {macro_result}")
        observation.grading = self._state.grading
        return observation

    def _is_done(self) -> bool:
        """Return True when the current episode should terminate."""

        return bool(self._state.done) or self._state.step_count >= self.MAX_EPISODE_STEPS

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        """Return descriptive metadata about this environment."""

        return EnvironmentMetadata(
            name="toolforge_env",
            description=(
                "A DevOps benchmark where an LLM agent learns to identify "
                "recurring tool-call patterns and compose them into reusable "
                "macro tools to minimise token consumption."
            ),
            version="0.1.0",
        )
