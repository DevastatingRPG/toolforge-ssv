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
        Tool,
        ToolCall,
        ToolforgeAction,
        ToolforgeObservation,
        ToolForgeState,
        Task,
    )
except ImportError:
    from models import (
        Tool,
        ToolCall,
        ToolforgeAction,
        ToolforgeObservation,
        ToolForgeState,
        Task,
    )

try:
    from server.tools import build_atomic_tools
    from server.evaluation_pipeline import run_evaluation_pipeline
    from server.plan_evaluator import update_sequence_counts
except ImportError:
    from .tools import build_atomic_tools
    from .evaluation_pipeline import run_evaluation_pipeline
    from .plan_evaluator import update_sequence_counts


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
        >>> obs = env.step(ToolforgeAction(message="Hello"))
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
        super().__init__(transform=None, rubric=None)
        self._state = self._create_default_state()
        self._reset_count = 0

        self._task_selector = TaskSelector()
        self._input_provider_factory = create_input_provider
        self._input_provider: Optional[InputProvider] = None
        self._last_approval: Optional[bool] = None

        # persistent config
        self.mode = None
        self.difficulty: str = "easy"
        self.initialized = False

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
            required_slots=[],
            baseline_token_cost=0,
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
            available_tools=self._available_tools_to_prompt_specs(self._state.available_tools),
        )

    def _tool_to_prompt_spec(self, tool: Tool) -> Dict[str, Any]:
        """Convert a Tool model into a plain prompt-friendly dictionary."""

        return {
            "name": tool.name,
            "description": tool.description,
            "is_macro": tool.is_macro,
            "steps": tool.steps or [],
        }

    def _available_tools_to_prompt_specs(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert available tools to plain dictionaries ready for prompt injection.
        Macros are listed first, followed by atomic tools.
        """
        sorted_tools = sorted(tools, key=lambda t: not t.is_macro)
        return [self._tool_to_prompt_spec(tool) for tool in sorted_tools]

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
        ep_id = episode_id if episode_id is not None else str(uuid4())

        # old state preservation
        prev_accepted_macros = self._state.accepted_macros if hasattr(self, '_state') else []
        prev_macro_defs = self._state.macro_definitions if hasattr(self, '_state') else {}
        prev_rejected_count = self._state.rejected_macro_count if hasattr(self, '_state') else 0

        self._state = self._create_default_state()
        self._state.episode_id = ep_id
        self._reset_count += 1
        resolved_task_id = task_id or kwargs.get("task_id", "easy")

        # subsequent resets ignore incoming values
        task_list = self._task_selector.next_task_list(resolved_task_id)

        self._input_provider = self._input_provider_factory(task_list)
        first_task = self._get_next_task_from_generator()
        self._state.current_task = first_task
        self._sync_task_queue_from_generator()
        self._state.completed_tasks = []
        self._state.available_tools = build_atomic_tools() + prev_accepted_macros
        self._state.accepted_macros = prev_accepted_macros
        self._state.rejected_macro_count = prev_rejected_count
        self._state.call_history = []
        self._state.tokens_used = 0
        self._state.done = False
        # Reset episode-level macro tracking state
        self._state.sequence_counts = {}
        self._state.macro_usage_counts = {}
        self._state.macro_definitions = prev_macro_defs
        self._last_approval = None


        obs = self._create_default_observation()
        obs.done = False
        obs.reward = 0.0

        return obs

    def step(self, action: ToolforgeAction) -> ToolforgeObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: ToolforgeAction containing the message to echo

        Returns:
            ToolforgeObservation with the echoed message and its length
        """
        if self._is_done():
            obs_terminal = self._create_default_observation()
            obs_terminal.done = True
            obs_terminal.reward = 0.0
            obs_terminal.metadata = {
                "summary": "Episode already completed.",
                "terminal": True,
            }
            return obs_terminal

        if not isinstance(action, ToolforgeAction):
            logger.warning(
                "Malformed action rejected. Expected ToolforgeAction, got %s",
                type(action).__name__,
            )
            self._state.step_count += 1
            self._last_approval = False

            progression = "advanced_to_next_task_malformed_action"
            if self._state.step_count >= self.MAX_EPISODE_STEPS:
                self._state.done = True
                progression = "episode_terminated_max_steps"
            else:
                advanced = self._advance_to_next_task()
                progression = "advanced_to_next_task" if advanced else "episode_completed"

            obs_malformed = self._create_default_observation()
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

        plan_accounting = self._analyze_plan(action.plan)
        step_call_count = plan_accounting["step_call_count"]
        unknown_tool_calls = plan_accounting["unknown_tool_calls"]

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
        self._last_approval = bool(pipeline_result.passed_validation)

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
            advanced = self._advance_to_next_task()
            progression = "advanced_to_next_task" if advanced else "episode_completed"

        macro_result = self._process_macro_proposal(
            action=action,
            can_accept=bool(pipeline_result.passed_validation),
            reject_reason="plan_not_accepted",
        )

        if macro_result["decision"] == "approved" and action.macro_proposal is not None:
            self._state.available_tools.append(action.macro_proposal)

        # Fetch the scalar reward from the evaluation pipeline
        reward = float(pipeline_result.reward)
        print(self._state.current_task.prompt, reward, progression, macro_result)
        return ToolforgeObservation(
            current_task=self._state.current_task,
            available_tools=self._available_tools_to_prompt_specs(self._state.available_tools),
            done=self._is_done(),
            reward=reward
        )

    def _advance_to_next_task(self) -> bool:
        """Advance from current task to the next queued task."""

        completed_task_id = self._state.current_task.id
        self._state.completed_tasks.append(self._state.current_task)

        if self._input_provider is None or self._input_provider.is_done():
            self._state.done = True
            self._state.task_queue = []
            logger.info(
                "Episode complete. Final task '%s' finished; no tasks remain.",
                completed_task_id,
            )
            return False

        next_task = self._get_next_task_from_generator()
        self._state.current_task = next_task
        print("Prompt change")
        self._sync_task_queue_from_generator()
        logger.info(
            "Task advanced from '%s' to '%s'. Remaining tasks=%d",
            completed_task_id,
            next_task.id,
            len(self._state.task_queue),
        )
        logger.debug("Next task prompt: %s", next_task.prompt)
        return True

    def _analyze_plan(self, plan: List[ToolCall]) -> Dict[str, Any]:
        """Compute deterministic call accounting from server-side tool registry."""

        available_tools_by_name = {
            tool.name: tool for tool in self._state.available_tools
        }
        unknown_tool_calls: List[str] = []

        for call in plan:
            tool_def = available_tools_by_name.get(call.tool_name)
            if tool_def is None:
                unknown_tool_calls.append(call.tool_name)

        return {
            "step_call_count": len(plan),
            "unknown_tool_calls": unknown_tool_calls,
        }

    def _process_macro_proposal(
        self,
        action: ToolforgeAction,
        can_accept: bool,
        reject_reason: str,
    ) -> Dict[str, Any]:
        """Evaluate and apply macro proposal lifecycle for this step."""

        result: Dict[str, Any] = {
            "attempted": False,
            "decision": "none",
            "name": None,
            "reason": "no_macro_proposal",
        }

        has_macro_intent = (
            action.action_type == "propose_plan_with_macro"
            or action.macro_proposal is not None
        )
        if not has_macro_intent:
            return result

        result["attempted"] = True

        if action.action_type != "propose_plan_with_macro":
            return self._reject_macro(
                result=result,
                name=action.macro_proposal.name if action.macro_proposal else None,
                reason="macro_proposal_requires_propose_plan_with_macro_action_type",
            )

        if action.macro_proposal is None:
            return self._reject_macro(
                result=result,
                name=None,
                reason="missing_macro_proposal_payload",
            )

        proposal = action.macro_proposal
        macro_name = proposal.name.strip()

        if not can_accept:
            return self._reject_macro(
                result=result,
                name=macro_name,
                reason=reject_reason,
            )

        if not macro_name:
            return self._reject_macro(
                result=result,
                name=None,
                reason="macro_name_cannot_be_empty",
            )

        if not proposal.steps:
            return self._reject_macro(
                result=result,
                name=None,
                reason="macro_steps_cannot_be_empty",
            )


        existing_names = {tool.name for tool in self._state.available_tools}
        if macro_name in existing_names:
            return self._reject_macro(
                result=result,
                name=macro_name,
                reason="macro_name_already_exists",
            )

        if len(proposal.steps) < 2:
            return self._reject_macro(
                result=result,
                name=macro_name,
                reason="macro_requires_at_least_two_steps",
            )

        available_tools_by_name = {
            tool.name: tool for tool in self._state.available_tools
        }

        missing_steps = [
            call.tool_name
            for call in proposal.steps
            if call.tool_name not in available_tools_by_name
        ]
        if missing_steps:
            return self._reject_macro(
                result=result,
                name=macro_name,
                reason=f"macro_contains_unknown_tools:{','.join(missing_steps)}",
            )

        if any(
            available_tools_by_name[call.tool_name].is_macro
            for call in proposal.steps
        ):
            return self._reject_macro(
                result=result,
                name=macro_name,
                reason="nested_macro_steps_not_supported",
            )

        composed_of: List[str] = [call.tool_name for call in proposal.steps]

        macro_tool = Tool(
            name=macro_name,
            description=proposal.description.strip() or f"Macro: {' -> '.join(composed_of)}",
            is_macro=True,
            steps=proposal.steps,
        )

        self._state.accepted_macros.append(macro_tool)
        self._state.available_tools.append(macro_tool)

        # Store macro definition for sequence-based recognition tracking
        self._state.macro_definitions[macro_name] = composed_of

        result["decision"] = "approved"
        result["name"] = macro_name
        result["reason"] = "macro_registered"
        logger.info(
            "Macro approved: name='%s', steps=%s",
            macro_name,
            composed_of,
        )
        return result

    def _reject_macro(
        self,
        result: Dict[str, Any],
        name: Optional[str],
        reason: str,
    ) -> Dict[str, Any]:
        """Record macro rejection and return standardized metadata payload."""

        self._state.rejected_macro_count += 1
        result["decision"] = "rejected"
        result["name"] = name
        result["reason"] = reason
        logger.info("Macro rejected: name='%s', reason='%s'", name, reason)
        return result

    def _get_next_task_from_generator(self) -> Task:
        """Return next task from generator supporting get() or get_input()."""
        print("Prompt change")
        if self._input_provider is None:
            raise RuntimeError("Task data generator not initialized. Call reset() first.")

        return self._input_provider.get_input()

    def _sync_task_queue_from_generator(self) -> None:
        """Best-effort sync of remaining tasks for state visibility."""

        if self._input_provider is None:
            self._state.task_queue = []
            return

        if hasattr(self._input_provider, "data") and hasattr(self._input_provider, "idx"):
            data = getattr(self._input_provider, "data")
            idx = getattr(self._input_provider, "idx")
            self._state.task_queue = list(data[idx:])
            return

        # For generic providers without index access, keep queue opaque.
        self._state.task_queue = []

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
