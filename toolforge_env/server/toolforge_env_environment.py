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
from inputs.base import InputProvider
from typing import Any, Dict, List, Optional

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
    from . import tasks as task_catalog
    from .tools import build_atomic_tools
    from .judge_pipeline import run_judge_pipeline
    from .inputs.simulated.data_loader import SimulatedDataLoader
except ImportError:
    import tasks as task_catalog
    from tools import build_atomic_tools
    from judge_pipeline import run_judge_pipeline
    from inputs.simulated.data_loader import SimulatedDataLoader


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

    # Deterministic fallback cost for unknown tool calls.
    UNKNOWN_TOOL_CALL_TOKEN_COST: int = 20

    def __init__(self, input_provider: Optional[InputProvider] = None):
        """Initialize the toolforge_env environment."""
        super().__init__(transform=None, rubric=None)
        self._state = self._create_default_state()
        self._reset_count = 0
        self._input_provider = input_provider
        self._data_generator: Optional[InputProvider] = None
        self._last_approval: Optional[bool] = None

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
            available_tools=self._state.available_tools,
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

        # Initialize task-level data generator for this episode and fetch first task.
        difficulty = str(kwargs.get("difficulty") or kwargs.get("task_level") or "easy").lower()
        full_queue = self._build_task_queue(task_id=task_id, difficulty=difficulty)
        self._data_generator = SimulatedDataLoader(full_queue)
        first_task = self._get_next_task_from_generator()
        self._state.current_task = first_task
        self._sync_task_queue_from_generator()
        self._state.completed_tasks = []
        self._state.available_tools = build_atomic_tools()
        self._state.accepted_macros = []
        self._state.rejected_macro_count = 0
        self._state.call_history = []
        self._state.tokens_used = 0
        self._state.done = False
        self._last_approval = None

        obs = self._create_default_observation()
        obs.done = False
        obs.reward = 0.0
        obs.metadata = {
            "summary": "Episode initialized.",
            "task_prompt": first_task.prompt,
            "task_level": first_task.difficulty,
            "progression": "episode_started",
        }

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

        token_accounting = self._compute_plan_token_cost(action.plan)
        step_token_cost = token_accounting["step_token_cost"]
        unknown_tool_calls = token_accounting["unknown_tool_calls"]

        self._state.call_history.extend(action.plan)
        self._state.tokens_used += step_token_cost

        available_tools_by_name = {
            tool.name: tool for tool in self._state.available_tools
        }
        pipeline_result = run_judge_pipeline(
            plan=action.plan,
            task=self._state.current_task,
            available_tools=available_tools_by_name,
            accepted_macros=self._state.accepted_macros,
            baseline_token_cost=self._state.current_task.baseline_token_cost,
        )
        self._last_approval = bool(pipeline_result.passed_validation)

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

        

        # Simple reward: longer messages get higher rewards
        reward = float(pipeline_result.final_score)

        return ToolforgeObservation(
            current_task=self._state.current_task,
            available_tools=self._state.available_tools,
            done=self._is_done(),
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "summary": pipeline_result.summary,
                "passed_validation": bool(pipeline_result.passed_validation),
                "plan_accepted": bool(pipeline_result.passed_validation),
                "task_prompt": self._state.current_task.prompt,
                "task_level": self._state.current_task.difficulty,
                "progression": progression,
                "macro_attempted": macro_result["attempted"],
                "macro_decision": macro_result["decision"],
                "macro_name": macro_result["name"],
                "macro_reason": macro_result["reason"],
                "accepted_macro_count": len(self._state.accepted_macros),
                "step_token_cost": step_token_cost,
                "token_accounting_source": "tool_registry",
                "unknown_tool_calls": unknown_tool_calls,
                "last_approval": self._last_approval,
            },
        )

    def _advance_to_next_task(self) -> bool:
        """Advance from current task to the next queued task."""

        completed_task_id = self._state.current_task.id
        self._state.completed_tasks.append(self._state.current_task)

        if self._data_generator is None or self._data_generator.is_done():
            self._state.done = True
            self._state.task_queue = []
            logger.info(
                "Episode complete. Final task '%s' finished; no tasks remain.",
                completed_task_id,
            )
            return False

        next_task = self._get_next_task_from_generator()
        self._state.current_task = next_task
        self._sync_task_queue_from_generator()
        logger.info(
            "Task advanced from '%s' to '%s'. Remaining tasks=%d",
            completed_task_id,
            next_task.id,
            len(self._state.task_queue),
        )
        logger.debug("Next task prompt: %s", next_task.prompt)
        return True

    def _compute_plan_token_cost(self, plan: List[ToolCall]) -> Dict[str, Any]:
        """Compute deterministic token accounting from server-side tool registry."""

        available_tools_by_name = {
            tool.name: tool for tool in self._state.available_tools
        }
        unknown_tool_calls: List[str] = []
        step_token_cost = 0

        for call in plan:
            tool_def = available_tools_by_name.get(call.tool_name)
            if tool_def is None:
                step_token_cost += self.UNKNOWN_TOOL_CALL_TOKEN_COST
                unknown_tool_calls.append(call.tool_name)
                continue

            step_token_cost += tool_def.token_cost

        return {
            "step_token_cost": step_token_cost,
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
        atomic_cost_total = sum(
            available_tools_by_name[tool_name].token_cost
            for tool_name in composed_of
        )

        macro_token_cost = max(1, atomic_cost_total - 2)

        macro_tool = Tool(
            name=macro_name,
            description=proposal.description.strip() or f"Macro: {' -> '.join(composed_of)}",
            params_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            is_macro=True,
            token_cost=macro_token_cost,
            composed_of=composed_of,
        )

        self._state.accepted_macros.append(macro_tool)
        self._state.available_tools.append(macro_tool)

        result["decision"] = "approved"
        result["name"] = macro_name
        result["reason"] = f"macro_registered_token_cost:{macro_token_cost}"
        logger.info(
            "Macro approved: name='%s', steps=%s, token_cost=%d",
            macro_name,
            composed_of,
            macro_token_cost,
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

        if self._data_generator is None:
            raise RuntimeError("Task data generator not initialized. Call reset() first.")

        getter = getattr(self._data_generator, "get", None)
        if callable(getter):
            return getter()

        return self._data_generator.get_input()

    def _sync_task_queue_from_generator(self) -> None:
        """Best-effort sync of remaining tasks for state visibility."""

        if self._data_generator is None:
            self._state.task_queue = []
            return

        if hasattr(self._data_generator, "data") and hasattr(self._data_generator, "idx"):
            data = getattr(self._data_generator, "data")
            idx = getattr(self._data_generator, "idx")
            self._state.task_queue = list(data[idx:])
            return

        # For generic providers without index access, keep queue opaque.
        self._state.task_queue = []

    def _build_task_queue(self, task_id: Optional[str], difficulty: str) -> List[Task]:
        """Build a task queue for the requested difficulty level."""

        builder_name = f"build_{difficulty}_task_queue"
        builder = getattr(task_catalog, builder_name, None)
        if not callable(builder):
            available_levels = sorted(
                name.replace("build_", "").replace("_task_queue", "")
                for name in dir(task_catalog)
                if name.startswith("build_") and name.endswith("_task_queue")
            )
            raise ValueError(
                f"Unsupported task difficulty '{difficulty}'. "
                f"Available levels: {available_levels}"
            )

        return builder(task_id=task_id)

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
