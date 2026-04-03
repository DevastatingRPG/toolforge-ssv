"""
ToolForge Environment — Server-Side Implementation.

Implements the OpenEnv Environment interface for the ToolForge DevOps
benchmark. The environment simulates a DevOps workflow where an LLM agent
must identify recurring tool-call patterns and compose them into macro tools
to reduce token consumption.

Episode lifecycle:
    reset() → initial observation
    step()  → (observation, reward from judge pipeline, done, metadata)
    state   → ToolForgeState property

Current implementation status:
    - step() accounting is fully deterministic (step_count, call_history, tokens_used).
    - Reward comes from the four-stage judge pipeline (placeholder stages).
    - Task progression, macro approval logic are still STUBBED.
"""

import logging
import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from toolforge_env.models import (
    ToolForgeAction,
    ToolForgeObservation,
    ToolForgeState,
)
from toolforge_env.server.judge_pipeline import run_judge_pipeline
from toolforge_env.server.tools import build_atomic_tools
from toolforge_env.server.tasks import build_easy_task_queue
from toolforge_env.server.user_sim import SimulatedUser

# ──────────────────────────────────────────────────────────────────────────────
# Module logger — used throughout this file for structured diagnostics.
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


class ToolForgeEnvironment(Environment):
    """
    ToolForge DevOps environment implementation for OpenEnv.

    This class is the central coordinator for a ToolForge episode. It manages:
        - Episode state (ToolForgeState)
        - Task queue progression (stubbed)
        - Macro tool accumulation (stubbed)
        - Action accounting (step_count, call_history, tokens_used)
        - Observation construction from state

    Inherits from openenv.core.env_server.interfaces.Environment which
    requires reset(), step(), and a state property.

    Attribute SUPPORTS_CONCURRENT_SESSIONS is True because each WebSocket
    connection receives its own ToolForgeEnvironment instance via the factory
    pattern in create_app(), and instances share no mutable state.
    """

    # Each WebSocket session gets its own ToolForgeEnvironment instance —
    # no shared mutable state exists between instances.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Hard guardrail to ensure episodes terminate deterministically even if the
    # agent keeps submitting malformed or low-quality plans.
    MAX_EPISODE_STEPS: int = 100

    def __init__(self) -> None:
        """
        Initialize the ToolForge environment with safe, deterministic defaults.

        Calls the parent Environment.__init__() to wire up transform/rubric
        support (both None until set), then seeds a dummy pre-reset state
        so that .state is never unbound before the first reset() call.
        """

        # ── Parent initializer ──────────────────────────────────────────────
        # Required: Environment base class stores self.transform and self.rubric.
        # We pass no transform or rubric at construction time; they can be
        # injected later via the server layer if needed.
        super().__init__(transform=None, rubric=None)

        # ── Difficulty level ────────────────────────────────────────────────
        # Controls which task queue is loaded during reset.
        # Currently only "easy" tasks are defined.
        self._difficulty: str = "easy"

        # ── Simulated user ──────────────────────────────────────────────────
        # The SimulatedUser approves/rejects plans and sends task prompts.
        # Instantiated here so it is available immediately after __init__.
        self._user: SimulatedUser = SimulatedUser(difficulty=self._difficulty)

        # ── Pre-reset dummy state ───────────────────────────────────────────
        # Provides a valid ToolForgeState before reset() is called.
        # Marked done=True so any premature step() call is visibly wrong.
        dummy_task = build_easy_task_queue()[0]
        self._state: ToolForgeState = ToolForgeState(
            episode_id="",
            step_count=0,
            current_task=dummy_task,
            task_queue=[],
            completed_tasks=[],
            available_tools=[],
            accepted_macros=[],
            rejected_macro_count=0,
            call_history=[],
            tokens_used=0,
            last_approval=None,
            done=True,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # *** reset ***
    # Starts a new episode. Builds a fresh ToolForgeState from the task
    # queue and available atomic tools, then returns the initial observation.
    #
    # Accepts task_id to pin which task starts the episode (for deterministic
    # benchmark runs). seed and episode_id follow the standard OpenEnv reset
    # signature.
    # ──────────────────────────────────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolForgeObservation:
        """
        Reset the environment for a new episode.

        Builds a fresh ToolForgeState with the specified (or default) task
        queue. Resets step count, call history, token accumulator, and macro
        registry. Returns the initial observation.

        Args:
            seed:       Random seed (reserved for future stochastic tasks).
            episode_id: Optional external episode identifier for tracking;
                        a UUID is generated when not provided.
            task_id:    Optional ID of the task to start the episode with.
                        When provided, that task is placed first in the queue.
            **kwargs:   Accepted for OpenEnv forward-compatibility.

        Returns:
            ToolForgeObservation reflecting the initial environment state.
        """

        logger.info(
            "Resetting ToolForgeEnvironment (episode_id=%s, task_id=%s)",
            episode_id,
            task_id,
        )

        # ── Episode ID ──────────────────────────────────────────────────────
        # Use caller-supplied ID, or generate a fresh UUID.
        ep_id: str = episode_id if episode_id else str(uuid.uuid4())

        # ── Tool and task setup ─────────────────────────────────────────────
        # build_atomic_tools() is deterministic; it returns the full set of
        # atomic DevOps tools available at the start of every episode.
        tools = build_atomic_tools()

        # build_easy_task_queue() returns a list; pop the first task as the
        # active task, leaving the rest in the queue.
        full_queue = build_easy_task_queue(task_id=task_id)
        first_task = full_queue.pop(0)
        initial_prompt = self._user.generate_task_prompt(first_task)

        # ── State construction ──────────────────────────────────────────────
        self._state = ToolForgeState(
            episode_id=ep_id,
            step_count=0,
            current_task=first_task,
            task_queue=full_queue,
            completed_tasks=[],
            available_tools=tools,
            accepted_macros=[],
            rejected_macro_count=0,
            call_history=[],
            tokens_used=0,
            last_approval=None,
            done=False,
        )

        obs_initial = self._get_observation()
        obs_initial.metadata = {
            "summary": "Episode initialized.",
            "user_prompt": initial_prompt,
            "user_approved": None,
            "progression": "episode_started",
        }
        return obs_initial

    def step(
        self,
        action: ToolForgeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ToolForgeObservation:  # type: ignore[override]
        """Take a single step in the environment by executing an action plan.

        Performs deterministic accounting (step_count, call_history,
        tokens_used), gates plans through simulated user approval, runs
        the judge pipeline only for approved plans, and returns an
        observation. Macro acceptance is not yet implemented.

                Phase-2 progression and gating rules:
                        - Plans rejected by the simulated user do not run through the
                            judge pipeline and must retry the same task.
                        - For user-approved plans, structural validation controls
                            advancement to the next queued task.

        Args:
            action:    The agent's proposed plan (ToolForgeAction).
            timeout_s: Execution timeout (reserved; unused).
            **kwargs:  Accepted for OpenEnv forward-compatibility.

        Returns:
            ToolForgeObservation reflecting state after the action.

        Raises:
            ValueError: If action is not a ToolForgeAction instance.
        """
        if not isinstance(action, ToolForgeAction):
            raise ValueError(
                f"Expected ToolForgeAction, got {type(action).__name__}"
            )

        # If a caller keeps stepping after terminal state, return a stable
        # terminal observation instead of mutating state.
        if self._is_done():
            obs_terminal: ToolForgeObservation = self._get_observation()
            obs_terminal.done = True
            obs_terminal.reward = 0.0
            obs_terminal.metadata = {
                "summary": "Episode already completed.",
                "terminal": True,
            }
            return obs_terminal

        # 1. Increment step counter
        self._state.step_count += 1

        # 2. Simulated user reviews the proposed plan before any pipeline run.
        user_approved = self._user.evaluate_plan(action.plan, self._state.current_task)
        self._state.last_approval = user_approved

        # 3. Record tool calls and accumulate token cost
        for call in action.plan:
            self._state.call_history.append(call)
            self._state.tokens_used += call.token_cost

        # 4. If the simulated user rejects the plan, skip judge pipeline and
        # keep the current task active for retry.
        if not user_approved:
            progression = "retry_current_task_user_rejected"
            if self._state.step_count >= self.MAX_EPISODE_STEPS:
                self._state.done = True
                progression = "episode_terminated_max_steps"

            obs_rejected: ToolForgeObservation = self._get_observation()
            obs_rejected.reward = 0.0
            obs_rejected.done = self._is_done()
            obs_rejected.metadata = {
                "stub": True,
                "summary": "User rejected plan. Retry current task.",
                "passed_validation": None,
                "user_approved": False,
                "progression": progression,
            }

            logger.debug(
                "step() | step_count=%d | tokens_used=%d | user_approved=%s | %s",
                self._state.step_count,
                self._state.tokens_used,
                user_approved,
                obs_rejected.metadata["summary"],
            )
            return obs_rejected

        # 5. Run the judge pipeline (validator → slot judge → reward → token cost)
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

        # 6. Lifecycle control for approved plans.
        progression = "retry_current_task_validation_failed"
        if self._state.step_count >= self.MAX_EPISODE_STEPS:
            self._state.done = True
            progression = "episode_terminated_max_steps"
        elif pipeline_result.passed_validation:
            advanced = self._advance_to_next_task()
            progression = "advanced_to_next_task" if advanced else "episode_completed"

        # 7. Build observation from updated state
        obs: ToolForgeObservation = self._get_observation()

        # 8. Set reward from pipeline and attach lifecycle metadata
        obs.reward = pipeline_result.final_score
        obs.done = self._is_done()
        obs.metadata = {
            "stub": True,
            "summary": pipeline_result.summary,
            "passed_validation": pipeline_result.passed_validation,
            "user_approved": True,
            "progression": progression,
        }

        logger.debug(
            "step() | step_count=%d | tokens_used=%d | score=%.3f | %s",
            self._state.step_count,
            self._state.tokens_used,
            pipeline_result.final_score,
            pipeline_result.summary,
        )

        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # *** _advance_to_next_task ***
    # Marks current task complete and loads the next queued task, if available.
    # Returns True when a new task is loaded, False when the episode is complete.
    # ──────────────────────────────────────────────────────────────────────────
    def _advance_to_next_task(self) -> bool:
        """
        Advance from current task to the next queued task.

        Returns:
            bool: True if next task was loaded, False if queue is exhausted.
        """

        completed_task_id = self._state.current_task.id
        self._state.completed_tasks.append(self._state.current_task)

        if len(self._state.task_queue) == 0:
            self._state.done = True
            logger.info(
                "Episode complete. Final task '%s' finished; no tasks remain.",
                completed_task_id,
            )
            return False

        next_task = self._state.task_queue.pop(0)
        self._state.current_task = next_task
        next_prompt = self._user.generate_task_prompt(next_task)
        logger.info(
            "Task advanced from '%s' to '%s'. Remaining tasks=%d",
            completed_task_id,
            next_task.id,
            len(self._state.task_queue),
        )
        logger.debug("Next task prompt: %s", next_prompt)
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # *** _get_observation ***
    # Internal helper — constructs a ToolForgeObservation from the current
    # ToolForgeState. Called by both reset() and step().
    # ──────────────────────────────────────────────────────────────────────────
    def _get_observation(self) -> ToolForgeObservation:
        """
        Build a ToolForgeObservation snapshot from the current internal state.

        This is the single point of truth for observation construction.
        Both reset() and step() call this helper to ensure consistency.

        Returns:
            ToolForgeObservation populated from self._state.
        """

        return ToolForgeObservation(
            # The task the agent must currently solve
            current_task=self._state.current_task,

            # Full set of tools available to the agent (atomic + accepted macros)
            available_tools=self._state.available_tools,

            # Running log of all tool calls made so far in the episode
            call_history=self._state.call_history,

            # Cumulative token cost incurred across all steps
            tokens_used=self._state.tokens_used,

            # Last simulated-user approval for the previous plan submission
            last_approval=self._state.last_approval,

            # How many tasks remain after the current one
            tasks_remaining=len(self._state.task_queue),

            # Macros approved by the simulated user so far
            accepted_macros=self._state.accepted_macros,

            # done and reward are set by the caller (reset or step)
            done=self._is_done(),
            reward=0.0,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # *** _is_done ***
    # Thin predicate that reads the done flag from ToolForgeState.
    # Will later incorporate task-queue exhaustion and timeout checks.
    # ──────────────────────────────────────────────────────────────────────────
    def _is_done(self) -> bool:
        """
        Return True when the current episode should terminate.

        Episode is considered done when either:
            - The explicit state.done flag is set, or
            - The max-step guardrail is reached.

        Returns:
            bool: True if the episode is over, False otherwise.
        """

        return bool(self._state.done) or self._state.step_count >= self.MAX_EPISODE_STEPS

    # ──────────────────────────────────────────────────────────────────────────
    # *** state (property) ***
    # Required abstract property from Environment. Returns the current
    # ToolForgeState so the server can serialise it for GET /state.
    # ──────────────────────────────────────────────────────────────────────────
    @property
    def state(self) -> ToolForgeState:
        """
        Return the current internal ToolForgeState.

        Used by the OpenEnv HTTP server to respond to state() requests.
        Callers should treat this as a read-only snapshot.

        Returns:
            ToolForgeState: The current episode state.
        """

        return self._state

    # ──────────────────────────────────────────────────────────────────────────
    # *** get_metadata ***
    # Returns structured EnvironmentMetadata (the OpenEnv-standard type),
    # rather than a plain dict, to match the base class return annotation:
    #   interfaces.Environment.get_metadata() -> EnvironmentMetadata
    # ──────────────────────────────────────────────────────────────────────────
    def get_metadata(self) -> EnvironmentMetadata:
        """
        Return descriptive metadata about this environment.

        Overrides the base class default to provide ToolForge-specific
        name, description, and version. Returns the OpenEnv-standard
        EnvironmentMetadata Pydantic model (not a plain dict) so that
        the HTTP server can serialise it correctly via /metadata.

        Returns:
            EnvironmentMetadata with environment name, description, version.
        """

        return EnvironmentMetadata(
            name="toolforge_env",
            description=(
                "A DevOps benchmark where an LLM agent learns to identify "
                "recurring tool-call patterns and compose them into reusable "
                "macro tools to minimise token consumption."
            ),
            version="0.1.0",
        )
