import logging
import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment

from toolforge_env.models import (
    ToolForgeAction,
    ToolForgeObservation,
    ToolForgeState,
)
from toolforge_env.server.tools import build_atomic_tools
from toolforge_env.server.tasks import build_easy_task_queue
from toolforge_env.server.user_sim import SimulatedUser

logger = logging.getLogger(__name__)

class ToolForgeEnvironment(Environment):
    """
    ToolForge DevOps environment implementation for OpenEnv.
    """
    
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """
        Initialize the ToolForge environment with safe, deterministic defaults.
        """
        self._difficulty = "easy"
        self._user = SimulatedUser(difficulty=self._difficulty)
        
        # Initialize an empty state before first reset
        # We need realistic dummy values until reset() is properly called
        dummy_task = build_easy_task_queue()[0]
        self._state = ToolForgeState(
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
            done=True
        )

    def reset(
        self, 
        seed: Optional[int] = None, 
        episode_id: Optional[str] = None, 
        task_id: Optional[str] = None, 
        **kwargs
    ) -> ToolForgeObservation:
        """
        Reset the environment for a new episode.
        """
        logger.info(f"Resetting ToolForgeEnvironment (episode_id={episode_id}, task_id={task_id})")
        
        ep_id = episode_id if episode_id else str(uuid.uuid4())
        
        tools = build_atomic_tools()
        full_queue = build_easy_task_queue(task_id=task_id)
        
        # Pop the first task to be the current one
        first_task = full_queue.pop(0)
        
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
            done=False
        )
        
        return self._get_observation()

    def step(self, action: ToolForgeAction, timeout_s: Optional[float] = None, **kwargs) -> ToolForgeObservation:  # type: ignore[override]
        """
        Takes a step in the environment by executing an action plan.
        Currently a stub implementing deterministic accounting.
        """
        if not isinstance(action, ToolForgeAction):
            raise ValueError(f"Expected ToolForgeAction, got {type(action)}")
            
        self._state.step_count += 1
        
        # Accounting for this action
        for call in action.plan:
            self._state.call_history.append(call)
            self._state.tokens_used += call.token_cost
            
        # Stub logic: 
        # do not advance tasks yet
        # do not evaluate approval yet
        # do not modify macros yet
        
        obs = self._get_observation()
        obs.reward = 0.0
        obs.done = self._is_done()
        obs.metadata = {"stub": True}
        
        return obs
        
    def _get_observation(self) -> ToolForgeObservation:
        """
        Builds the observation from the current state.
        """
        return ToolForgeObservation(
            current_task=self._state.current_task,
            available_tools=self._state.available_tools,
            call_history=self._state.call_history,
            tokens_used=self._state.tokens_used,
            last_approval=None,
            tasks_remaining=len(self._state.task_queue),
            accepted_macros=self._state.accepted_macros,
            done=self._is_done(),
            reward=0.0
        )

    def _is_done(self) -> bool:
        """
        Check if episode is done based on state.
        """
        return bool(self._state.done)

    @property
    def state(self) -> ToolForgeState:
        """
        Get the current environment state.
        """
        return self._state
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Provide environment descriptive metadata.
        """
        return {
            "name": "toolforge_env",
            "description": "A ToolForge standard DevOps reinforcement learning environment",
            "version": "0.1.0"
        }
