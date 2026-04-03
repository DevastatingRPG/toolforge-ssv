import logging
from typing import Literal, List
from toolforge_env.models import Task, ToolCall

logger = logging.getLogger(__name__)

class SimulatedUser:
    """
    Simulates a DevOps user interacting with the agent.
    Validates plans proposed by the agent against the task's expected steps.
    """
    
    def __init__(self, difficulty: Literal["easy", "medium", "hard"]):
        """
        Initializes the simulated user with a given difficulty mode.
        Currently, only 'easy' is supported.
        """
        self.difficulty = difficulty

    def generate_task_prompt(self, task: Task) -> str:
        """
        Fetches the prompt text for the task to give to the agent.
        """
        if self.difficulty != "easy":
            raise NotImplementedError(f"Task generation for difficulty '{self.difficulty}' is not yet implemented.")
            
        return task.prompt

    def evaluate_plan(self, plan: List[ToolCall], task: Task) -> bool:
        """
        Evaluates an agent's proposed plan against the task's expected steps.
        For 'easy' mode, this is an exact deterministic match of tool names.
        """
        if self.difficulty != "easy":
            raise NotImplementedError(f"Plan evaluation for difficulty '{self.difficulty}' is not yet implemented.")
            
        proposed_steps = [call.tool_name for call in plan]
        is_exact_match = proposed_steps == task.required_steps
        
        if not is_exact_match:
            logger.debug(f"Plan evaluation failed. Expected: {task.required_steps}, got: {proposed_steps}")
            
        return is_exact_match
