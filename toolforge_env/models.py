# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Toolforge Env Environment.

The toolforge_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation, State
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field, BaseModel

class ToolCall(BaseModel):
    """
    Represents a single invocation of a tool.
    It encapsulates the tool's identity, the arguments provided, and the cost in tokens.
    """
    # Name of the tool being called
    tool_name: str
    
    # Parameters passed to the tool
    params: Dict[str, Any]
    
    # Token cost of calling this tool
    token_cost: int


class Tool(BaseModel):
    """
    Represents an available tool in the environment.
    This can be an atomic tool or a composed macro tool.
    """
    # Identifier name of the tool
    name: str
    
    # Human-readable description of what the tool does
    description: str
    
    # Schema defining the expected parameters
    params_schema: Dict[str, Any]
    
    # Flag indicating whether this tool is a macro (composed of smaller tools)
    is_macro: bool = False
    
    # The base token cost to use this tool
    token_cost: int
    
    # Optional list of tool names this macro is composed of
    composed_of: Optional[List[str]] = None


class Task(BaseModel):
    """
    Represents a DevOps task that the agent needs to accomplish.
    Contains the prompt, difficulty, and expected steps for completion.
    """
    # Unique identifier for the task
    id: str
    
    # The user-facing prompt describing the task
    prompt: str
    
    # The difficulty level of the task
    difficulty: Literal["easy", "medium", "hard"]
    
    # The exact list of steps required to complete the task
    required_steps: List[str]
    
    # The core, essential steps identifying the task's primary goal
    core_steps: List[str]


class MacroProposal(BaseModel):
    """
    Represents a proposal made by the agent to create a new macro tool
    from a sequence of existing tool calls.
    """
    # Proposed name for the new macro
    name: str
    
    # Description of what the proposed macro accomplishes
    description: str
    
    # List of sequential tool calls that make up the macro
    steps: List[ToolCall]


class ToolforgeAction(Action):
    """Action for the Toolforge Env environment - just a message to echo."""

    # The type of action being performed
    action_type: Literal["propose_plan", "propose_plan_with_macro"] = Field(
        ..., description="The type of action being performed"
    )

    # The execution plan consisting of sequential tool calls
    plan: List[ToolCall] = Field(
        ..., description="The execution plan consisting of sequential tool calls"
    )

    # Optional proposal for a new macro, if action_type is "propose_plan_with_macro"
    macro_proposal: Optional[MacroProposal] = Field(
        None,
        description="Optional proposal for a new macro, used when action_type is 'propose_plan_with_macro'"
    )

    # Agent's reasoning for this action
    reasoning: str = Field(
        ..., description="Agent's reasoning for this action"
    )


class ToolforgeObservation(Observation):
    """Observation from the Toolforge Env environment - the echoed message."""

    # The active task the agent must complete
    current_task: Task = Field(
        ..., description="The active task the agent must complete"
    )

    # List of tools currently available to the agent
    available_tools: List[Tool] = Field(
        ..., description="List of tools currently available to the agent"
    )


class ToolForgeState(State):
    """
    The internal State class for the ToolForge environment.
    Keeps track of all task queues, completed metrics, and session statistics.
    Subclasses the core OpenEnv State type.
    """
    # The task currently being worked on
    current_task: Task
    
    # Queue of upcoming tasks
    task_queue: List[Task]
    
    # List of tasks successfully completed
    completed_tasks: List[Task]
    
    # All currently available tools
    available_tools: List[Tool]
    
    # Successfully created macros
    accepted_macros: List[Tool]
    
    # Count of how many proposed macros were rejected
    rejected_macro_count: int
    
    # Full history of tool calls in the session
    call_history: List[ToolCall]
    
    # Total accumulated token cost
    tokens_used: int
    
    # Flag indicating if the environment episode has concluded
    done: bool

