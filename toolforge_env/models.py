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
    Contains the prompt, difficulty, expected steps, semantic slots,
    and baseline token cost for grading.
    """
    # Unique identifier for the task
    id: str

    # The user-facing prompt describing the task
    prompt: str

    # The difficulty level of the task
    difficulty: Literal["easy", "medium", "hard"]

    # Semantic slot names the judge checks against (e.g. deployment_execution)
    required_slots: List[str]

    # The baseline number of tool calls minimally needed for a correct atomic solution
    baseline_call_count: int



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


class PlanAccuracyResult(BaseModel):
    """Output of the Stage-3 plan accuracy calculator."""
    # Fraction of required slots successfully filled
    slot_completion_ratio: float
    # Score generated from completion curve (<= 0)
    slot_score: float
    # Penalty magnitude for unnecessary steps (<= 0)
    unnecessary_penalty: float
    # Final Stage 3 score bounded [-1.0, 0.0]
    score: float
    # Named sub-scores for debugging/logging
    breakdown: Dict[str, float]


class ToolEvaluation(BaseModel):
    """
    Per-tool-call evaluation produced by the Stage-2 semantic slot judge.
    """
    # Index of this tool call within the submitted plan
    tool_call_index: int

    # Name of the tool that was called
    tool_name: str

    # Which semantic slot this call fills, if any
    fills_slot: Optional[str]

    # Classification decided by the simulated LLM judge
    classification: Literal["relevant", "unnecessary", "harmful"]

    # Short explanation of the judgment
    reason: str


class SlotJudgmentResult(BaseModel):
    """
    Aggregate output of the Stage-2 semantic slot judge.
    Contains per-call evaluations plus slot-level summary.
    """
    # One ToolEvaluation per tool call in the plan
    evaluations: List[ToolEvaluation]

    # Slot names that were successfully filled
    slots_filled: List[str]

    # Slot names that remain unfulfilled
    slots_missing: List[str]

    # Whether all required slots were filled
    task_complete: bool

    # Flag set if any tool call was classified as harmful
    harmful_calls_present: bool


class TokenCostResult(BaseModel):
    """Output of the Stage-4 token-cost calculator."""
    # Actual tokens consumed by the plan
    tokens_used: int
    # Naive baseline cost for comparison
    baseline_tokens: int
    # tokens_used / baseline_tokens (lower is better)
    efficiency_ratio: float
    # Normalised efficiency score (0.0–1.0, higher is better)
    efficiency_score: float
    # Tokens saved through macro reuse
    macro_savings: int
    # Bonus points applied when a macro is used efficiently
    macro_bonus: float

class ValidationResult(BaseModel):
    """
    Output of the Stage-1 algorithmic validator.
    Indicates whether a proposed plan is structurally valid.
    """
    # Whether the plan passed all structural checks
    valid: bool

    # Machine-readable reason code:
    # VALID | EMPTY_PLAN | INVALID_TOOL | MISSING_PARAM | EXTRA_PARAM
    reason: str

    # Reward penalty to apply (0.0 for valid, negative otherwise)
    penalty: float

    # Optional human-readable detail (e.g. which tool/param failed)
    detail: Optional[str] = None


class PipelineResult(BaseModel):
    """Aggregate output of the full judge pipeline."""
    # Stage-1 validation result (always present)
    validation: ValidationResult
    # Stage-2 slot judgment (None when validation fails)
    slot_judgment: Optional[SlotJudgmentResult] = None
    # Stage-3 plan accuracy (None when validation fails or harmful)
    plan_accuracy: Optional[PlanAccuracyResult] = None
    # Stage-4 token cost (None when validation fails or harmful)
    token_cost: Optional[TokenCostResult] = None
    # Final blended score clamped to [-1.0, 1.0]
    reward: float
    # Whether the plan passed structural validation
    passed_validation: bool
    # Human-readable one-line summary
    summary: str




