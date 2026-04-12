# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Toolforge Env Environment.

The ToolForge environment is a Tool-Call optimizing environment which abstracts commonly used tool sequences into reusable macros.
"""

from openenv.core.env_server.types import Action, Observation, State
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field, BaseModel, ConfigDict, model_validator

# ==============================================================================
# SECTION 1: PRIMITIVES & DOMAIN MODELS
# The foundational building blocks for tasks, tools, and proposals.
# ==============================================================================

class ToolCall(BaseModel):
    """
    Represents a single invocation of a tool.
    It encapsulates the tool's identity and the arguments provided.
    """
    model_config = ConfigDict(extra="forbid")

    tool_name: str    
    """Name of the tool being called"""


class Tool(BaseModel):
    """
    Represents an executable tool within the environment.
    This can be either a standard atomic tool or a composed macro tool.
    """
    
    name: str 
    """The unique identifier name of the tool."""
    
    description: str
    """A clear, human-readable explanation of what the tool accomplishes."""
    
    is_macro: bool = False
    """A flag indicating whether this tool is a macro (composed of a sequence of smaller tool calls)."""
    
    steps: Optional[List['ToolCall']] = None
    """
    The sequential list of tool calls that make up the macro. 
    This must be populated if `is_macro` is True, and must be None if it is an atomic tool.
    """

    @model_validator(mode="after")
    def validate_macro_steps(self) -> "Tool":
        """Ensures the structural integrity of the tool by validating the steps."""
        if self.is_macro and not self.steps:
            raise ValueError(f"Macro '{self.name}' must have steps.")
        if not self.is_macro and self.steps:
            raise ValueError(f"Atomic tool '{self.name}' cannot have steps.")
        return self


class Task(BaseModel):
    """
    Represents a DevOps task that the agent needs to accomplish.
    Contains the prompt, difficulty, semantic slots, and baseline cost metadata.
    """

    id: str
    """Unique identifier for the task"""

    prompt: str
    """The user-facing prompt describing the task"""

    difficulty: Literal["easy", "medium", "hard"]
    """The difficulty level of the task"""

    required_slots: List[str]
    """Semantic slot names the judge checks against (e.g. DEPLOYMENT_ACTION)"""

    baseline_token_cost: int = 0
    """Naive token cost of executing the task's intended atomic sequence"""

    baseline_call_count: int = 0
    """Backward-compatible field used by task fixtures in this repository"""

    @model_validator(mode="before")
    @classmethod
    def _sync_baseline_fields(cls, data: Any) -> Any:
        """Allow either baseline_call_count or baseline_token_cost in task payloads."""
        if not isinstance(data, dict):
            return data

        has_token_cost = "baseline_token_cost" in data
        has_call_count = "baseline_call_count" in data

        if has_call_count and not has_token_cost:
            data["baseline_token_cost"] = data["baseline_call_count"]
        elif has_token_cost and not has_call_count:
            data["baseline_call_count"] = data["baseline_token_cost"]

        return data


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


# ==============================================================================
# SECTION 2: CORE ENVIRONMENT API
# Models defining the inputs, outputs, and internal state of the environment.
# ==============================================================================


class ToolForgeAction(Action):
    """Action for the Toolforge Env environment - Contains the plan needed to fulfill the prompt."""

    action_type: Literal["propose_plan", "propose_plan_with_macro"] 
    """The type of action being performed"""

    plan: List[ToolCall] 
    """The execution plan consisting of sequential tool calls"""

    macro_proposal: Optional[Tool] = None
    """Optional proposal for a new macro, used when action_type is 'propose_plan_with_macro'"""


class EpisodeGradingState(BaseModel):
    """Aggregate episode-level signals consumed by the grader.
    
    Accumulated incrementally during the episode by the environment.
    Read by the grader at episode end to compute a normalized score.
    """
    # Total steps taken in this episode
    episode_steps: int = 0
    # Steps where structural validation failed
    validation_failures: int = 0
    # Steps where harmful tool calls were detected
    harmful_plan_count: int = 0
    # Steps with full slot completion AND passed validation
    correct_plan_count: int = 0
    # Steps where efficiency was computed (slot_ratio == 1.0)
    fully_correct_efficiency_opportunities: int = 0
    # Sum of efficiency scores across fully-correct steps
    sum_efficiency_score: float = 0.0
    # Times the agent attempted to create a macro
    macro_creation_attempts: int = 0
    # Times macro creation was approved by the environment
    macro_creation_approved: int = 0
    # Approved AND plan was semantically valid
    macro_creation_correct: int = 0
    # Sum of macro creation bonuses awarded
    macro_creation_bonus_total: float = 0.0
    # Steps where any accepted macro was used in the plan
    macro_usage_attempts: int = 0
    # Macro used AND slot_ratio >= threshold
    macro_usage_correct: int = 0
    # Times macro creation was rejected
    macro_rejected_count: int = 0
    # Total accumulated macro miss penalty
    macro_miss_penalty_total: float = 0.0


class ToolForgeObservation(Observation):
    """Observation from the ToolForge Env environment."""

    current_task: Task 
    """The active task the agent must complete"""

    available_tools: List[Dict[str, Any]] 
    """List of tools currently available to the agent"""

    grading: Optional[EpisodeGradingState] = None
    """Episode-level grading signals accumulated during the episode"""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Observation metadata including total_tasks indicator"""

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to prevent metadata from being excluded by env serializers."""
        exclude = kwargs.get("exclude")
        if isinstance(exclude, set) and "metadata" in exclude:
            exclude = set(exclude)
            exclude.discard("metadata")
            kwargs["exclude"] = exclude
        return super().model_dump(**kwargs)


class ToolForgeState(State):
    """
    The internal State class for the ToolForge environment.
    Keeps track of all task queues, completed metrics, and session statistics.
    """
    current_task: Task
    """The task currently being worked on"""
    
    available_tools: List[Tool]
    """All currently available tools"""
    
    accepted_macros: List[Tool]
    """Successfully created macros"""
    
    rejected_macro_count: int
    """Count of how many proposed macros were rejected"""
    
    call_history: List[ToolCall]
    """Full history of tool calls in the session"""
    
    tokens_used: int
    """Total accumulated token cost"""
    
    done: bool
    """Flag indicating if the environment episode has concluded"""

    sequence_counts: Dict[str, int] = Field(default_factory=dict)
    """Exact ordered contiguous sequence counts observed earlier in the episode"""

    macro_usage_counts: Dict[str, int] = Field(default_factory=dict)
    """Number of times each macro tool has been used"""

    macro_definitions: Dict[str, List[str]] = Field(default_factory=dict)
    """Macro name -> ordered atomic tool names it represents"""

    grading: EpisodeGradingState = Field(default_factory=EpisodeGradingState)
    """Episode-level grading accumulator (reset each episode)"""


# ==============================================================================
# SECTION 3: EVALUATION & JUDGING PIPELINE
# Results outputted by the multi-stage grading system.
# ==============================================================================


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

    # Flag set if the live LLM judge failed after all retries
    judge_failed: bool = False


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
    # Bonus for recognizing a repeated sequence at or above threshold
    macro_recognition_bonus: float
    # Bonus for macro actually saving tokens vs atomic equivalent
    macro_utility_bonus: float
    # Combined macro bonus (recognition + utility)
    macro_bonus: float


class PipelineResult(BaseModel):
    """Aggregate output of the full judge pipeline."""
    # Stage-1 validation result (always present)
    validation: ValidationResult
    # Stage-2 slot judgment (None when validation fails)
    slot_judgment: Optional[SlotJudgmentResult] = None
    # Stage-3 plan accuracy (deprecated, kept for compatibility)
    plan_accuracy: Optional[PlanAccuracyResult] = None
    # Stage-4 token cost (deprecated, kept for compatibility)
    token_cost: Optional[TokenCostResult] = None
    # Final blended score clamped to [-0.2, 1.0]
    reward: float
    # Whether the plan passed structural validation
    passed_validation: bool
    # Human-readable one-line summary
    summary: str

    # --- Grader-facing step facts (set by pipeline, consumed by environment) ---
    step_slot_ratio: Optional[float] = None
    step_task_complete: Optional[bool] = None
    step_harmful: Optional[bool] = None
    step_macro_prior_count: Optional[int] = None
    step_macro_used: Optional[bool] = None
    step_macro_miss_count: Optional[int] = None
    step_baseline_calls: Optional[int] = None
    step_actual_calls: Optional[int] = None
    step_macro_creation_bonus: Optional[float] = None
    step_macro_usage_bonus: Optional[float] = None
    step_macro_miss_penalty: Optional[float] = None
    step_efficiency_score: Optional[float] = None


