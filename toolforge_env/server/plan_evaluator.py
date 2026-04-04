"""
Plan Evaluator

Consolidates the core evaluation logic:
1. Structural validation
2. Semantic slot judgment
3. Reward calculation
4. Token-cost calculation
5. Final score combination

Functions are pure and deterministic where possible.
"""

import logging
import math
from typing import Dict, List, Any

from toolforge_env.models import (
    PlanAccuracyResult,
    SlotJudgmentResult,
    Task,
    TokenCostResult,
    Tool,
    ToolCall,
    ToolEvaluation,
    ValidationResult,
)
from toolforge_env.server.slots import DEVOPS_SLOTS

logger = logging.getLogger(__name__)

# --- Named Constants ---
PLAN_ACCURACY_WEIGHT = 0.7
TOKEN_EFFICIENCY_WEIGHT = 0.3
SLOT_COMPLETION_CURVE_K = 3.0
UNNECESSARY_CALL_PENALTY = 0.05
MAX_UNNECESSARY_CALL_PENALTY = 0.20
MACRO_USAGE_BONUS = 0.10
FINAL_REWARD_MIN = -1.0
FINAL_REWARD_MAX = 1.0
TOKEN_EFFICIENCY_MIN = -1.0
TOKEN_EFFICIENCY_MAX = 1.0

# --- Helper Functions for Stage 2 (Semantic Judge) ---

def _build_judge_request(
    task_prompt: str,
    required_slots: List[str],
    slot_definitions: Dict[str, str],
    available_tools: List[Tool],
    plan: List[ToolCall],
) -> Dict[str, Any]:
    """Helper to build the expected LLM judge prompt/input."""
    # Placeholder structure for when real call is integrated
    return {
        "task_prompt": task_prompt,
        "required_slots": required_slots,
        "slot_definitions": slot_definitions,
        "tools": [t.name for t in available_tools],
        "plan": [{"tool": c.tool_name, "params": c.params} for c in plan]
    }

def _simulate_llm_judgment(
    judge_request: Dict[str, Any],
    plan: List[ToolCall],
    required_slots: List[str]
) -> List[Dict[str, Any]]:
    """Helper to simulate the LLM response deterministically.
    
    Produces classification: 'relevant', 'unnecessary', or 'harmful'.
    """
    results = []
    slots_filled_so_far = set()
    
    # We simulate semantic relevance by simply mapping the sequence
    # to the required slots.
    for i, call in enumerate(plan):
        # Extremely naive heuristic for simulation:
        if call.tool_name == "delete" or "drop" in call.tool_name:
            # Simulate a harmful destructive call
            classification = "harmful"
            slot = None
        else:
            # If we haven't filled all slots and it's not a duplicate, let's pretend it fills a slot
            if len(slots_filled_so_far) < len(required_slots):
                slot = required_slots[len(slots_filled_so_far)]
                classification = "relevant"
                slots_filled_so_far.add(slot)
            else:
                slot = None
                classification = "unnecessary"
                
        results.append({
            "tool_call_index": i,
            "tool_name": call.tool_name,
            "fills_slot": slot,
            "classification": classification,
            "reason": f"Simulated classification: {classification}"
        })
        
    return results

def _parse_simulated_judgment(raw_results: List[Dict[str, Any]], required_slots: List[str]) -> SlotJudgmentResult:
    """Helper to convert the raw simulated LLM output into SlotJudgmentResult."""
    evaluations = []
    slots_filled = []
    harmful_calls_present = False
    
    for r in raw_results:
        evaluations.append(
            ToolEvaluation(
                tool_call_index=r["tool_call_index"],
                tool_name=r["tool_name"],
                fills_slot=r["fills_slot"],
                classification=r["classification"],
                reason=r["reason"],
            )
        )
        if r["classification"] == "harmful":
            harmful_calls_present = True
            
        if r["fills_slot"] and r["classification"] == "relevant":
             if r["fills_slot"] not in slots_filled:
                 slots_filled.append(r["fills_slot"])
                 
    slots_missing = [s for s in required_slots if s not in slots_filled]
    task_complete = (len(slots_missing) == 0)
    
    return SlotJudgmentResult(
        evaluations=evaluations,
        slots_filled=slots_filled,
        slots_missing=slots_missing,
        task_complete=task_complete,
        harmful_calls_present=harmful_calls_present,
    )

# --- Helper Function for Stage 4 (Dynamic Baseline) ---

def calculate_dynamic_baseline_tokens(task: Task, available_tools: Dict[str, Tool]) -> int:
    """Computes the expected baseline token cost for a task.
    
    Looks up cost of each required atomic step. Uses a fallback if lookup fails.
    """
    if not task.required_steps:
        return 0
        
    baseline = 0
    for step in task.required_steps:
        if step in available_tools:
            baseline += available_tools[step].token_cost
        else:
            baseline += 20 # Deterministic fallback
            
    return baseline

# --- Public APIs ---

def get_relevant_slots(required_slots: List[str]) -> Dict[str, str]:
    """Return the subset of DEVOPS_SLOTS matching required_slots."""
    return {
        name: DEVOPS_SLOTS[name]
        for name in required_slots
        if name in DEVOPS_SLOTS
    }

def run_sanity_validation(
    plan: List[ToolCall],
    available_tools: Dict[str, Tool],
) -> ValidationResult:
    """Stage 1: Structural validation of the plan.
    
    Validation order:
      1. Empty plan            → EMPTY_PLAN    (penalty -1.0)
      2. Unknown tool name     → INVALID_TOOL  (penalty -0.8)
      3. Missing required param→ MISSING_PARAM (penalty -0.6)
      4. Extra unknown param   → EXTRA_PARAM   (penalty -0.3)
      5. All pass              → VALID         (penalty  0.0)
    """
    if plan is None or len(plan) == 0:
        return ValidationResult(
            valid=False,
            reason="EMPTY_PLAN",
            penalty=-1.0,
            detail="Plan contains no tool calls.",
        )

    for call in plan:
        if call.tool_name not in available_tools:
            return ValidationResult(
                valid=False,
                reason="INVALID_TOOL",
                penalty=-0.8,
                detail=f"Tool '{call.tool_name}' does not exist in toolbox.",
            )

    for call in plan:
        tool = available_tools[call.tool_name]
        allowed_params = set(tool.params_schema.get("properties", {}).keys())
        required_params = set(tool.params_schema.get("required", []))
        provided_params = set(call.params.keys())

        missing = required_params - provided_params
        if missing:
            return ValidationResult(
                valid=False,
                reason="MISSING_PARAM",
                penalty=-0.6,
                detail=f"Tool '{call.tool_name}' is missing required parameter(s): {sorted(missing)}.",
            )

        extra = provided_params - allowed_params
        if extra:
            return ValidationResult(
                valid=False,
                reason="EXTRA_PARAM",
                penalty=-0.3,
                detail=f"Tool '{call.tool_name}' received unexpected parameter(s): {sorted(extra)}.",
            )

    return ValidationResult(
        valid=True,
        reason="VALID",
        penalty=0.0,
    )

def run_slot_judgment(
    task_prompt: str,
    required_slots: List[str],
    slot_definitions: Dict[str, str],
    available_tools: List[Tool],
    plan: List[ToolCall],
) -> SlotJudgmentResult:
    """Stage 2: Evaluate a validated plan against the task's semantic slots."""
    req = _build_judge_request(task_prompt, required_slots, slot_definitions, available_tools, plan)
    raw = _simulate_llm_judgment(req, plan, required_slots)
    
    result = _parse_simulated_judgment(raw, required_slots)
    if result.harmful_calls_present:
        logger.warning("Simulated Judge detected harmful calls in plan.")
        
    return result

def plan_accuracy_score(
    validation_result: ValidationResult,
    slot_judgment: SlotJudgmentResult,
    task: Task,
    goal_achieved: bool = False,
) -> PlanAccuracyResult:
    """Stage 3: Measures semantic correctness only (bounded [-1.0, 0.0])."""
    
    if len(task.required_slots) > 0:
        filled_ratio = len(slot_judgment.slots_filled) / len(task.required_slots)
    else:
        filled_ratio = 1.0
        
    # Reverse-exponential curve
    slot_score = ((math.exp(SLOT_COMPLETION_CURVE_K * filled_ratio) - 1.0) / (math.exp(SLOT_COMPLETION_CURVE_K) - 1.0)) - 1.0
    
    unnecessary_count = sum(1 for e in slot_judgment.evaluations if e.classification == "unnecessary")
    unnecessary_penalty = -min(UNNECESSARY_CALL_PENALTY * unnecessary_count, MAX_UNNECESSARY_CALL_PENALTY)
    
    score = slot_score + unnecessary_penalty
    
    # Bound to [-1.0, 0.0] 
    score = max(-1.0, min(0.0, score))
    
    logger.debug(f"Plan Accuracy Step 3: score={score:.3f} (slot={slot_score:.3f}, unnec_pen={unnecessary_penalty:.3f})")

    return PlanAccuracyResult(
        slot_completion_ratio=filled_ratio,
        slot_score=slot_score,
        unnecessary_penalty=unnecessary_penalty,
        score=score,
        breakdown={
            "slot_score": slot_score,
            "unnecessary_penalty": unnecessary_penalty,
        }
    )

def run_token_calculation(
    plan: List[ToolCall],
    accepted_macros: List[Tool],
    task: Task,
    available_tools: Dict[str, Tool],
) -> TokenCostResult:
    """Stage 4: Compute a token-efficiency score and macro savings for a plan."""
    baseline_tokens = calculate_dynamic_baseline_tokens(task, available_tools)
    tokens_used = sum(call.token_cost for call in plan)
    
    if baseline_tokens <= 0:
        efficiency_ratio = 0.0
    else:
        efficiency_ratio = (baseline_tokens - tokens_used) / baseline_tokens
        
    # Determine macro usage
    macro_names = {m.name for m in accepted_macros}
    macro_used = any(call.tool_name in macro_names for call in plan)
    
    macro_savings = max(0, baseline_tokens - tokens_used)
    
    macro_bonus = MACRO_USAGE_BONUS if (macro_used and efficiency_ratio > 0) else 0.0
    
    efficiency_score = efficiency_ratio + macro_bonus
    efficiency_score = max(TOKEN_EFFICIENCY_MIN, min(TOKEN_EFFICIENCY_MAX, efficiency_score))
    
    logger.debug(f"Token Calculation Step 4: efficiency={efficiency_score:.3f} (ratio={efficiency_ratio:.3f}, bonus={macro_bonus:.3f})")

    return TokenCostResult(
        tokens_used=tokens_used,
        baseline_tokens=baseline_tokens,
        efficiency_ratio=efficiency_ratio,
        efficiency_score=efficiency_score,
        macro_savings=macro_savings,
        macro_bonus=macro_bonus,
    )

def reward_calculation(
    plan_accuracy: PlanAccuracyResult,
    token_cost: TokenCostResult,
) -> float:
    """Stage 5: True reward function combining Stage 3 and Stage 4. (-1.0 to 1.0)"""
    reward = (PLAN_ACCURACY_WEIGHT * plan_accuracy.score) + (TOKEN_EFFICIENCY_WEIGHT * token_cost.efficiency_score)
    final_score = max(FINAL_REWARD_MIN, min(FINAL_REWARD_MAX, reward))
    
    logger.info(f"Final Reward: {final_score:.3f}")
    
    return final_score
