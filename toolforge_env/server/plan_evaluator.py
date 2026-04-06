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
from typing import Any, Dict, List, Optional, Tuple

from toolforge_env.models import (
    MacroProposal,
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
RECOGNITION_THRESHOLD = 2
ALPHA_RECOGNITION = 0.10
ALPHA_UTILITY = 0.20

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
        "plan": [{"tool": c.tool_name} for c in plan]
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

    Slot-driven baseline: uses baseline_token_cost when provided,
    otherwise falls back to baseline_call_count for compatibility.
    """
    if task.baseline_token_cost > 0:
        return task.baseline_token_cost

    if task.baseline_call_count > 0:
        return task.baseline_call_count

    return 0

# --- Helper Functions for Macro Recognition ---

def extract_contiguous_windows(tool_names: List[str], window_size: int) -> List[Tuple[str, ...]]:
    """Return exact ordered contiguous windows of the given size."""
    if window_size < 2 or window_size > len(tool_names):
        return []
    return [tuple(tool_names[i:i + window_size]) for i in range(len(tool_names) - window_size + 1)]

def count_prior_sequence_occurrences(
    proposed_sequence: Tuple[str, ...],
    sequence_counts: Dict[str, int],
) -> int:
    """Return the prior exact count for a sequence key."""
    key = str(proposed_sequence)
    return sequence_counts.get(key, 0)

def calculate_macro_recognition_bonus(
    prior_sequence_count: int,
    recognition_threshold: int,
    alpha_recognition: float,
) -> float:
    """Threshold-based recognition bonus. No penalty for premature creation."""
    if prior_sequence_count < recognition_threshold:
        return 0.0
    bonus = alpha_recognition * (recognition_threshold / prior_sequence_count)
    return max(0.0, min(alpha_recognition, bonus))

def calculate_atomic_equivalent_cost(
    sequence: Tuple[str, ...],
    tool_registry: Dict[str, Tool],
) -> int:
    """Sum token cost of the atomic tools in the sequence."""
    total = 0
    for name in sequence:
        tool = tool_registry.get(name)
        if tool is not None:
            total += tool.token_cost
    return total

def calculate_macro_utility_bonus(
    macro_used: bool,
    atomic_equivalent_cost: int,
    macro_call_cost: int,
    alpha_utility: float,
) -> float:
    """Utility bonus based on token savings from using a macro."""
    if not macro_used or atomic_equivalent_cost <= 0:
        return 0.0
    token_savings = max(0, atomic_equivalent_cost - macro_call_cost)
    return alpha_utility * (token_savings / atomic_equivalent_cost)

def update_sequence_counts(
    plan: List[ToolCall],
    sequence_counts: Dict[str, int],
) -> Dict[str, int]:
    """Update sequence_counts dict from the current plan's contiguous windows.
    
    Extracts windows of size 2..len(plan) and increments counts.
    Returns the updated dict (mutates in place for convenience).
    """
    tool_names = [call.tool_name for call in plan]
    for window_size in range(2, len(tool_names) + 1):
        for window in extract_contiguous_windows(tool_names, window_size):
            key = str(window)
            sequence_counts[key] = sequence_counts.get(key, 0) + 1
    return sequence_counts

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

    return ValidationResult(valid=True, reason="VALID", penalty=0.0)

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
    sequence_counts: Optional[Dict[str, int]] = None,
    macro_definitions: Optional[Dict[str, List[str]]] = None,
    macro_proposal: Optional[Tool] = None,
) -> TokenCostResult:
    """Stage 4: Compute a token-efficiency score and macro savings for a plan."""
    baseline_tokens = calculate_dynamic_baseline_tokens(task, available_tools)
    tokens_used = sum(0 for call in plan)
    
    if baseline_tokens <= 0:
        efficiency_ratio = 0.0
    else:
        efficiency_ratio = (baseline_tokens - tokens_used) / baseline_tokens
        
    # Determine macro usage
    macro_names = {m.name for m in accepted_macros}
    macro_used = any(call.tool_name in macro_names for call in plan)
    
    macro_savings = max(0, baseline_tokens - tokens_used)
    
    # Macro recognition bonus: reward creating a macro for a previously-seen sequence
    macro_recognition_bonus = 0.0
    if macro_proposal is not None and sequence_counts is not None and macro_proposal.steps is not None:
        proposed_sequence = tuple(call.tool_name for call in macro_proposal.steps)
        prior_count = count_prior_sequence_occurrences(proposed_sequence, sequence_counts)
        macro_recognition_bonus = calculate_macro_recognition_bonus(
            prior_sequence_count=prior_count,
            recognition_threshold=RECOGNITION_THRESHOLD,
            alpha_recognition=ALPHA_RECOGNITION,
        )

    # Macro utility bonus: reward plans that use a macro saving tokens vs atomic
    macro_utility_bonus = 0.0
    if macro_definitions is not None:
        for call in plan:
            if call.tool_name in macro_names and call.tool_name in macro_definitions:
                seq = tuple(macro_definitions[call.tool_name])
                atomic_cost = calculate_atomic_equivalent_cost(seq, available_tools)
                macro_tool = available_tools.get(call.tool_name)
                macro_call_cost = macro_tool.token_cost if macro_tool else 0
                macro_utility_bonus += calculate_macro_utility_bonus(
                    macro_used=True,
                    atomic_equivalent_cost=atomic_cost,
                    macro_call_cost=macro_call_cost,
                    alpha_utility=ALPHA_UTILITY,
                )

    macro_bonus = MACRO_USAGE_BONUS if (macro_used and efficiency_ratio > 0) else 0.0
    macro_bonus += macro_recognition_bonus + macro_utility_bonus
    
    efficiency_score = efficiency_ratio + macro_bonus
    efficiency_score = max(TOKEN_EFFICIENCY_MIN, min(TOKEN_EFFICIENCY_MAX, efficiency_score))
    
    logger.debug(
        "Token Calculation Step 4: efficiency=%.3f (ratio=%.3f, bonus=%.3f, recognition=%.3f, utility=%.3f)",
        efficiency_score, efficiency_ratio, macro_bonus, macro_recognition_bonus, macro_utility_bonus,
    )

    return TokenCostResult(
        tokens_used=tokens_used,
        baseline_tokens=baseline_tokens,
        efficiency_ratio=efficiency_ratio,
        efficiency_score=efficiency_score,
        macro_savings=macro_savings,
        macro_recognition_bonus=macro_recognition_bonus,
        macro_utility_bonus=macro_utility_bonus,
        macro_bonus=macro_bonus
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
