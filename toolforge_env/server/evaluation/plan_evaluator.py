"""
Plan Evaluator

Core evaluation logic for the ToolForge environment:
  Stage 1: Structural validation
  Stage 2: Semantic slot judgment (LLM-based with retry + fallback)
  Stage 3: Macro bonuses (creation + usage)
  Stage 4: Tool efficiency scoring
  Final:   Bounded reward combination
"""

import logging
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from models import (
    SlotJudgmentResult,
    Task,
    Tool,
    ToolCall,
    ValidationResult,
)
from server.evaluation.llm_eval_prompts import (
    SLOT_JUDGE_SYSTEM_PROMPT,
    build_slot_judge_user_prompt,
)
from server.evaluation.tool_slot_mappings import (
    TOOL_TO_POSSIBLE_SLOTS,
    HARMFUL_TOOLS_TO_REQUIRED_SLOT,
)
from server.slots import DEVOPS_SLOTS

logger = logging.getLogger(__name__)

# --- Named Constants (Bounded Reward Design) ---

# Final reward bounds
FINAL_REWARD_MIN = -0.2
FINAL_REWARD_MAX = 1.0

# Stage 1: Validation
VALIDATION_PENALTY = -0.2

# Stage 2: Slot score bounds
SLOT_THRESHOLD = 0.65
SLOT_SCORE_MIN = -0.15   # slot_ratio == 0.0
SLOT_SCORE_MAX = 0.25    # slot_ratio == 1.0

# Stage 3: Macro bonuses
MACRO_CREATION_MAX = 0.20
MACRO_CREATION_DECAY_FLOOR = 0.05
MACRO_CREATION_THRESHOLD = 2
MACRO_CREATION_FULL_RANGE = {2, 3}  # counts that get full reward
MACRO_USAGE_PARTIAL = 0.03   # when 0.65 <= slot_ratio < 1.0
MACRO_USAGE_FULL = 0.05      # when slot_ratio == 1.0

# Stage 4: Tool efficiency bounds
EFFICIENCY_SCORE_BASELINE = 0.2  # exact baseline match
EFFICIENCY_SCORE_MIN = 0.0
EFFICIENCY_SCORE_MAX = 0.5
EFFICIENCY_SCALE = 0.3  # multiplier on efficiency_ratio

# LLM configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")


# ─── Stage 2 Helpers: Semantic Judge ─────────────────────────────────────────

def _call_llm_slot_judgment(
    task_prompt: str,
    required_slots: List[str],
    slot_definitions: Dict[str, str],
    available_tools: List[Tool],
    plan: List[ToolCall],
) -> Dict[str, Any]:
    """Call the OpenAI-compatible LLM and return its JSON response."""
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or None)
    tool_desc_map = {t.name: t.description for t in available_tools}

    plan_with_descs = []
    for call in plan:
        desc = tool_desc_map.get(call.tool_name, "No description available.")
        plan_with_descs.append({
            "tool_name": call.tool_name,
            "tool_description": desc
        })

    user_prompt = build_slot_judge_user_prompt(
        task_prompt=task_prompt,
        required_slots=required_slots,
        slot_definitions=slot_definitions,
        plan=plan_with_descs,
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SLOT_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content or "{}"
    content = content.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(content)

def _fallback_parse_plan_to_llm_summary(
    expanded_plan: List[ToolCall],
    required_slots: List[str],
) -> Dict[str, Any]:
    """Rule-based fallback semantic parser when LLM judge fails."""
    slots_filled = []
    unnecessary_calls = []
    harmful_calls = []
    
    filled_so_far = set()
    
    for call in expanded_plan:
        tool_name = call.tool_name
        possible_slots = TOOL_TO_POSSIBLE_SLOTS.get(tool_name, [])
        
        filled_any = False
        for slot in possible_slots:
            if slot in required_slots and slot not in filled_so_far:
                slots_filled.append(slot)
                filled_so_far.add(slot)
                filled_any = True
                break
                
        if not filled_any:
            if tool_name in HARMFUL_TOOLS_TO_REQUIRED_SLOT:
                req_slot = HARMFUL_TOOLS_TO_REQUIRED_SLOT[tool_name]
                if req_slot not in required_slots:
                    harmful_calls.append(tool_name)
                else:
                    unnecessary_calls.append(tool_name)
            else:
                unnecessary_calls.append(tool_name)
                
    slots_missing = [s for s in required_slots if s not in filled_so_far]
    
    return {
        "slots_filled": slots_filled,
        "slots_missing": slots_missing,
        "unnecessary_calls": unnecessary_calls,
        "harmful_calls": harmful_calls
    }

def _parse_llm_judgment(raw_json: Dict[str, Any], required_slots: List[str]) -> SlotJudgmentResult:
    """Convert the flat LLM summary output into a SlotJudgmentResult.

    Expected LLM schema:
    {
        "slots_filled": ["SLOT_NAME"],
        "slots_missing": ["SLOT_NAME"],
        "unnecessary_calls": ["tool_name"],
        "harmful_calls": ["tool_name"]
    }
    """
    slots_filled = raw_json.get("slots_filled", [])
    slots_missing = raw_json.get("slots_missing", [])

    # Ensure slots_filled only contains requested slots
    slots_filled = [s for s in slots_filled if s in required_slots]

    if not slots_missing:
        slots_missing = [s for s in required_slots if s not in slots_filled]

    harmful_calls = raw_json.get("harmful_calls", [])
    harmful_calls_present = len(harmful_calls) > 0

    task_complete = len(slots_missing) == 0

    evaluations = []

    return SlotJudgmentResult(
        evaluations=evaluations,
        slots_filled=slots_filled,
        slots_missing=slots_missing,
        task_complete=task_complete,
        harmful_calls_present=harmful_calls_present,
    )


# ─── Stage 4 Helper: Dynamic Baseline ───────────────────────────────────────

def calculate_dynamic_baseline_tokens(task: Task, available_tools: Dict[str, Tool]) -> int:
    """Compute the expected baseline token cost for a task.

    Uses baseline_call_count when provided, otherwise returns 0.
    """
    if task.baseline_call_count > 0:
        return task.baseline_call_count
    return 0


# ─── Macro Recognition Helpers ───────────────────────────────────────────────

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


# ─── Stage Scoring Functions ─────────────────────────────────────────────────

def compute_slot_score(slot_ratio: float) -> float:
    """Stage 2: Piecewise linear slot score bounded to [-0.15, 0.25].

    - slot_ratio < 0.65:  maps [-0.15, 0.0]
    - slot_ratio >= 0.65: maps [0.0,  0.25]
    """
    if slot_ratio < SLOT_THRESHOLD:
        return SLOT_SCORE_MIN * (1.0 - slot_ratio / SLOT_THRESHOLD)
    else:
        return SLOT_SCORE_MAX * ((slot_ratio - SLOT_THRESHOLD) / (1.0 - SLOT_THRESHOLD))


def compute_macro_creation_bonus(
    macro_proposal: Optional[Tool],
    sequence_counts: Optional[Dict[str, int]],
    slot_ratio: float,
) -> float:
    """Stage 3a: Macro creation bonus bounded to [0.0, 0.20].

    Gate: slot_ratio >= 0.65
    - prior_count < 2: 0.0
    - prior_count in {2, 3}: 0.20
    - prior_count > 3: decays with floor 0.05
    """
    if slot_ratio < SLOT_THRESHOLD:
        return 0.0
    if macro_proposal is None or sequence_counts is None:
        return 0.0
    if macro_proposal.steps is None or len(macro_proposal.steps) < 2:
        return 0.0

    proposed_sequence = tuple(call.tool_name for call in macro_proposal.steps)
    prior_count = count_prior_sequence_occurrences(proposed_sequence, sequence_counts)

    if prior_count < MACRO_CREATION_THRESHOLD:
        return 0.0
    if prior_count in MACRO_CREATION_FULL_RANGE:
        return MACRO_CREATION_MAX
    return max(MACRO_CREATION_DECAY_FLOOR, MACRO_CREATION_MAX * (3.0 / prior_count))


def compute_macro_usage_bonus(
    plan: List[ToolCall],
    accepted_macros: List[Tool],
    slot_ratio: float,
) -> float:
    """Stage 3b: Macro usage bonus bounded to [0.0, 0.05].

    Gate: slot_ratio >= 0.65
    """
    if slot_ratio < SLOT_THRESHOLD:
        return 0.0
    macro_names = {m.name for m in accepted_macros}
    macro_used = any(call.tool_name in macro_names for call in plan)
    if not macro_used:
        return 0.0

    if slot_ratio == 1.0:
        return MACRO_USAGE_FULL
    return MACRO_USAGE_PARTIAL


def compute_efficiency_score(
    plan: List[ToolCall],
    task: Task,
    available_tools: Dict[str, Tool],
) -> float:
    """Stage 4: Count-based efficiency score bounded to [0.0, 0.5].

    Only called when slot_ratio == 1.0.
    - baseline match -> 0.2
    - better than baseline -> above 0.2
    - worse than baseline -> below 0.2
    """
    baseline = calculate_dynamic_baseline_tokens(task, available_tools)
    actual = len(plan)

    if baseline <= 0:
        return EFFICIENCY_SCORE_BASELINE

    efficiency_ratio = (baseline - actual) / baseline
    score = EFFICIENCY_SCORE_BASELINE + EFFICIENCY_SCALE * efficiency_ratio
    return max(EFFICIENCY_SCORE_MIN, min(EFFICIENCY_SCORE_MAX, score))


# ─── Public APIs ─────────────────────────────────────────────────────────────

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
      1. Empty plan            → EMPTY_PLAN    (penalty -0.2)
      2. Unknown tool name     → INVALID_TOOL  (penalty -0.2)
      3. All pass              → VALID         (penalty  0.0)
    """
    if plan is None or len(plan) == 0:
        return ValidationResult(
            valid=False,
            reason="EMPTY_PLAN",
            penalty=VALIDATION_PENALTY,
            detail="Plan contains no tool calls.",
        )

    for call in plan:
        if call.tool_name not in available_tools:
            return ValidationResult(
                valid=False,
                reason="INVALID_TOOL",
                penalty=VALIDATION_PENALTY,
                detail=f"Tool '{call.tool_name}' does not exist in toolbox.",
            )

    return ValidationResult(valid=True, reason="VALID", penalty=0.0)

def _expand_macros_in_plan(plan: List[ToolCall], available_tools: List[Tool]) -> List[ToolCall]:
    """Recursively expand macro calls into their atomic components for semantic evaluation."""
    tool_map = {t.name: t for t in available_tools}
    expanded_plan = []

    for call in plan:
        tool = tool_map.get(call.tool_name)
        if tool and tool.is_macro and tool.steps:
            expanded_plan.extend(_expand_macros_in_plan(tool.steps, available_tools))
        else:
            expanded_plan.append(call)

    return expanded_plan

def run_slot_judgment(
    task_prompt: str,
    required_slots: List[str],
    slot_definitions: Dict[str, str],
    available_tools: List[Tool],
    plan: List[ToolCall],
) -> SlotJudgmentResult:
    """Stage 2: Evaluate a validated plan against the task's semantic slots."""

    max_attempts = 3
    raw_json = None
    for attempt in range(max_attempts):
        try:
            raw_json = _call_llm_slot_judgment(
                task_prompt=task_prompt,
                required_slots=required_slots,
                slot_definitions=slot_definitions,
                available_tools=available_tools,
                plan=plan,
            )
            if not raw_json or "slots_filled" not in raw_json:
                raise ValueError("LLM returned an invalid or empty response")
            break  # Success
        except Exception as exc:
            logger.warning("LLM slot judge attempt %d/%d failed: %s", attempt + 1, max_attempts, exc)
            if attempt == max_attempts - 1:
                logger.error("LLM slot judge failed after %d attempts. Activating fallback parser.", max_attempts)
                expanded_plan = _expand_macros_in_plan(plan, available_tools)
                raw_json = _fallback_parse_plan_to_llm_summary(expanded_plan, required_slots)
                break

    result = _parse_llm_judgment(raw_json, required_slots)
    if result.harmful_calls_present:
        logger.warning("Slot judge detected harmful calls in plan.")

    return result


def compute_step_reward(
    slot_judgment: SlotJudgmentResult,
    task: Task,
    plan: List[ToolCall],
    available_tools: Dict[str, Tool],
    accepted_macros: List[Tool],
    macro_proposal: Optional[Tool] = None,
    sequence_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Compute the full step reward using bounded additive stages.

    Returns a dict with per-stage contributions and the final clamped reward.
    """
    n_required = len(task.required_slots)
    n_filled = len(slot_judgment.slots_filled)
    slot_ratio = n_filled / n_required if n_required > 0 else 1.0

    slot_score = compute_slot_score(slot_ratio)

    macro_creation = compute_macro_creation_bonus(macro_proposal, sequence_counts, slot_ratio)
    macro_usage = compute_macro_usage_bonus(plan, accepted_macros, slot_ratio)

    efficiency_score = 0.0

    if slot_ratio < SLOT_THRESHOLD:
        final_raw = slot_score
    elif slot_ratio < 1.0:
        final_raw = slot_score + macro_creation + macro_usage
    else:
        efficiency_score = compute_efficiency_score(plan, task, available_tools)
        final_raw = slot_score + macro_creation + macro_usage + efficiency_score

    final_reward = max(FINAL_REWARD_MIN, min(FINAL_REWARD_MAX, final_raw))

    logger.info("Final Reward: %.3f", final_reward)

    return {
        "slot_ratio": slot_ratio,
        "slot_score": slot_score,
        "macro_creation": macro_creation,
        "macro_usage": macro_usage,
        "efficiency_score": efficiency_score,
        "final_reward": final_reward,
    }
