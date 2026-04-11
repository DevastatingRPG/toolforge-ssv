# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared data, constants, and CSS for the ToolForge Gradio UI.

Owns:
    - SYSTEM_PROMPT       : The default agent system prompt (editable in BYOA tab)
    - ATOMIC_TOOLS        : List of all atomic tool names available to agents
    - SCRIPTED_PROFILES   : Pre-scripted behavioral profiles used in Demo Mode
    - CUSTOM_CSS          : Global CSS injected into the Gradio app
    - Helper formatters   : Pure functions that render episode frames as HTML

Episode frame schema (dict):
    task_id          (str)             task identifier
    task_prompt      (str)             task description shown to agent
    difficulty       (str)             "easy" | "medium" | "hard"
    required_slots   (list[str])       semantic slots the task requires
    plan             (list[str])       ordered tool names in the agent's plan
    used_macro       (bool)            True if the plan contains a macro call
    macro_proposed   (dict | None)     {"name": str, "steps": list[str]} or None
    reward           (float)           step reward in [-0.2, 1.0]
    turns_used       (int)             actual length of the plan
    turns_baseline   (int)             baseline_call_count for the task
    macros_so_far    (list[dict])      cumulative macro library after this step
    note             (str)             narrative explaining agent behaviour
"""

import logging
import textwrap
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ===========================================================================
# SECTION 1: AGENT SYSTEM PROMPT
# ---------------------------------------------------------------------------
# Copied verbatim from inference.py so the BYOA tab pre-fills the correct
# prompt.  Users MAY edit the strategy section; the JSON-response requirement
# (bottom of the prompt) is locked by the parser and must not be removed.
# ===========================================================================

SYSTEM_PROMPT: str = textwrap.dedent(
    """
    You are an agent acting in the Toolforge environment.
    Return ONLY valid JSON matching the ToolForgeAction schema.

    Objective:
    - Maximize reward by completing task-required behavior with the relevant useful tool calls.

    Rules:
    - Use only tool names that appear in Available tools.
    - Keep the plan minimal and avoid unnecessary calls.
    - Use action_type="propose_plan" by default.
    - If proposing a macro, use action_type="propose_plan_with_macro" and include macro_proposal.
    - macro_proposal.steps must be an ordered sequence of at least 2 existing non-macro tools.
    - Do not use a newly proposed macro in the same step's plan.
    - If reusing an existing macro, do NOT send macro_proposal.
    - Never propose a macro name that already exists in Available tools.

    Macro policy:
    - Detect repetition by operation signature, not by exact wording.
    - Treat different service names/channels/contexts as the same pattern if tool order is the same.
    - Build a canonical pattern signature from tool order (for example: restart->healthcheck->notify).
    - Reuse an existing macro immediately when it matches the needed sequence.
    - Create a macro only when a contiguous ordered sequence has already repeated in prior steps.
    - Propose each macro only once. After approval, switch to action_type="propose_plan" with macro_proposal=null.
    - Prefer reusable patterns seen across task names and phases, not one-off service-specific steps.
    - Good reusable patterns include deploy->healthcheck->notify, restart->healthcheck->notify,
      rollback->healthcheck->notify, rollback->restart->healthcheck, scale->healthcheck->notify,
      and restart->deploy->healthcheck.
    - Some tasks are intentionally varied in wording; still group them by the same underlying
      tool-order signature.
    - This evaluator often treats each plan entry as filling at most one required slot.
    - Therefore, avoid macro-only one-entry plans for multi-slot tasks.
    - If task.required_slots has length N, prefer a plan with at least N entries, mixing macro
      calls with needed atomic calls.

    Naming:
    - Use short snake_case names that describe the operation pattern.

    ⚠️  IMPORTANT — DO NOT EDIT BELOW THIS LINE  ⚠️
    Respond ONLY with valid JSON matching the ToolForgeAction schema.
    No markdown, no explanation, no commentary outside the JSON object.
    """
).strip()

# ===========================================================================
# SECTION 2: ATOMIC TOOL INVENTORY
# ---------------------------------------------------------------------------
# These mirror server/tools.py.  Kept here so UI files never import from
# the server layer, preserving the clean UI/backend boundary.
# ===========================================================================

ATOMIC_TOOLS: List[str] = [
    "deploy",
    "patch",
    "healthcheck",
    "run_tests",
    "ping",
    "notify",
    "pagerduty_alert",
    "rollback",
    "scale",
    "restart",
]

# Human-readable descriptions for each atomic tool (shown in HvL tool picker)
TOOL_DESCRIPTIONS: Dict[str, str] = {
    "deploy":          "Deploy a service / application version",
    "patch":           "Apply security patches or hotfixes",
    "healthcheck":     "Check service health status",
    "run_tests":       "Execute the test suite",
    "ping":            "Network connectivity check",
    "notify":          "Send a notification to a channel",
    "pagerduty_alert": "Trigger a PagerDuty incident alert",
    "rollback":        "Revert to a previous stable version",
    "scale":           "Adjust the number of running replicas",
    "restart":         "Perform a rolling restart of a service",
}

# ===========================================================================
# SECTION 3: SCRIPTED BEHAVIORAL PROFILES
# ---------------------------------------------------------------------------
# Each profile is keyed by MODEL_LABEL -> DIFFICULTY -> list[episode_frame].
#
# Episode frames are purely presentational data — they do NOT call the real
# environment.  They simulate what a real agent run would look like so the
# Demo tab can function without any API key.
#
# Narrative arc per profile:
#   GPT-4o   : Fast learner — proposes macro by episode 3, reuses it by 4.
#   Llama 3  : Moderate learner — one unnecessary call early, late macro.
#   Mistral  : Inconsistent — one harmful call (negative reward), eventual macro.
# ===========================================================================

SCRIPTED_PROFILES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {

    # -----------------------------------------------------------------------
    # Profile 1 — GPT-4o (simulated) | Fast, efficient macro learner
    # -----------------------------------------------------------------------
    "GPT-4o (simulated)": {
        "easy": [
            {
                "task_id":       "e-dep-1",
                "task_prompt":   "Release 'inventory-db-proxy' v1.0.5, check health status, and notify #product.",
                "difficulty":    "easy",
                "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
                "plan":          ["deploy", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        0.65,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [],
                "note":          "Agent executes the canonical 3-step deployment pipeline correctly.",
            },
            {
                "task_id":       "e-dep-2",
                "task_prompt":   "Roll out 'auth-svc' v1.8.0, check health, and notify #ops-chat.",
                "difficulty":    "easy",
                "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
                "plan":          ["deploy", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        0.65,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [],
                "note":          "Same pattern repeated. Agent internally tracks this sequence.",
            },
            {
                "task_id":       "e-dep-3",
                "task_prompt":   "Deploy 'api-v2' v2.3.4, check health status, and notify #support.",
                "difficulty":    "easy",
                "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
                "plan":          ["deploy", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": {
                    "name":  "deploy_verify_notify",
                    "steps": ["deploy", "healthcheck", "notify"],
                },
                "reward":        0.83,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [
                    {"name": "deploy_verify_notify", "steps": ["deploy", "healthcheck", "notify"]},
                ],
                "note":          "Pattern seen 3 times. Agent proposes macro 'deploy_verify_notify'. "
                                 "Macro creation bonus (+0.18) applied on top of slot score.",
            },
            {
                "task_id":       "e-dep-4",
                "task_prompt":   "Release 'image-processor' v2.7.4, check health, and notify #security.",
                "difficulty":    "easy",
                "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
                "plan":          ["deploy_verify_notify"],
                "used_macro":    True,
                "macro_proposed": None,
                "reward":        0.90,
                "turns_used":    1,
                "turns_baseline": 3,
                "macros_so_far": [
                    {"name": "deploy_verify_notify", "steps": ["deploy", "healthcheck", "notify"]},
                ],
                "note":          "Agent reuses 'deploy_verify_notify'. 2 turns saved vs baseline. "
                                 "Macro usage bonus (+0.05) and efficiency score (+0.50) both applied.",
            },
            {
                "task_id":       "e-res-1",
                "task_prompt":   "Scale 'websocket-server' to 6, ping it to ensure connectivity, and notify #monitoring.",
                "difficulty":    "easy",
                "required_slots": ["scaling_execution", "scaling_verification", "scaling_notification"],
                "plan":          ["scale", "ping", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        0.65,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [
                    {"name": "deploy_verify_notify", "steps": ["deploy", "healthcheck", "notify"]},
                ],
                "note":          "New pattern (scale→ping→notify). Agent starts tracking this sequence.",
            },
            {
                "task_id":       "e-res-2",
                "task_prompt":   "Restart 'redis-cache', check health, and notify #db-team.",
                "difficulty":    "easy",
                "required_slots": ["restart_execution", "restart_verification", "restart_notification"],
                "plan":          ["restart", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": {
                    "name":  "restart_check_notify",
                    "steps": ["restart", "healthcheck", "notify"],
                },
                "reward":        0.83,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [
                    {"name": "deploy_verify_notify",  "steps": ["deploy",  "healthcheck", "notify"]},
                    {"name": "restart_check_notify",  "steps": ["restart", "healthcheck", "notify"]},
                ],
                "note":          "Agent generalises macro learning. Proposes 'restart_check_notify' "
                                 "for the restart→healthcheck→notify pattern.",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Profile 2 — Llama 3 (simulated) | Moderate learner, one wasteful call
    # -----------------------------------------------------------------------
    "Llama 3 (simulated)": {
        "easy": [
            {
                "task_id":       "e-dep-1",
                "task_prompt":   "Release 'inventory-db-proxy' v1.0.5, check health status, and notify #product.",
                "difficulty":    "easy",
                "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
                "plan":          ["deploy", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        0.63,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [],
                "note":          "Correct but slightly lower slot score from conservative LLM reasoning.",
            },
            {
                "task_id":       "e-dep-2",
                "task_prompt":   "Roll out 'auth-svc' v1.8.0, check health, and notify #ops-chat.",
                "difficulty":    "easy",
                "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
                "plan":          ["deploy", "run_tests", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        0.52,
                "turns_used":    4,
                "turns_baseline": 3,
                "macros_so_far": [],
                "note":          "Agent added 'run_tests' unnecessarily. 4 calls vs baseline 3. "
                                 "Harmless but reduces efficiency score.",
            },
            {
                "task_id":       "e-dep-3",
                "task_prompt":   "Deploy 'api-v2' v2.3.4, check health status, and notify #support.",
                "difficulty":    "easy",
                "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
                "plan":          ["deploy", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        0.63,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [],
                "note":          "Back to correct atomic plan. Macro opportunity missed again.",
            },
            {
                "task_id":       "e-dep-4",
                "task_prompt":   "Release 'image-processor' v2.7.4, check health, and notify #security.",
                "difficulty":    "easy",
                "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
                "plan":          ["deploy", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": {
                    "name":  "deploy_pipeline",
                    "steps": ["deploy", "healthcheck", "notify"],
                },
                "reward":        0.75,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [
                    {"name": "deploy_pipeline", "steps": ["deploy", "healthcheck", "notify"]},
                ],
                "note":          "Late macro recognition on episode 4. Still earns creation bonus.",
            },
            {
                "task_id":       "e-res-1",
                "task_prompt":   "Scale 'websocket-server' to 6, ping it to ensure connectivity, and notify #monitoring.",
                "difficulty":    "easy",
                "required_slots": ["scaling_execution", "scaling_verification", "scaling_notification"],
                "plan":          ["scale", "ping", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        0.63,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [
                    {"name": "deploy_pipeline", "steps": ["deploy", "healthcheck", "notify"]},
                ],
                "note":          "New slot pattern. Agent does not yet reuse deploy_pipeline (correct).",
            },
            {
                "task_id":       "e-res-2",
                "task_prompt":   "Restart 'redis-cache', check health, and notify #db-team.",
                "difficulty":    "easy",
                "required_slots": ["restart_execution", "restart_verification", "restart_notification"],
                "plan":          ["restart", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        0.63,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [
                    {"name": "deploy_pipeline", "steps": ["deploy", "healthcheck", "notify"]},
                ],
                "note":          "Agent completes correctly but doesn't propose a second macro yet.",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Profile 3 — Mistral (simulated) | Inconsistent, one harmful call
    # -----------------------------------------------------------------------
    "Mistral (simulated)": {
        "easy": [
            {
                "task_id":       "e-dep-1",
                "task_prompt":   "Release 'inventory-db-proxy' v1.0.5, check health status, and notify #product.",
                "difficulty":    "easy",
                "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
                "plan":          ["deploy", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        0.38,
                "turns_used":    2,
                "turns_baseline": 3,
                "macros_so_far": [],
                "note":          "Missed 'healthcheck' (deployment_verification slot unfilled). "
                                 "slot_ratio = 0.67 — above threshold but incomplete.",
            },
            {
                "task_id":       "e-dep-2",
                "task_prompt":   "Roll out 'auth-svc' v1.8.0, check health, and notify #ops-chat.",
                "difficulty":    "easy",
                "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
                "plan":          ["deploy", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        0.65,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [],
                "note":          "Correct full plan this time.",
            },
            {
                "task_id":       "e-dep-3",
                "task_prompt":   "Deploy 'api-v2' v2.3.4, check health status, and notify #support.",
                "difficulty":    "easy",
                "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
                "plan":          ["rollback", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        -0.10,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [],
                "note":          "⚠️ HARMFUL CALL: 'rollback' during a deployment task is destructive. "
                                 "Pipeline short-circuited. Penalty applied.",
            },
            {
                "task_id":       "e-dep-4",
                "task_prompt":   "Release 'image-processor' v2.7.4, check health, and notify #security.",
                "difficulty":    "easy",
                "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
                "plan":          ["deploy", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        0.65,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [],
                "note":          "Recovery. Correct plan, no macro proposed.",
            },
            {
                "task_id":       "e-res-1",
                "task_prompt":   "Scale 'websocket-server' to 6, ping it to ensure connectivity, and notify #monitoring.",
                "difficulty":    "easy",
                "required_slots": ["scaling_execution", "scaling_verification", "scaling_notification"],
                "plan":          ["scale", "ping", "notify"],
                "used_macro":    False,
                "macro_proposed": None,
                "reward":        0.65,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [],
                "note":          "Correct plan on new pattern.",
            },
            {
                "task_id":       "e-res-2",
                "task_prompt":   "Restart 'redis-cache', check health, and notify #db-team.",
                "difficulty":    "easy",
                "required_slots": ["restart_execution", "restart_verification", "restart_notification"],
                "plan":          ["restart", "healthcheck", "notify"],
                "used_macro":    False,
                "macro_proposed": {
                    "name":  "op_and_notify",
                    "steps": ["restart", "healthcheck", "notify"],
                },
                "reward":        0.80,
                "turns_used":    3,
                "turns_baseline": 3,
                "macros_so_far": [
                    {"name": "op_and_notify", "steps": ["restart", "healthcheck", "notify"]},
                ],
                "note":          "Recovers strongly. Proposes a macro on episode 6.",
            },
        ],
    },
}

# ===========================================================================
# SECTION 4: HTML / CSS HELPERS
# ---------------------------------------------------------------------------
# Pure rendering functions used by multiple tabs.  No Gradio imports here —
# these return plain HTML strings.
# ===========================================================================

def render_tools_html(tools: List[str], macros: List[Dict[str, Any]]) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    render_tools_html
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Render the available-tools list as colour-coded HTML.

    Atomic tools appear in a neutral badge.
    Macro tools appear in a highlighted accent badge with a ⚙ icon.

    Args:
        tools   : List of atomic tool names currently available.
        macros  : Cumulative list of accepted macro dicts (name, steps).

    Returns:
        HTML string ready for gr.HTML component.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # Build set of macro names for quick membership testing
    macro_names = {m["name"] for m in macros}

    # Accumulate badge HTML for each tool
    badges_html = ""
    for tool in tools:
        if tool in macro_names:
            # Macro badge — accent colour with gear icon
            badges_html += (
                f'<span style="display:inline-block;margin:3px 4px;padding:4px 10px;'
                f'border-radius:12px;background:#7c3aed;color:#fff;font-size:0.82em;font-weight:600;">'
                f'⚙ {tool}</span>'
            )
        else:
            # Atomic badge — neutral grey
            badges_html += (
                f'<span style="display:inline-block;margin:3px 4px;padding:4px 10px;'
                f'border-radius:12px;background:#374151;color:#d1d5db;font-size:0.82em;">'
                f'{tool}</span>'
            )

    # Add each macro at the end as its own highlighted entry
    for macro in macros:
        if macro["name"] not in tools:  # avoid double-showing if already in tools list
            badges_html += (
                f'<span style="display:inline-block;margin:3px 4px;padding:4px 10px;'
                f'border-radius:12px;background:#7c3aed;color:#fff;font-size:0.82em;font-weight:600;">'
                f'⚙ {macro["name"]}</span>'
            )

    return (
        f'<div style="padding:8px;line-height:2;">{badges_html}</div>'
        if badges_html
        else '<div style="padding:8px;color:#9ca3af;font-style:italic;">No tools available</div>'
    )


def render_plan_html(plan: List[str], macros: List[Dict[str, Any]]) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    render_plan_html
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Render the agent's proposed plan as an ordered list.

    Macro calls are annotated with a ⚙ icon and purple styling.
    Atomic calls use a standard bullet.

    Args:
        plan   : Ordered list of tool names in the agent's plan.
        macros : Accepted macros so the renderer can identify macro calls.

    Returns:
        HTML string ready for gr.HTML component.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    macro_names = {m["name"] for m in macros}  # set for O(1) lookup

    items_html = ""
    for i, tool in enumerate(plan, start=1):
        if tool in macro_names:
            items_html += (
                f'<li style="margin:6px 0;padding:6px 12px;border-radius:8px;'
                f'background:#4c1d95;color:#e9d5ff;font-weight:600;">'
                f'⚙ {i}. {tool} <span style="font-size:0.75em;opacity:0.7;">(macro)</span></li>'
            )
        else:
            items_html += (
                f'<li style="margin:6px 0;padding:6px 12px;border-radius:8px;'
                f'background:#1f2937;color:#d1d5db;">'
                f'🔧 {i}. {tool}</li>'
            )

    return (
        f'<ol style="list-style:none;padding:4px 0;margin:0;">{items_html}</ol>'
        if items_html
        else '<p style="color:#9ca3af;font-style:italic;">No plan submitted yet.</p>'
    )


def render_macro_library_html(macros: List[Dict[str, Any]]) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    render_macro_library_html
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Render the accumulated macro library as an HTML card list.

    Shows macro name, its composite steps, and a purple accent border.

    Args:
        macros : List of macro dicts with keys "name" and "steps".

    Returns:
        HTML string ready for gr.HTML component.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not macros:
        return (
            '<div style="padding:12px;color:#9ca3af;font-style:italic;text-align:center;">'
            'No macros created yet — run more episodes to see macro learning in action.'
            '</div>'
        )

    cards_html = ""
    for macro in macros:
        steps_display = " → ".join(macro["steps"])
        cards_html += (
            f'<div style="margin:6px 0;padding:10px 14px;border-left:3px solid #7c3aed;'
            f'background:#1e1b4b;border-radius:0 8px 8px 0;">'
            f'<div style="color:#c4b5fd;font-weight:700;margin-bottom:4px;">⚙ {macro["name"]}</div>'
            f'<div style="color:#a5b4fc;font-size:0.82em;">{steps_display}</div>'
            f'</div>'
        )

    return f'<div style="padding:4px 0;">{cards_html}</div>'


def render_reward_html(reward: float) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    render_reward_html
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Render the step reward as a large, colour-coded number.

    Colour scale:
        reward >= 0.75  → green  (strong performance)
        reward >= 0.50  → amber  (acceptable)
        reward >= 0.0   → orange (partial credit)
        reward < 0.0    → red    (penalty)

    Args:
        reward : Float reward value in [-0.2, 1.0].

    Returns:
        HTML string ready for gr.HTML component.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # Determine colour based on reward magnitude
    if reward >= 0.75:
        colour = "#22c55e"   # green
        label  = "Excellent"
    elif reward >= 0.50:
        colour = "#f59e0b"   # amber
        label  = "Good"
    elif reward >= 0.0:
        colour = "#f97316"   # orange
        label  = "Partial"
    else:
        colour = "#ef4444"   # red
        label  = "Penalty"

    return (
        f'<div style="text-align:center;padding:12px;">'
        f'<div style="font-size:2.8em;font-weight:800;color:{colour};">{reward:+.2f}</div>'
        f'<div style="font-size:0.85em;color:{colour};opacity:0.8;margin-top:2px;">{label}</div>'
        f'</div>'
    )


# ===========================================================================
# SECTION 5: GLOBAL CUSTOM CSS
# ---------------------------------------------------------------------------
# Injected once at the gr.Blocks level so all tabs share the same theme
# overrides.  Uses CSS custom properties where possible for easy theming.
# ===========================================================================

CUSTOM_CSS: str = """
/* ---- Global font & background tweaks ---- */
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ---- Tab label styling ---- */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95em !important;
}

/* ---- Score card highlight ---- */
.score-card {
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    background: #1f2937;
}

/* ---- Section header within a tab ---- */
.section-header {
    font-size: 1.05em;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 6px;
}

/* ---- Thinking spinner (shown during auto-play pause) ---- */
@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
.thinking-spinner {
    display: inline-block;
    width: 18px;
    height: 18px;
    border: 3px solid #7c3aed44;
    border-top-color: #7c3aed;
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
    vertical-align: middle;
    margin-right: 6px;
}

/* ---- Winner banner ---- */
.winner-banner {
    font-size: 1.6em;
    font-weight: 800;
    text-align: center;
    padding: 16px;
    border-radius: 12px;
}
"""
