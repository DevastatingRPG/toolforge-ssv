# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ui/shared.py — Shared data, constants, and HTML renderers for the ToolForge Gradio UI.

Import pattern (runs from toolforge_env/ dir — no toolforge_env prefix):
    from ui.shared import SCRIPTED_PROFILES, render_plan_html, ...

What lives here:
    SYSTEM_PROMPT       : Default agent system prompt (editable in BYOA tab)
    ATOMIC_TOOLS        : Tool names (mirror of server/tools.py — no server import needed)
    TOOL_DESCRIPTIONS   : Human-readable tool descriptions for HvL tool picker
    SCRIPTED_PROFILES   : Pre-scripted agent plans per model × difficulty × episode × step
    render_*            : Pure HTML-rendering functions (no Gradio imports)
    CUSTOM_CSS          : Global CSS string injected into gr.Blocks

SCRIPTED_PROFILES structure:
    model_label (str)
      → difficulty (str: "easy" | "medium" | "hard")
        → List[Episode]

    Episode = {
        "episode_id"    : str   — passed to env_reset() as task_id
        "episode_label" : str   — human-readable name shown in progress bar
        "steps"         : List[StepScript]
    }

    StepScript = {
        "plan"           : List[str]        — ordered tool names the agent proposes
        "macro_proposed" : dict | None      — {"name": str, "steps": List[str]} or None
        "note"           : str              — narrative annotation shown in the UI
    }

Note: rewards, task prompts, and macro state are NOT stored here.
They come from the real environment via env_client.env_step() / env_reset().
The scripted profiles only prescribe which plan each model "would" submit.
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
# prompt.  Users may edit the strategy section.  The ⚠️ line and everything
# below it must NOT be removed — it will break the server-side JSON parser.
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
    - If task.required_slots has length N, prefer a plan with at least N entries.

    Naming:
    - Use short snake_case names that describe the operation pattern.

    ⚠️  IMPORTANT — DO NOT EDIT BELOW THIS LINE  ⚠️
    Respond ONLY with valid JSON matching the ToolForgeAction schema.
    No markdown, no explanation, no commentary outside the JSON object.
    """
).strip()

# ===========================================================================
# SECTION 2: TOOL INVENTORY
# ---------------------------------------------------------------------------
# Mirrors server/tools.py without importing from the server layer.
# UI files reference this list instead of importing build_atomic_tools().
# ===========================================================================

# Ordered list of all atomic tool names
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

# Human-readable descriptions for each atomic tool (HvL tab tool picker)
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
# Each profile prescribes ONLY the plan (and optional macro_proposal) for
# each step.  Rewards, task prompts, and macro state come from the real env.
#
# Episode "episode_id" must match a valid task_id accepted by env_reset():
#   "easy-deployment-sprints"   "easy-resource-management"  "easy-rollback-drills"
#   "medium-traffic-readiness"  "medium-incident-response"  ...
#   "hard-project-legacy-migration"  "hard-project-zero-trust"
#
# The number of steps in a scripted episode should match the number of tasks
# in the corresponding task group in tasks.py.  If the env returns done=True
# before the scripted steps are exhausted, the UI will stop stepping.
# ===========================================================================

SCRIPTED_PROFILES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {

    # -----------------------------------------------------------------------
    # GPT-4o (simulated) — Fast learner: proposes macro by step 3, reuses by 4
    # -----------------------------------------------------------------------
    "GPT-4o (simulated)": {
        "easy": [
            # ---- Episode 1: easy-deployment-sprints (4 tasks) -------------
            {
                "episode_id":    "easy-deployment-sprints",
                "episode_label": "Deployment Sprints",
                "steps": [
                    {
                        "plan":           ["deploy", "healthcheck", "notify"],
                        "macro_proposed": None,
                        "note":           "Agent executes the canonical 3-step deployment pipeline correctly.",
                    },
                    {
                        "plan":           ["deploy", "healthcheck", "notify"],
                        "macro_proposed": None,
                        "note":           "Same pattern repeated. Agent internally tracks this sequence.",
                    },
                    {
                        "plan":           ["deploy", "healthcheck", "notify"],
                        "macro_proposed": {
                            "name":  "deploy_verify_notify",
                            "steps": ["deploy", "healthcheck", "notify"],
                        },
                        "note": (
                            "Pattern seen 3 times. Agent proposes macro 'deploy_verify_notify'. "
                            "Macro creation bonus applied on top of slot score."
                        ),
                    },
                    {
                        "plan":           ["deploy_verify_notify"],
                        "macro_proposed": None,
                        "note": (
                            "Agent reuses 'deploy_verify_notify'. 2 turns saved vs baseline. "
                            "Macro usage bonus and efficiency score both applied."
                        ),
                    },
                ],
            },

            # ---- Episode 2: easy-resource-management (macros reset) -------
            {
                "episode_id":    "easy-resource-management",
                "episode_label": "Resource Management",
                "steps": [
                    {
                        "plan":           ["scale", "ping", "notify"],
                        "macro_proposed": None,
                        "note":           "New episode — macros reset by env. New pattern (scale→ping→notify).",
                    },
                    {
                        "plan":           ["restart", "healthcheck", "notify"],
                        "macro_proposed": {
                            "name":  "restart_check_notify",
                            "steps": ["restart", "healthcheck", "notify"],
                        },
                        "note": "Agent proposes 'restart_check_notify' for the restart→healthcheck→notify pattern.",
                    },
                ],
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Llama 3 (simulated) — Moderate: one unnecessary call, late macro
    # -----------------------------------------------------------------------
    "Llama 3 (simulated)": {
        "easy": [
            # ---- Episode 1: easy-deployment-sprints -----------------------
            {
                "episode_id":    "easy-deployment-sprints",
                "episode_label": "Deployment Sprints",
                "steps": [
                    {
                        "plan":           ["deploy", "healthcheck", "notify"],
                        "macro_proposed": None,
                        "note":           "Correct but slightly lower slot score from conservative LLM reasoning.",
                    },
                    {
                        "plan":           ["deploy", "run_tests", "healthcheck", "notify"],
                        "macro_proposed": None,
                        "note": (
                            "Agent added 'run_tests' unnecessarily. 4 calls vs baseline 3. "
                            "Harmless but reduces efficiency score."
                        ),
                    },
                    {
                        "plan":           ["deploy", "healthcheck", "notify"],
                        "macro_proposed": None,
                        "note":           "Back to correct atomic plan. Macro opportunity missed again.",
                    },
                    {
                        "plan":           ["deploy", "healthcheck", "notify"],
                        "macro_proposed": {
                            "name":  "deploy_pipeline",
                            "steps": ["deploy", "healthcheck", "notify"],
                        },
                        "note":           "Late macro recognition on step 4. Still earns creation bonus.",
                    },
                ],
            },

            # ---- Episode 2: easy-resource-management ----------------------
            {
                "episode_id":    "easy-resource-management",
                "episode_label": "Resource Management",
                "steps": [
                    {
                        "plan":           ["scale", "ping", "notify"],
                        "macro_proposed": None,
                        "note":           "New episode — macros reset. New slot pattern.",
                    },
                    {
                        "plan":           ["restart", "healthcheck", "notify"],
                        "macro_proposed": None,
                        "note":           "Agent completes correctly but doesn't propose a second macro yet.",
                    },
                ],
            },
        ],
    },

    # -----------------------------------------------------------------------
    # Mistral (simulated) — Inconsistent: incomplete plan, one harmful call
    # -----------------------------------------------------------------------
    "Mistral (simulated)": {
        "easy": [
            # ---- Episode 1: easy-deployment-sprints -----------------------
            {
                "episode_id":    "easy-deployment-sprints",
                "episode_label": "Deployment Sprints",
                "steps": [
                    {
                        "plan":           ["deploy", "notify"],
                        "macro_proposed": None,
                        "note": (
                            "Missed 'healthcheck' (deployment_verification slot unfilled). "
                            "slot_ratio = 0.67 — above threshold but incomplete."
                        ),
                    },
                    {
                        "plan":           ["deploy", "healthcheck", "notify"],
                        "macro_proposed": None,
                        "note":           "Correct full plan this time.",
                    },
                    {
                        "plan":           ["rollback", "healthcheck", "notify"],
                        "macro_proposed": None,
                        "note": (
                            "⚠️ HARMFUL CALL: 'rollback' during a deployment task is destructive. "
                            "Pipeline short-circuited. Penalty applied by env."
                        ),
                    },
                    {
                        "plan":           ["deploy", "healthcheck", "notify"],
                        "macro_proposed": None,
                        "note":           "Recovery. Correct plan, no macro proposed.",
                    },
                ],
            },

            # ---- Episode 2: easy-resource-management ----------------------
            {
                "episode_id":    "easy-resource-management",
                "episode_label": "Resource Management",
                "steps": [
                    {
                        "plan":           ["scale", "ping", "notify"],
                        "macro_proposed": None,
                        "note":           "New episode — macros reset. Correct plan on new pattern.",
                    },
                    {
                        "plan":           ["restart", "healthcheck", "notify"],
                        "macro_proposed": {
                            "name":  "op_and_notify",
                            "steps": ["restart", "healthcheck", "notify"],
                        },
                        "note":           "Recovers strongly. Proposes a macro on step 2 of episode 2.",
                    },
                ],
            },
        ],
    },
}

# ===========================================================================
# SECTION 4: HTML RENDERERS
# ---------------------------------------------------------------------------
# Pure functions that produce HTML strings for gr.HTML components.
# No Gradio imports here — these are safe to call from any thread.
# ===========================================================================

def render_tools_html(
    atomic_tool_names: List[str],
    macros: List[Dict[str, Any]],
) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    render_tools_html
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Render the available-tools list as colour-coded HTML badges.

    Atomic tools   → neutral grey badge.
    Macro tools    → purple badge with ⚙ icon + step preview inline.

    Args:
        atomic_tool_names : Names of atomic tools (from ATOMIC_TOOLS or env obs).
        macros            : Accepted macro dicts with "name" and "steps" keys.
                            Comes from env_client.extract_macros().

    Returns:
        HTML string ready for gr.HTML.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # Atomic badges
    badges_html = ""
    for tool in atomic_tool_names:
        badges_html += (
            f'<span style="display:inline-block;margin:3px 4px;padding:4px 10px;'
            f'border-radius:12px;background:#374151;color:#d1d5db;font-size:0.82em;">'
            f'{tool}</span>'
        )

    # Macro section
    if macros:
        badges_html += (
            '<div style="margin-top:10px;margin-bottom:2px;'
            'color:#a78bfa;font-size:0.78em;font-weight:700;letter-spacing:0.03em;">'
            '⚙ Macros in library:</div>'
        )
        for macro in macros:
            steps_preview = " → ".join(macro.get("steps", []))
            badges_html += (
                f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                f'<span style="padding:4px 12px;border-radius:12px;background:#7c3aed;'
                f'color:#fff;font-size:0.82em;font-weight:600;white-space:nowrap;">⚙ {macro["name"]}</span>'
                f'<span style="color:#9ca3af;font-size:0.75em;">{steps_preview}</span>'
                f'</div>'
            )

    return (
        f'<div style="padding:8px;line-height:2.2;">{badges_html}</div>'
        if badges_html
        else '<div style="padding:8px;color:#9ca3af;font-style:italic;">No tools available</div>'
    )


def render_plan_html(
    plan:           List[str],
    macros:         List[Dict[str, Any]],
    macro_proposal: Optional[Dict[str, Any]] = None,
) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    render_plan_html
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Render the agent's proposed plan as a numbered list.

    Macro calls  → purple row with ⚙ icon.
    Atomic calls → dark row with 🔧 icon.

    If macro_proposal is provided (step where a new macro is introduced),
    a dashed annotation card is appended showing the macro name and steps.

    Args:
        plan           : Ordered list of tool name strings.
        macros         : Accepted macros list (name, steps) — used to identify macro calls.
        macro_proposal : Dict with "name" and "steps" if the agent proposes a new macro
                         this step, else None.

    Returns:
        HTML string ready for gr.HTML.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # Set of accepted macro names for O(1) lookup
    macro_names = {m["name"] for m in macros}

    # Build plan rows
    items_html = ""
    for i, tool in enumerate(plan, start=1):
        if tool in macro_names:
            # Macro call — purple accent
            items_html += (
                f'<li style="margin:6px 0;padding:6px 12px;border-radius:8px;'
                f'background:#4c1d95;color:#e9d5ff;font-weight:600;">'
                f'⚙ {i}. {tool} <span style="font-size:0.75em;opacity:0.7;">(macro)</span></li>'
            )
        else:
            # Atomic call — neutral dark
            items_html += (
                f'<li style="margin:6px 0;padding:6px 12px;border-radius:8px;'
                f'background:#1f2937;color:#d1d5db;">'
                f'🔧 {i}. {tool}</li>'
            )

    plan_block = (
        f'<ol style="list-style:none;padding:4px 0;margin:0;">{items_html}</ol>'
        if items_html
        else '<p style="color:#9ca3af;font-style:italic;">No plan submitted yet.</p>'
    )

    # Macro introduction annotation (shown on the step where a new macro is proposed)
    if macro_proposal:
        steps_str  = " → ".join(macro_proposal.get("steps", []))
        macro_name = macro_proposal.get("name", "?")
        annotation = (
            f'<div style="margin-top:10px;padding:10px 14px;'
            f'border:2px dashed #7c3aed;border-radius:8px;background:#1e1b4b;">'
            f'<div style="color:#c4b5fd;font-weight:700;font-size:0.88em;margin-bottom:4px;">'
            f'📦 New macro introduced: '
            f'<code style="color:#e9d5ff;background:#2e1065;padding:2px 6px;border-radius:4px;">'
            f'{macro_name}</code>'
            f'</div>'
            f'<div style="color:#a5b4fc;font-size:0.80em;">{steps_str}</div>'
            f'</div>'
        )
        return plan_block + annotation

    return plan_block


def render_macro_library_html(macros: List[Dict[str, Any]]) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    render_macro_library_html
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Render the accumulated macro library as a card list.

    Args:
        macros : List of macro dicts with "name" and "steps" keys.

    Returns:
        HTML string ready for gr.HTML.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not macros:
        return (
            '<div style="padding:12px;color:#9ca3af;font-style:italic;text-align:center;">'
            'No macros created yet — run more steps to see macro learning in action.'
            '</div>'
        )

    cards_html = ""
    for macro in macros:
        steps_display = " → ".join(macro.get("steps", []))
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
    Render the step reward as a large colour-coded number.

    Colour scale:
        >= 0.75  → green  (Excellent)
        >= 0.50  → amber  (Good)
        >= 0.0   → orange (Partial)
        < 0.0    → red    (Penalty)

    Args:
        reward : Float in [-0.2, 1.0].

    Returns:
        HTML string ready for gr.HTML.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if reward >= 0.75:
        colour = "#22c55e"
    elif reward >= 0.50:
        colour = "#f59e0b"
    elif reward >= 0.0:
        colour = "#f97316"
    else:
        colour = "#ef4444"

    return (
        f'<div style="text-align:center;padding:12px;">'
        f'<div style="font-size:2.8em;font-weight:800;color:{colour};">{reward:+.2f}</div>'
        f'</div>'
    )


# ===========================================================================
# SECTION 5: GLOBAL CUSTOM CSS
# ===========================================================================

CUSTOM_CSS: str = """
/* ---- Global font tweaks ---- */
body, .gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ---- Tab label styling ---- */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95em !important;
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
