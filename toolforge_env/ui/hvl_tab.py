# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tab 3 — Human vs LLM
======================

An interactive game where the user and an LLM both propose plans for the
same ToolForge task.  After both submit, scores are compared and a winner
is declared.

Flow:
    1. User configures connection (API key or ngrok) and picks a difficulty.
    2. User clicks "Start Game".
    3. A task is shown. The user types their plan (one tool per line).
    4. User clicks "Submit My Plan".
    5. The LLM plan is revealed simultaneously (placeholder for now).
    6. Both plans are scored and displayed side-by-side.
    7. A winner banner is shown.
    8. User can click "Next Task" to continue.
    9. A running scoreboard tracks cumulative Human vs LLM scores.
   10. After all tasks are exhausted, a final summary screen is shown.

The tasks come from tasks.py (via SCRIPTED_PROFILES or a direct import),
cycling through the selected difficulty group episode-by-episode.

TODO: Wire the LLM plan column to a real openai.OpenAI() call.
TODO: Wire both plans through the real ToolForge evaluation pipeline
      (server-side step() call) to get authentic scores.
TODO: After scoring, persist the run log and offer a download.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from ui.shared import (
    ATOMIC_TOOLS,
    TOOL_DESCRIPTIONS,
    render_macro_library_html,
    render_plan_html,
    render_reward_html,
)

# ---------------------------------------------------------------------------
# Provider presets re-defined locally — shared.py intentionally does not
# export these so each UI tab stays self-contained.
# ---------------------------------------------------------------------------
PROVIDER_PRESETS: Dict[str, str] = {
    "OpenAI":    "https://api.openai.com/v1",
    "Anthropic": "https://api.anthropic.com/v1",
    "Other":     "",
}

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Difficulty options available in the game tab
# ---------------------------------------------------------------------------
DIFFICULTY_OPTIONS: List[str] = ["Easy", "Medium", "Hard"]

# ---------------------------------------------------------------------------
# Tasks used by the game, grouped by difficulty.
# Each entry: {"task_id": str, "prompt": str, "required_slots": list[str],
#              "baseline_call_count": int}
# Pulling a representative subset here to keep the file self-contained.
# TODO: Replace with a live import from tasks.py once the import path
#       between the server package and UI is cleanly resolved.
# ---------------------------------------------------------------------------
GAME_TASKS: Dict[str, List[Dict[str, Any]]] = {
    "easy": [
        {
            "task_id": "e-dep-1",
            "prompt": "Release 'inventory-db-proxy' v1.0.5, check health status, and notify #product.",
            "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
            "baseline_call_count": 3,
        },
        {
            "task_id": "e-dep-2",
            "prompt": "Roll out 'auth-svc' v1.8.0, check health, and notify #ops-chat.",
            "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
            "baseline_call_count": 3,
        },
        {
            "task_id": "e-res-1",
            "prompt": "Scale 'websocket-server' to 6, ping it to ensure connectivity, and notify #monitoring.",
            "required_slots": ["scaling_execution", "scaling_verification", "scaling_notification"],
            "baseline_call_count": 3,
        },
        {
            "task_id": "e-roll-1",
            "prompt": "Rollback 'inventory-db-proxy', run core tests, and notify #logistics.",
            "required_slots": ["rollback_execution", "rollback_verification", "rollback_notification"],
            "baseline_call_count": 3,
        },
    ],
    "medium": [
        {
            "task_id": "m-tr-1",
            "prompt": "The 'data-cruncher' is lagging. Deploy the latest performance patch v4.3 and restart all workers.",
            "required_slots": ["deployment_execution", "restart_execution"],
            "baseline_call_count": 2,
        },
        {
            "task_id": "m-inc-1",
            "prompt": "Database connections are maxed out. Restart 'connection-pooler', check health. Then scale down 'background-jobs'.",
            "required_slots": ["restart_execution", "restart_verification", "scaling_execution"],
            "baseline_call_count": 3,
        },
        {
            "task_id": "m-tr-3",
            "prompt": "Rollout feature-flag to 'frontend-canary'. Check health; if it fails, rollback immediately.",
            "required_slots": ["deployment_execution", "deployment_verification", "rollback_execution"],
            "baseline_call_count": 3,
        },
    ],
    "hard": [
        {
            "task_id": "h-mig-1",
            "prompt": "Phase 1: Deploy 'cloud-connector-v1' and scale it to 20 replicas to handle initial data sync. Verify the connection health and notify #migration-hq.",
            "required_slots": ["deployment_execution", "scaling_execution", "deployment_verification", "deployment_notification"],
            "baseline_call_count": 4,
        },
        {
            "task_id": "h-sec-1",
            "prompt": "Phase 1: Deploy 'envoy-sidecar-injector' to the cluster. Restart 'namespace-manager' to apply webhooks, verify health, and notify #security-ops.",
            "required_slots": ["deployment_execution", "restart_execution", "deployment_verification", "deployment_notification"],
            "baseline_call_count": 4,
        },
    ],
}


# ===========================================================================
# SCORING HELPERS
# ===========================================================================

def _score_plan(plan_lines: str, required_slots: List[str], baseline: int) -> Dict[str, Any]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _score_plan
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Placeholder scorer for a submitted plan (human or LLM).

    TODO: Replace with a real call to the ToolForge evaluation pipeline
          (server-side step() or a direct pipeline.evaluate() call) once
          the UI→server integration is built.

    Heuristic used for the placeholder:
        - Parse tool names from lines.
        - Check how many required_slot keywords appear in ATOMIC_TOOLS.
        - Compute a rough slot_ratio from tool presence.
        - Penalise plans that are too long.

    Args:
        plan_lines      : Raw text from the user's plan textarea (one tool
                          per line).
        required_slots  : List of semantic slot names the task requires.
        baseline        : baseline_call_count for the task.

    Returns:
        Dict with keys: plan (list[str]), reward (float), score_100 (int),
        note (str).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # Parse and clean tool names from the text input
    all_tools_lower = {t.lower() for t in ATOMIC_TOOLS}
    raw_lines  = [line.strip().lower() for line in plan_lines.strip().splitlines()]
    valid_plan = [t for t in raw_lines if t in all_tools_lower]

    if not valid_plan:
        return {
            "plan":      [],
            "reward":    -0.20,
            "score_100": 0,
            "note":      "No valid tool names found in plan.",
        }

    # Slot heuristic: map slot keywords to relevant tools
    # (simplified — real scoring uses the LLM judge)
    slot_tool_hints: Dict[str, List[str]] = {
        "deployment_execution":    ["deploy", "patch"],
        "deployment_verification": ["healthcheck", "run_tests", "ping"],
        "deployment_notification": ["notify", "pagerduty_alert"],
        "scaling_execution":       ["scale"],
        "scaling_verification":    ["ping", "healthcheck"],
        "scaling_notification":    ["notify", "pagerduty_alert"],
        "restart_execution":       ["restart"],
        "restart_verification":    ["healthcheck", "ping"],
        "restart_notification":    ["notify", "pagerduty_alert"],
        "rollback_execution":      ["rollback"],
        "rollback_verification":   ["healthcheck", "run_tests"],
        "rollback_notification":   ["notify", "pagerduty_alert"],
    }

    # Count how many required slots are covered by the plan
    filled = 0
    for slot in required_slots:
        hints = slot_tool_hints.get(slot, [])
        if any(t in valid_plan for t in hints):
            filled += 1

    slot_ratio = filled / len(required_slots) if required_slots else 0.0

    # Simple reward formula for the placeholder
    if slot_ratio == 1.0:
        # Efficiency bonus for shorter plans
        eff = max(0.0, min(0.5, (baseline - len(valid_plan)) / baseline * 0.5 + 0.25))
        reward = 0.40 + eff  # base 0.40 slot score + efficiency
    elif slot_ratio >= 0.65:
        reward = 0.25 * slot_ratio
    else:
        reward = max(-0.15, -0.15 + slot_ratio * 0.23)

    reward = max(-0.20, min(1.0, reward))

    # Convert to 0–100 scale for display
    score_100 = int((reward + 0.20) / 1.20 * 100)

    note = (
        f"Filled {filled}/{len(required_slots)} required slots. "
        f"Plan length: {len(valid_plan)} (baseline: {baseline})."
    )

    return {
        "plan":      valid_plan,
        "reward":    round(reward, 2),
        "score_100": score_100,
        "note":      note,
    }


def _get_llm_placeholder_plan(task: Dict[str, Any]) -> List[str]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _get_llm_placeholder_plan
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Return a hardcoded "LLM" plan for the placeholder implementation.

    TODO: Replace with a real openai.OpenAI() call using the user's
          configured credentials.  The call should send the task prompt
          and the ToolForgeAction schema, then parse the JSON response.

    Args:
        task : Task dict with keys "prompt", "required_slots",
               "baseline_call_count".

    Returns:
        List of tool name strings representing the LLM's proposed plan.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # Heuristic: pick canonical tools for the first required slot pattern
    slot_to_plan: Dict[str, List[str]] = {
        "deployment_execution":  ["deploy", "healthcheck", "notify"],
        "scaling_execution":     ["scale", "ping", "notify"],
        "restart_execution":     ["restart", "healthcheck", "notify"],
        "rollback_execution":    ["rollback", "run_tests", "notify"],
    }

    required = task.get("required_slots", [])
    for slot in required:
        if slot in slot_to_plan:
            return slot_to_plan[slot]

    # Fallback
    return ["deploy", "healthcheck", "notify"]


# ===========================================================================
# EVENT HANDLERS
# ===========================================================================

def on_hvl_mode_change(mode: str) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_hvl_mode_change
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Show/hide API key vs ngrok connection sections.

    Args:
        mode : Connection mode string.

    Returns:
        Tuple of gr.update dicts for the two connection sections.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    is_api = (mode == "API Key (OpenAI-compatible)")
    return gr.update(visible=is_api), gr.update(visible=not is_api)


def on_hvl_provider_change(provider: str) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_hvl_provider_change
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Pre-fill the Base URL field when provider changes.

    Args:
        provider : Selected provider name.

    Returns:
        Base URL string.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    return PROVIDER_PRESETS.get(provider, "")


def on_start_game(difficulty: str) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_start_game
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Initialise the game state when the user clicks "Start Game".

    Loads the task list for the selected difficulty and shows the first
    task.  Resets the human and LLM scoreboards.

    Args:
        difficulty : Difficulty string ("Easy", "Medium", "Hard").

    Returns:
        Tuple updating: game_area visibility, task display, available
        tool list, scoreboard, and the hidden state variables.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    logger.info("HvL game started | difficulty=%s", difficulty)

    diff_key  = difficulty.lower()
    task_list = GAME_TASKS.get(diff_key, [])

    if not task_list:
        # No tasks available — show a warning
        return (
            gr.update(visible=True),
            "No tasks found for this difficulty. TODO: Add more task groups.",
            "<div style='color:#9ca3af;'>No tools</div>",
            "You: 0  |  LLM: 0",
            gr.update(visible=False),   # results row hidden
            "",                         # llm plan placeholder
            0,                          # task index
            task_list,                  # tasks state
            0,                          # human cumulative
            0,                          # llm cumulative
        )

    first_task = task_list[0]
    tools_html = _render_tool_checklist()

    return (
        gr.update(visible=True),                    # show game area
        first_task["prompt"],                        # task prompt
        tools_html,                                  # tool reference
        "You: 0  |  LLM: 0",                        # scoreboard
        gr.update(visible=False),                    # results hidden
        "Waiting for your submission…",              # llm plan placeholder
        0,                                           # task_index state
        task_list,                                   # tasks_state
        0,                                           # human_score_state
        0,                                           # llm_score_state
    )


def on_submit_human_plan(
    human_plan_text: str,
    task_index: int,
    tasks_state: List[Dict[str, Any]],
    human_score_state: int,
    llm_score_state: int,
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_submit_human_plan
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Called when the user clicks "Submit My Plan".

    Simultaneously:
        - Scores the human's plan using the placeholder scorer.
        - Generates + scores the LLM's plan (placeholder).
        - Reveals both plans and scores.
        - Updates cumulative scoreboard.
        - Shows a winner banner for this round.

    Args:
        human_plan_text  : Raw text from the human's plan textarea.
        task_index       : Current task index (gr.State).
        tasks_state      : List of task dicts (gr.State).
        human_score_state: Running human cumulative score (gr.State).
        llm_score_state  : Running LLM cumulative score (gr.State).

    Returns:
        Large tuple updating all result + state components.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not tasks_state or task_index >= len(tasks_state):
        # Guard against bad state
        return _blank_results() + (task_index, tasks_state, human_score_state, llm_score_state)

    task = tasks_state[task_index]

    # --- Score human plan ---
    human_result = _score_plan(
        human_plan_text,
        task["required_slots"],
        task["baseline_call_count"],
    )

    # --- Get + score LLM plan ---
    llm_plan    = _get_llm_placeholder_plan(task)
    llm_plan_text = "\n".join(llm_plan)
    llm_result  = _score_plan(
        llm_plan_text,
        task["required_slots"],
        task["baseline_call_count"],
    )

    # --- Cumulative scores ---
    new_human_score = human_score_state + human_result["score_100"]
    new_llm_score   = llm_score_state   + llm_result["score_100"]
    scoreboard_text = f"You: {new_human_score}  |  LLM: {new_llm_score}"

    # --- Winner for this round ---
    if human_result["score_100"] > llm_result["score_100"]:
        winner_html = (
            '<div class="winner-banner" style="background:#14532d;color:#4ade80;">🏆 You Win This Round!</div>'
        )
    elif llm_result["score_100"] > human_result["score_100"]:
        winner_html = (
            '<div class="winner-banner" style="background:#450a0a;color:#f87171;">🤖 LLM Wins This Round!</div>'
        )
    else:
        winner_html = (
            '<div class="winner-banner" style="background:#1e3a5f;color:#93c5fd;">🤝 It\'s a Tie!</div>'
        )

    # --- Human plan display ---
    human_plan_html = render_plan_html(human_result["plan"], [])
    human_score_html = (
        f'<div style="text-align:center;font-size:2em;font-weight:800;color:#4ade80;">'
        f'{human_result["score_100"]}<span style="font-size:0.5em;opacity:0.6;">/100</span></div>'
    )

    # --- LLM plan display ---
    llm_plan_html_rendered = render_plan_html(llm_result["plan"], [])
    llm_score_html = (
        f'<div style="text-align:center;font-size:2em;font-weight:800;color:#60a5fa;">'
        f'{llm_result["score_100"]}<span style="font-size:0.5em;opacity:0.6;">/100</span></div>'
    )

    # --- Slot breakdown table (placeholder) ---
    slot_rows = ""
    for slot in task["required_slots"]:
        human_filled = "✅" if slot in _infer_filled_slots(human_result["plan"], slot) else "❌"
        llm_filled   = "✅" if slot in _infer_filled_slots(llm_result["plan"],   slot) else "❌"
        slot_rows += f"<tr><td>{slot}</td><td style='text-align:center'>{human_filled}</td><td style='text-align:center'>{llm_filled}</td></tr>"

    slot_table_html = (
        f'<table style="width:100%;border-collapse:collapse;font-size:0.85em;">'
        f'<thead><tr>'
        f'<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #374151">Slot</th>'
        f'<th style="padding:4px 8px;border-bottom:1px solid #374151">You</th>'
        f'<th style="padding:4px 8px;border-bottom:1px solid #374151">LLM</th>'
        f'</tr></thead><tbody>{slot_rows}</tbody></table>'
    )

    return (
        gr.update(visible=True),       # show results row
        human_plan_html,               # human plan HTML
        human_score_html,              # human score HTML
        llm_plan_html_rendered,        # llm plan HTML
        llm_score_html,                # llm score HTML
        winner_html,                   # winner banner HTML
        slot_table_html,               # slot breakdown table
        scoreboard_text,               # updated scoreboard
        task_index,                    # task index unchanged (Next btn advances)
        tasks_state,                   # tasks unchanged
        new_human_score,               # updated human cumulative
        new_llm_score,                 # updated llm cumulative
    )


def on_next_task(
    task_index: int,
    tasks_state: List[Dict[str, Any]],
    human_score_state: int,
    llm_score_state: int,
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_next_task
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Advance to the next task in the game sequence.

    If all tasks are exhausted, show the final summary screen.

    Args:
        task_index        : Current task index.
        tasks_state       : List of task dicts.
        human_score_state : Running human cumulative score.
        llm_score_state   : Running LLM cumulative score.

    Returns:
        Tuple updating the task prompt, results visibility, and state.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    new_index = task_index + 1

    if new_index >= len(tasks_state):
        # All tasks done — determine overall winner
        if human_score_state > llm_score_state:
            final_msg = f"🏆 **You Win!** Final: You {human_score_state} – LLM {llm_score_state}"
        elif llm_score_state > human_score_state:
            final_msg = f"🤖 **LLM Wins!** Final: You {human_score_state} – LLM {llm_score_state}"
        else:
            final_msg = f"🤝 **It's a Tie!** Final: You {human_score_state} – LLM {llm_score_state}"

        return (
            f"🎉 Game Over! {final_msg}",      # task prompt shows final result
            "",                                 # clear human plan input
            "Game over — no more tasks.",       # llm placeholder
            gr.update(visible=False),           # hide results row
            new_index,                          # task index (past end)
            tasks_state,
            human_score_state,
            llm_score_state,
        )

    next_task = tasks_state[new_index]
    return (
        next_task["prompt"],           # new task prompt
        "",                            # clear human plan input
        "Waiting for your submission…",
        gr.update(visible=False),      # hide previous results
        new_index,
        tasks_state,
        human_score_state,
        llm_score_state,
    )


# ===========================================================================
# INTERNAL HELPERS
# ===========================================================================

def _render_tool_checklist() -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _render_tool_checklist
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Render the available tool list as a reference HTML card, shown
    alongside the human's plan input area.

    Returns:
        HTML string with tool name + description pairs.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    rows = ""
    for tool in ATOMIC_TOOLS:
        desc = TOOL_DESCRIPTIONS.get(tool, "")
        rows += (
            f'<div style="margin:4px 0;padding:4px 10px;border-radius:6px;background:#1f2937;">'
            f'<code style="color:#a78bfa;">{tool}</code>'
            f'<span style="color:#9ca3af;font-size:0.82em;margin-left:8px;">{desc}</span>'
            f'</div>'
        )
    return f'<div style="padding:4px 0;">{rows}</div>'


def _infer_filled_slots(plan: List[str], slot: str) -> List[str]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _infer_filled_slots
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Very lightweight helper: returns the slot name if any plan tool
    is a plausible hint for that slot — used only for the breakdown
    table in the Human vs LLM tab.

    Args:
        plan : List of tool names.
        slot : Slot name to check.

    Returns:
        List containing the slot name if it is (likely) filled,
        otherwise an empty list.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    hints: Dict[str, List[str]] = {
        "deployment_execution":    ["deploy", "patch"],
        "deployment_verification": ["healthcheck", "run_tests", "ping"],
        "deployment_notification": ["notify", "pagerduty_alert"],
        "scaling_execution":       ["scale"],
        "scaling_verification":    ["ping", "healthcheck"],
        "scaling_notification":    ["notify", "pagerduty_alert"],
        "restart_execution":       ["restart"],
        "restart_verification":    ["healthcheck", "ping"],
        "restart_notification":    ["notify", "pagerduty_alert"],
        "rollback_execution":      ["rollback"],
        "rollback_verification":   ["healthcheck", "run_tests"],
        "rollback_notification":   ["notify", "pagerduty_alert"],
    }
    tool_hints = hints.get(slot, [])
    return [slot] if any(t in plan for t in tool_hints) else []


def _blank_results() -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _blank_results
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Return blank values for all result components (before a submission).

    Returns:
        Tuple of blank display values.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    return (
        gr.update(visible=False),   # results row hidden
        "<p>—</p>",                 # human plan
        "<p>—</p>",                 # human score
        "<p>—</p>",                 # llm plan
        "<p>—</p>",                 # llm score
        "",                         # winner banner
        "",                         # slot table
        "You: 0  |  LLM: 0",       # scoreboard
    )


# ===========================================================================
# TAB BUILDER
# ===========================================================================

def build_hvl_tab() -> gr.Tab:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    build_hvl_tab
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Construct and return the "Human vs LLM" gr.Tab component.

    Returns:
        A configured gr.Tab component.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    with gr.Tab("🎮 Human vs LLM") as tab:

        # -------------------------------------------------------------------
        # HEADER
        # -------------------------------------------------------------------
        gr.Markdown(
            """
            # Human vs LLM

            ### Can you out-plan the model?

            You and an LLM both propose a tool plan for the same ToolForge task.
            Both plans are scored by the same evaluation pipeline.
            The one with the higher score wins the round!
            """
        )

        gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

        # -------------------------------------------------------------------
        # CONNECTION SECTION (duplicated cleanly from BYOA tab)
        # -------------------------------------------------------------------
        gr.Markdown("### Connect Your LLM")

        hvl_connection_mode = gr.Radio(
            choices=["API Key (OpenAI-compatible)", "Local Model via ngrok"],
            value="API Key (OpenAI-compatible)",
            label="Connection Mode",
        )

        # API key sub-section
        with gr.Group(visible=True) as hvl_api_section:
            with gr.Row():
                hvl_provider = gr.Dropdown(
                    choices=list(PROVIDER_PRESETS.keys()),
                    value="OpenAI",
                    label="Provider",
                    scale=1,
                )
                hvl_base_url = gr.Textbox(
                    value=PROVIDER_PRESETS["OpenAI"],
                    label="Base URL",
                    scale=3,
                )
            with gr.Row():
                hvl_model_name = gr.Textbox(
                    value="gpt-4o",
                    label="Model Name",
                    scale=2,
                )
                hvl_api_key = gr.Textbox(
                    value="",
                    label="API Key  (used only in your session — never stored)",
                    placeholder="sk-...",
                    type="password",
                    scale=3,
                )

        # ngrok sub-section
        with gr.Group(visible=False) as hvl_ngrok_section:
            hvl_ngrok_url = gr.Textbox(
                value="",
                label="ngrok Tunnel URL",
                placeholder="https://xxxx-xx-xx.ngrok-free.app/v1",
            )

        gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

        # -------------------------------------------------------------------
        # DIFFICULTY + START
        # -------------------------------------------------------------------
        with gr.Row():
            hvl_difficulty = gr.Dropdown(
                choices=DIFFICULTY_OPTIONS,
                value="Easy",
                label="Difficulty",
                scale=1,
            )
            start_btn = gr.Button("🎮 Start Game", variant="primary", scale=2)

        # -------------------------------------------------------------------
        # GAME AREA (hidden until Start Game is clicked)
        # -------------------------------------------------------------------
        with gr.Group(visible=False) as game_area:

            # Running scoreboard at the top of the game area
            scoreboard_box = gr.Textbox(
                label="📊 Scoreboard",
                value="You: 0  |  LLM: 0",
                interactive=False,
            )

            gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

            # Current task display
            task_display = gr.Textbox(
                label="📋 Current Task",
                value="",
                lines=4,
                interactive=False,
            )

            # Two-column layout: human plan (left) vs LLM plan (right)
            with gr.Row():

                # ===========================================================
                # LEFT — Human's Plan
                # ===========================================================
                with gr.Column(scale=1):
                    gr.Markdown("### 🧑 Your Plan")

                    # Tool reference card
                    gr.Markdown("**Available Tools** *(enter names below)*")
                    tool_reference_html = gr.HTML(
                        value=_render_tool_checklist()
                    )

                    # Human plan input: one tool name per line
                    human_plan_input = gr.Textbox(
                        label="Enter your plan (one tool name per line)",
                        placeholder="deploy\nhealthcheck\nnotify",
                        lines=6,
                    )

                    submit_plan_btn = gr.Button(
                        "Submit My Plan ▶",
                        variant="primary",
                    )

                # ===========================================================
                # RIGHT — LLM's Plan
                # ===========================================================
                with gr.Column(scale=1):
                    gr.Markdown("### 🤖 LLM Plan")

                    llm_plan_placeholder = gr.Textbox(
                        label="LLM's plan (revealed after you submit)",
                        value="Waiting for your submission…",
                        lines=6,
                        interactive=False,
                    )

            # ---------------------------------------------------------------
            # RESULTS ROW (hidden until both plans submitted)
            # ---------------------------------------------------------------
            with gr.Group(visible=False) as results_row:

                gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")
                gr.Markdown("### Results")

                # Winner banner (full width)
                winner_banner_html = gr.HTML(value="")

                # Score cards side by side
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**🧑 Your Score**")
                        human_score_html = gr.HTML(
                            value='<div style="text-align:center;font-size:2em;font-weight:800;">—</div>'
                        )
                        gr.Markdown("**Your Plan:**")
                        human_plan_result_html = gr.HTML(value="<p>—</p>")

                    with gr.Column(scale=1):
                        gr.Markdown("**🤖 LLM Score**")
                        llm_score_html = gr.HTML(
                            value='<div style="text-align:center;font-size:2em;font-weight:800;">—</div>'
                        )
                        gr.Markdown("**LLM Plan:**")
                        llm_plan_result_html = gr.HTML(value="<p>—</p>")

                # Slot breakdown table
                gr.Markdown("**Slot Breakdown:**")
                slot_table_html = gr.HTML(value="")

                # Next task button
                next_task_btn = gr.Button("Next Task ▶", variant="secondary")

        # -------------------------------------------------------------------
        # HIDDEN STATE
        # -------------------------------------------------------------------

        # 0-based index into tasks_state for the current task
        task_index_state = gr.State(value=0)

        # Full list of task dicts for the active difficulty
        tasks_state = gr.State(value=[])

        # Cumulative score for the human player (sum of score_100 per round)
        human_score_state = gr.State(value=0)

        # Cumulative score for the LLM (sum of score_100 per round)
        llm_score_state = gr.State(value=0)

        # -------------------------------------------------------------------
        # EVENT WIRING
        # -------------------------------------------------------------------

        # Connection mode toggle
        hvl_connection_mode.change(
            fn=on_hvl_mode_change,
            inputs=[hvl_connection_mode],
            outputs=[hvl_api_section, hvl_ngrok_section],
        )

        # Provider pre-fill
        hvl_provider.change(
            fn=on_hvl_provider_change,
            inputs=[hvl_provider],
            outputs=[hvl_base_url],
        )

        # Start Game → show game area, load first task
        start_btn.click(
            fn=on_start_game,
            inputs=[hvl_difficulty],
            outputs=[
                game_area,            # make visible
                task_display,
                tool_reference_html,
                scoreboard_box,
                results_row,          # hide results
                llm_plan_placeholder,
                task_index_state,
                tasks_state,
                human_score_state,
                llm_score_state,
            ],
        )

        # Submit human plan → score both sides, reveal results
        submit_plan_btn.click(
            fn=on_submit_human_plan,
            inputs=[
                human_plan_input,
                task_index_state,
                tasks_state,
                human_score_state,
                llm_score_state,
            ],
            outputs=[
                results_row,             # make visible
                human_plan_result_html,
                human_score_html,
                llm_plan_result_html,
                llm_score_html,
                winner_banner_html,
                slot_table_html,
                scoreboard_box,
                task_index_state,
                tasks_state,
                human_score_state,
                llm_score_state,
            ],
        )

        # Next Task → advance task index, hide results
        next_task_btn.click(
            fn=on_next_task,
            inputs=[task_index_state, tasks_state, human_score_state, llm_score_state],
            outputs=[
                task_display,
                human_plan_input,
                llm_plan_placeholder,
                results_row,
                task_index_state,
                tasks_state,
                human_score_state,
                llm_score_state,
            ],
        )

    return tab
