# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ui/demo_tab.py — Tab 1: Demo Mode
===================================

A read-only simulation tab.  The agent plans are scripted (from shared.py),
but ALL environment interactions are real:
    - "Run Simulation" calls env_reset() on the running ToolForge server.
    - "Next Step" sends the scripted plan via env_step() and displays the
      actual reward, task prompt, and macro state returned by the server.

This means the env is the source of truth for:
    - Task prompts
    - Rewards
    - Available tools (including approved macros)
    - Episode done state

The scripted profiles (SCRIPTED_PROFILES) only define WHAT PLAN each
simulated model would choose at each step.

Navigation model:
    Step    = one task prompt within an episode (one env.step() call).
    Episode = one task group (one env.reset() call).
    "Next Step ▶"    — advance within episode (calls env.step()).
    "🔄 Next Episode" — at last step; calls env.reset() with next episode_id.
    "◀ Previous Step" — steps back through already-executed frames (no re-calling env).

Import pattern (runs from toolforge_env/ dir):
    from ui.demo_tab import build_demo_tab
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from ui.shared import (
    ATOMIC_TOOLS,
    SCRIPTED_PROFILES,
    render_macro_library_html,
    render_plan_html,
    render_reward_html,
    render_tools_html,
)
from ui.env_client import (
    DEFAULT_ENV_URL,
    check_env_health,
    env_reset,
    env_step,
    extract_macros,
    parse_task_from_obs,
    parse_tools_from_obs,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------

# Available model display names — correspond to keys in SCRIPTED_PROFILES
MODEL_OPTIONS: List[str] = [
    "GPT-4o (simulated)",
    "Llama 3 (simulated)",
    "Mistral (simulated)",
]

# Difficulty options (only Easy has full scripted profiles for now)
DIFFICULTY_OPTIONS: List[str] = ["Easy"]

DISCLAIMER_TEXT: str = (
    "ℹ️  Agent plans are pre-scripted behavioral profiles. "
    "Rewards and task prompts come from the real ToolForge environment server."
)

# Note shown below Next button at episode boundary
_EPISODE_END_NOTE: str = (
    '<em style="color:#a78bfa;font-size:0.85em;">'
    '📌 Last step of this episode — clicking again starts next episode and resets macros.'
    '</em>'
)
_ALL_DONE_NOTE: str = (
    '<em style="color:#f59e0b;font-size:0.85em;">'
    '🎉 All scripted episodes complete — clicking restarts from the beginning.'
    '</em>'
)


# ===========================================================================
# INTERNAL HELPERS
# ===========================================================================

def _is_last_step(ep_idx: int, step_idx: int, episodes: List[Dict]) -> bool:
    """True if step_idx is the final step in the current episode."""
    if not episodes or ep_idx >= len(episodes):
        return False
    return step_idx >= len(episodes[ep_idx]["steps"]) - 1


def _is_last_episode(ep_idx: int, episodes: List[Dict]) -> bool:
    """True if ep_idx is the final episode in the list."""
    return bool(episodes) and ep_idx >= len(episodes) - 1


def _compute_btn_label_and_note(
    ep_idx: int, step_idx: int, episodes: List[Dict]
) -> Tuple[str, str]:
    """Return (next_button_label, hint_html) based on current position."""
    if not episodes:
        return "Next Step ▶", ""
    last_step = _is_last_step(ep_idx, step_idx, episodes)
    last_ep   = _is_last_episode(ep_idx, episodes)
    if last_step and last_ep:
        return "🔄 Restart Simulation", _ALL_DONE_NOTE
    elif last_step:
        return "🔄 Next Episode ▶", _EPISODE_END_NOTE
    return "Next Step ▶", ""


def _build_outputs_from_step_result(
    result:          Dict[str, Any],
    scripted_step:   Dict[str, Any],
    ep_idx:          int,
    step_idx:        int,
    episodes:        List[Dict[str, Any]],
    history:         List[Dict[str, Any]],
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _build_outputs_from_step_result
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Derive all 12 display values from a real env_step() or env_reset()
    response, combined with the scripted step annotation.

    Args:
        result        : Raw dict from env_reset() or env_step().
        scripted_step : The step script dict (plan, note, macro_proposed).
        ep_idx        : Current episode index.
        step_idx      : Current step index within the episode.
        episodes      : Full episode list.
        history       : List of already-executed frame dicts (for Prev).

    Returns:
        12-element tuple for the display components (indices 0–11).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # ---- Parse env observation -----------------------------------------
    task         = parse_task_from_obs(result)
    all_tools    = parse_tools_from_obs(result)
    macros       = extract_macros(all_tools)
    atomic_names = [t["name"] for t in all_tools if not t.get("is_macro")]

    # ---- Task prompt -------------------------------------------------------
    task_text: str = task.get("prompt", "— no task prompt —")

    # ---- Available tools HTML (atomic + macro section) --------------------
    tools_html: str = render_tools_html(atomic_names or ATOMIC_TOOLS, macros)

    # ---- Progress string ---------------------------------------------------
    current_ep    = episodes[ep_idx]
    total_eps     = len(episodes)
    total_steps   = len(current_ep["steps"])
    ep_label      = current_ep.get("episode_label", current_ep["episode_id"])
    progress_text = (
        f"Step {step_idx + 1} of {total_steps}  |  "
        f"Episode {ep_idx + 1} of {total_eps}: {ep_label}"
    )

    # ---- Agent plan (with macro intro annotation if applicable) -----------
    plan          = scripted_step.get("plan", [])
    macro_prop    = scripted_step.get("macro_proposed")
    plan_html     = render_plan_html(plan, macros, macro_proposal=macro_prop)

    # ---- Reward (comes from env — None on reset) --------------------------
    raw_reward    = result.get("reward") or 0.0
    reward_html   = render_reward_html(float(raw_reward))

    # ---- Turn counters -----------------------------------------------------
    baseline      = task.get("baseline_call_count") or task.get("baseline_token_cost") or len(plan)
    turns_used    = len(plan)
    turns_saved   = max(0, baseline - turns_used)

    # ---- Macro library panel -----------------------------------------------
    macro_lib_html = render_macro_library_html(macros)

    # ---- Agent note --------------------------------------------------------
    note_text: str = scripted_step.get("note", "")

    # ---- Next-button label + hint ------------------------------------------
    btn_label, btn_note = _compute_btn_label_and_note(ep_idx, step_idx, episodes)

    return (
        task_text,       # 0
        tools_html,      # 1
        progress_text,   # 2
        plan_html,       # 3
        reward_html,     # 4
        turns_used,      # 5
        turns_saved,     # 6
        macro_lib_html,  # 7
        note_text,       # 8
        btn_label,       # 9  — next button label
        btn_note,        # 10 — hint HTML
        "",              # 11 — clear status message
    )


def _blank_outputs() -> Tuple:
    """12-element blank tuple shown before any simulation is loaded."""
    return (
        "Configure the env URL and click 'Run Simulation' to start.",
        "<div style='color:#9ca3af;padding:8px;'>Connect to env to see tools.</div>",
        "Step — of —  |  Episode — of —",
        "<p style='color:#9ca3af;'>No plan yet.</p>",
        render_reward_html(0.0),
        0, 0,
        render_macro_library_html([]),
        "Notes will appear after simulation starts.",
        "Next Step ▶",  # btn_label
        "",             # btn_note
        "",             # status
    )


def _error_outputs(msg: str) -> Tuple:
    """12-element error tuple shown when env call fails."""
    err_html = (
        f'<div style="color:#f87171;padding:10px;border-radius:6px;'
        f'background:#450a0a;margin:4px 0;">{msg}</div>'
    )
    return (
        "— env error —",
        "<div style='color:#9ca3af;padding:8px;'>Env not reachable.</div>",
        "Step — of —  |  Episode — of —",
        "<p style='color:#9ca3af;'>No plan.</p>",
        render_reward_html(-0.2),
        0, 0,
        render_macro_library_html([]),
        msg,
        "Next Step ▶",
        "",
        err_html,
    )


# ===========================================================================
# EVENT HANDLERS
# ===========================================================================

def on_run_simulation(
    model_label: str,
    difficulty:  str,
    env_url:     str,
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_run_simulation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Called when the user clicks "Run Simulation".

    1. Loads the scripted episode list for the selected model + difficulty.
    2. Calls env_reset() on the env server with the first episode_id.
    3. Returns display outputs for step 0 of episode 0.

    Returns:
        12 display outputs + episodes_state + ep_idx + step_idx + history_state.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    logger.info("Demo run | model=%s difficulty=%s env=%s", model_label, difficulty, env_url)

    diff_key = difficulty.lower()
    profile  = SCRIPTED_PROFILES.get(model_label, {})
    episodes: List[Dict[str, Any]] = profile.get(diff_key, [])

    if not episodes or not episodes[0].get("steps"):
        blank = _blank_outputs()
        err_note = (
            f'<div style="color:#f87171;padding:8px;">'
            f'No scripted episodes for {model_label} / {difficulty}.</div>'
        )
        return blank[:-1] + (err_note,) + ([], 0, 0, [])

    ep_idx   = 0
    step_idx = 0
    episode  = episodes[ep_idx]

    # Call the real env
    result, err = env_reset(env_url, episode["episode_id"])
    if err or result is None:
        return _error_outputs(f"env_reset failed: {err}") + ([], 0, 0, [])

    scripted_step = episode["steps"][step_idx]
    outputs = _build_outputs_from_step_result(
        result, scripted_step, ep_idx, step_idx, episodes, []
    )

    # Seed history with this frame so Prev can navigate back
    history = [{"result": result, "scripted": scripted_step, "ep_idx": ep_idx, "step_idx": step_idx}]

    return outputs + (episodes, ep_idx, step_idx, history)


def on_next_step(
    ep_idx:   int,
    step_idx: int,
    episodes: List[Dict[str, Any]],
    env_url:  str,
    history:  List[Dict[str, Any]],
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_next_step
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Advance one step forward.  Sends the scripted plan to the real env.

    Within an episode: call env_step() with the next scripted plan.
    At end of episode: call env_reset() for the next episode_id.
    At end of all episodes: wrap back to start (restart).

    Returns:
        12 display outputs + new ep_idx + new step_idx + updated history.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not episodes:
        return _blank_outputs() + (0, 0, [])

    last_step = _is_last_step(ep_idx, step_idx, episodes)
    last_ep   = _is_last_episode(ep_idx, episodes)

    # ---- Determine next position ------------------------------------------
    if last_step and last_ep:
        # Restart from beginning
        new_ep_idx, new_step_idx = 0, 0
    elif last_step:
        # Advance to next episode
        new_ep_idx, new_step_idx = ep_idx + 1, 0
    else:
        # Normal step advance within episode
        new_ep_idx, new_step_idx = ep_idx, step_idx + 1

    episode       = episodes[new_ep_idx]
    scripted_step = episode["steps"][new_step_idx]

    # ---- Call the env -------------------------------------------------------
    if new_step_idx == 0:
        # Episode boundary — call reset for the new episode
        result, err = env_reset(env_url, episode["episode_id"])
    else:
        # Mid-episode — call step with the scripted plan
        result, err = env_step(
            env_url,
            scripted_step["plan"],
            scripted_step.get("macro_proposed"),
        )

    if err or result is None:
        return _error_outputs(f"Env call failed: {err}") + (new_ep_idx, new_step_idx, history)

    outputs = _build_outputs_from_step_result(
        result, scripted_step, new_ep_idx, new_step_idx, episodes, history
    )

    # Append to history for Prev navigation
    new_history = history + [
        {"result": result, "scripted": scripted_step, "ep_idx": new_ep_idx, "step_idx": new_step_idx}
    ]

    return outputs + (new_ep_idx, new_step_idx, new_history)


def on_prev_step(
    ep_idx:   int,
    step_idx: int,
    episodes: List[Dict[str, Any]],
    history:  List[Dict[str, Any]],
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_prev_step
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Step backwards through already-executed frames using the history
    buffer.  Does NOT re-call the env (env state is not reversible).

    Args:
        ep_idx   : Current episode index.
        step_idx : Current step index.
        episodes : Full episode list.
        history  : List of previously executed frame dicts.

    Returns:
        12 display outputs + new ep_idx + new step_idx + unchanged history.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not history or len(history) < 2:
        # Already at the start — return unchanged
        if not history:
            return _blank_outputs() + (ep_idx, step_idx, history)
        frame   = history[-1]
        outputs = _build_outputs_from_step_result(
            frame["result"], frame["scripted"], frame["ep_idx"], frame["step_idx"], episodes, history
        )
        return outputs + (frame["ep_idx"], frame["step_idx"], history)

    # Pop the latest frame and show the previous one
    prev_frame = history[-2]
    outputs    = _build_outputs_from_step_result(
        prev_frame["result"], prev_frame["scripted"],
        prev_frame["ep_idx"], prev_frame["step_idx"],
        episodes, history[:-1],
    )
    return outputs + (prev_frame["ep_idx"], prev_frame["step_idx"], history[:-1])


def on_auto_tick(
    auto_active: bool,
    ep_idx:      int,
    step_idx:    int,
    episodes:    List[Dict[str, Any]],
    env_url:     str,
    history:     List[Dict[str, Any]],
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_auto_tick
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Called by gr.Timer.  Advances one step if auto-play is active.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not auto_active or not episodes:
        if not episodes:
            return _blank_outputs() + (0, 0, [])
        # Return current frame unchanged
        if history:
            frame   = history[-1]
            outputs = _build_outputs_from_step_result(
                frame["result"], frame["scripted"],
                frame["ep_idx"], frame["step_idx"], episodes, history
            )
            return outputs + (frame["ep_idx"], frame["step_idx"], history)
        return _blank_outputs() + (ep_idx, step_idx, history)

    return on_next_step(ep_idx, step_idx, episodes, env_url, history)


def on_test_env(env_url: str) -> str:
    """Check env health and return a status HTML string."""
    _, msg = check_env_health(env_url)
    if msg.startswith("✅"):
        return (
            f'<div style="color:#4ade80;padding:8px;border-radius:6px;background:#052e16;">{msg}</div>'
        )
    return (
        f'<div style="color:#f87171;padding:8px;border-radius:6px;background:#450a0a;">{msg}</div>'
    )


# ===========================================================================
# TAB BUILDER
# ===========================================================================

def build_demo_tab() -> gr.Tab:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    build_demo_tab
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Build and return the "Demo Mode" gr.Tab with all event wiring.

    Returns:
        Configured gr.Tab instance.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    with gr.Tab("🎬 Demo Mode") as tab:

        # -------------------------------------------------------------------
        # HEADER
        # -------------------------------------------------------------------
        gr.Markdown(
            """
            # ToolForge Demo

            Watch a pre-scripted agent solve DevOps tasks and progressively learn to
            **compress repeated tool sequences into reusable macro tools**.

            Agent plans are scripted, but **rewards and task state come from the real
            ToolForge environment server** — connect it below before running.
            """
        )

        # -------------------------------------------------------------------
        # ENV CONNECTION ROW
        # -------------------------------------------------------------------
        with gr.Row():
            env_url_field = gr.Textbox(
                value=DEFAULT_ENV_URL,
                label="Environment Server URL",
                placeholder="http://localhost:8000",
                scale=4,
            )
            test_env_btn = gr.Button("🔗 Test Connection", scale=1)

        env_status_html = gr.HTML(value="")

        gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

        # -------------------------------------------------------------------
        # SIMULATION CONTROLS
        # -------------------------------------------------------------------
        with gr.Row():
            model_selector = gr.Dropdown(
                choices=MODEL_OPTIONS,
                value=MODEL_OPTIONS[0],
                label="Model Profile",
                scale=3,
            )
            difficulty_selector = gr.Dropdown(
                choices=DIFFICULTY_OPTIONS,
                value=DIFFICULTY_OPTIONS[0],
                label="Difficulty",
                scale=1,
            )
            run_btn = gr.Button("▶ Run Simulation", variant="primary", scale=2)

        gr.Markdown(f"*{DISCLAIMER_TEXT}*")

        gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

        # -------------------------------------------------------------------
        # MAIN TWO-COLUMN LAYOUT
        # -------------------------------------------------------------------
        with gr.Row():

            # ===============================================================
            # LEFT COLUMN
            # ===============================================================
            with gr.Column(scale=1):

                current_task_box = gr.Textbox(
                    label="📋 Current Task",
                    value="Configure env URL and click Run Simulation.",
                    lines=3,
                    interactive=False,
                )

                gr.Markdown("**🔧 Available Tools** *(purple = macro)*")
                available_tools_html = gr.HTML(
                    value="<div style='color:#9ca3af;padding:8px;'>Connect env to see tools.</div>"
                )

                episode_progress_box = gr.Textbox(
                    label="📊 Progress",
                    value="Step — of —  |  Episode — of —",
                    interactive=False,
                )

            # ===============================================================
            # RIGHT COLUMN
            # ===============================================================
            with gr.Column(scale=1):

                gr.Markdown("**🤖 Agent Plan**")
                agent_plan_html = gr.HTML(
                    value="<p style='color:#9ca3af;'>No plan yet.</p>"
                )

                gr.Markdown("**🏆 Step Reward** *(real env score)*")
                reward_html = gr.HTML(value=render_reward_html(0.0))

                with gr.Row():
                    turns_used_num = gr.Number(
                        label="LLM Turns Used",
                        value=0, interactive=False, precision=0,
                    )
                    turns_saved_num = gr.Number(
                        label="Turns Saved vs Baseline",
                        value=0, interactive=False, precision=0,
                    )

                gr.Markdown("**⚙ Macro Library**")
                macro_library_html = gr.HTML(value=render_macro_library_html([]))

        # Agent note — full width
        agent_note_box = gr.Textbox(
            label="📝 Agent Note",
            value="Notes will appear after simulation starts.",
            lines=3,
            interactive=False,
        )

        # Status bar for env errors
        step_status_html = gr.HTML(value="")

        gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

        # -------------------------------------------------------------------
        # NAVIGATION ROW
        # -------------------------------------------------------------------
        with gr.Row():
            prev_btn      = gr.Button("◀ Previous Step", scale=1)
            thinking_html = gr.HTML(value="", visible=True)
            next_btn      = gr.Button("Next Step ▶", variant="secondary", scale=1)
            auto_play_btn = gr.Button("▶ Auto Play", variant="secondary", scale=1)

        btn_note_html = gr.HTML(value="")

        # -------------------------------------------------------------------
        # HIDDEN STATE
        # -------------------------------------------------------------------

        # Index of current episode in episodes_state
        ep_index_state   = gr.State(value=0)

        # Index of current step within the episode
        step_index_state = gr.State(value=0)

        # Full episode+step list for the active profile
        episodes_state   = gr.State(value=[])

        # History of executed frames for Prev navigation
        # Each entry: {"result": dict, "scripted": dict, "ep_idx": int, "step_idx": int}
        history_state    = gr.State(value=[])

        # Whether auto-play is active
        auto_active_state = gr.State(value=False)

        # -------------------------------------------------------------------
        # AUTO-PLAY TIMER
        # -------------------------------------------------------------------
        try:
            auto_timer = gr.Timer(value=3.0, active=False)
            _has_timer = True
        except AttributeError:
            logger.warning("gr.Timer unavailable — auto-play disabled.")
            _has_timer = False

        # -------------------------------------------------------------------
        # CANONICAL OUTPUT LIST (12 display + state components)
        # -------------------------------------------------------------------
        _display = [
            current_task_box,      # 0
            available_tools_html,  # 1
            episode_progress_box,  # 2
            agent_plan_html,       # 3
            reward_html,           # 4
            turns_used_num,        # 5
            turns_saved_num,       # 6
            macro_library_html,    # 7
            agent_note_box,        # 8
            next_btn,              # 9  — label update
            btn_note_html,         # 10
            step_status_html,      # 11
        ]
        _step_state = [ep_index_state, step_index_state, history_state]

        # -------------------------------------------------------------------
        # EVENTS
        # -------------------------------------------------------------------

        # Test env connection
        test_env_btn.click(
            fn=on_test_env,
            inputs=[env_url_field],
            outputs=[env_status_html],
        )

        # Run Simulation → reset env, show step 0
        run_btn.click(
            fn=on_run_simulation,
            inputs=[model_selector, difficulty_selector, env_url_field],
            outputs=_display + [episodes_state] + _step_state,
        )

        # Next Step (also handles episode boundary)
        next_btn.click(
            fn=on_next_step,
            inputs=[ep_index_state, step_index_state, episodes_state, env_url_field, history_state],
            outputs=_display + _step_state,
        )

        # Previous Step (navigates history buffer, no env call)
        prev_btn.click(
            fn=on_prev_step,
            inputs=[ep_index_state, step_index_state, episodes_state, history_state],
            outputs=_display + _step_state,
        )

        # Auto Play toggle
        def toggle_auto_play(current_active: bool):
            """Toggle auto-play and update button + spinner."""
            new_active = not current_active
            new_label  = "⏹ Stop Auto Play" if new_active else "▶ Auto Play"
            thinking   = (
                '<span class="thinking-spinner"></span>'
                '<em style="color:#a78bfa;">Auto advancing…</em>'
                if new_active else ""
            )
            if _has_timer:
                return new_active, gr.update(value=new_label), gr.update(value=thinking), gr.Timer(active=new_active)
            return new_active, gr.update(value=new_label), gr.update(value=thinking)

        if _has_timer:
            auto_play_btn.click(
                fn=toggle_auto_play,
                inputs=[auto_active_state],
                outputs=[auto_active_state, auto_play_btn, thinking_html, auto_timer],
            )
            auto_timer.tick(
                fn=on_auto_tick,
                inputs=[auto_active_state, ep_index_state, step_index_state, episodes_state, env_url_field, history_state],
                outputs=_display + _step_state,
            )
        else:
            auto_play_btn.click(
                fn=toggle_auto_play,
                inputs=[auto_active_state],
                outputs=[auto_active_state, auto_play_btn, thinking_html],
            )

    return tab
