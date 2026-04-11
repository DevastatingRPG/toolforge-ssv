# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tab 1 — Demo Mode
=================

A read-only simulation tab.  The user selects a model profile + difficulty,
clicks "Run Simulation", and then steps through a pre-scripted agent episode
using Previous / Next buttons or Auto Play.

No real LLM or environment calls are made here.  All data comes from the
SCRIPTED_PROFILES dict in shared.py.

Layout (top-to-bottom):
    ┌─ Header (title + description) ─────────────────────────────────┐
    │  Controls row: model selector | difficulty selector | Run btn  │
    │  Disclaimer text                                                 │
    ├─ Left column ─────────────┬─ Right column ────────────────────┐│
    │  Current Task             │  Agent Plan                        ││
    │  Available Tools          │  Reward display                    ││
    │  Episode Progress         │  LLM Turns Used / Turns Saved      ││
    │                           │  Macro Library                     ││
    └───────────────────────────┴────────────────────────────────────┘│
    │  Navigation: [◀ Previous] [Thinking…] [Next ▶] [▶ Auto Play]  │
    └─────────────────────────────────────────────────────────────────┘
"""

import logging
import time
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

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model / difficulty options
# ---------------------------------------------------------------------------

# Available model display names — these are simulated profiles, not live calls
MODEL_OPTIONS: List[str] = [
    "GPT-4o (simulated)",
    "Llama 3 (simulated)",
    "Mistral (simulated)",
]

# Difficulty options exposed to the user (only Easy is scripted for now)
DIFFICULTY_OPTIONS: List[str] = ["Easy"]

# Disclaimer shown below the controls row
DISCLAIMER_TEXT: str = (
    "ℹ️  These are pre-scripted behavioral profiles, not live API calls. "
    "Each model profile demonstrates a different macro-learning strategy "
    "to illustrate how ToolForge incentivises pattern abstraction."
)


# ===========================================================================
# STATE HELPERS
# ---------------------------------------------------------------------------
# These functions derive display values from a single episode frame dict.
# They are called inside Gradio event handlers which must be pure functions
# (no side effects, no global mutation).
# ===========================================================================

def _frame_to_outputs(
    frame: Dict[str, Any],
    episode_idx: int,
    total_episodes: int,
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _frame_to_outputs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Convert one scripted episode frame into the tuple of values that
    Gradio uses to update every output component.

    The order of values in the returned tuple must exactly match the
    order of the `outputs` list in every event handler that calls this
    function.  See build_demo_tab() for the canonical output ordering.

    Args:
        frame           : One episode frame dict from SCRIPTED_PROFILES.
        episode_idx     : 0-based index of the current episode.
        total_episodes  : Total number of episodes in the current profile.

    Returns:
        Tuple of values — one per output component.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # Task prompt text
    task_text: str = frame["task_prompt"]

    # Available tools: all atomic tools + any macros created so far
    macros_so_far: List[Dict[str, Any]] = frame.get("macros_so_far", [])
    tools_html: str = render_tools_html(ATOMIC_TOOLS, macros_so_far)

    # Episode progress string e.g. "Episode 3 of 6"
    progress_text: str = f"Episode {episode_idx + 1} of {total_episodes}"

    # Agent plan HTML (highlights macro calls)
    plan_html: str = render_plan_html(frame["plan"], macros_so_far)

    # Reward display HTML
    reward_html: str = render_reward_html(frame["reward"])

    # LLM turns used = plan length (1 LLM call per step in our model)
    turns_used: int = frame["turns_used"]

    # Turns saved = max(0, baseline - actual_plan_length)
    turns_saved: int = max(0, frame["turns_baseline"] - frame["turns_used"])

    # Macro library HTML
    macro_lib_html: str = render_macro_library_html(macros_so_far)

    # Agent note / narrative
    note_text: str = frame.get("note", "")

    # Macro proposed annotation for the note area
    proposed = frame.get("macro_proposed")
    if proposed:
        steps_str = " → ".join(proposed["steps"])
        note_text += (
            f'\n\n📦 Macro proposed: "{proposed["name"]}" = [{steps_str}]'
        )

    return (
        task_text,       # Current Task textbox
        tools_html,      # Available Tools HTML
        progress_text,   # Episode Progress textbox
        plan_html,       # Agent Plan HTML
        reward_html,     # Reward HTML
        turns_used,      # LLM Turns Used number
        turns_saved,     # Turns Saved vs Baseline number
        macro_lib_html,  # Macro Library HTML
        note_text,       # Agent Note textbox
    )


def _blank_outputs() -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _blank_outputs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Return the tuple of blank / placeholder values shown before the
    user runs a simulation.

    Returns:
        Tuple of default values matching the same component order
        as _frame_to_outputs().
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    return (
        "Click 'Run Simulation' to load a scripted episode.",   # task
        "<div style='color:#9ca3af;padding:8px;'>Run simulation to see tools.</div>",  # tools
        "Episode — of —",                                        # progress
        "<p style='color:#9ca3af;'>No plan yet.</p>",           # plan
        render_reward_html(0.0),                                 # reward
        0,                                                       # turns used
        0,                                                       # turns saved
        render_macro_library_html([]),                           # macro library
        "Notes will appear here after simulation.",             # note
    )


# ===========================================================================
# EVENT HANDLERS
# ===========================================================================

def on_run_simulation(
    model_label: str,
    difficulty: str,
    ep_index_state: int,
    episodes_state: List[Dict[str, Any]],
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_run_simulation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Called when the user clicks "Run Simulation".

    Loads the scripted profile for the selected model + difficulty,
    resets the episode index to 0, and returns outputs for episode 0.

    Args:
        model_label      : Selected model display name.
        difficulty       : Selected difficulty string ("Easy" etc.).
        ep_index_state   : Current episode index (gr.State — ignored on run).
        episodes_state   : Current episodes list (gr.State — replaced on run).

    Returns:
        Flat tuple: (task, tools, progress, plan, reward, turns_used,
                     turns_saved, macro_lib, note,
                     new_ep_index, new_episodes_list)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    logger.info("Demo simulation started | model=%s difficulty=%s", model_label, difficulty)

    # Normalise difficulty key to lowercase for dict lookup
    diff_key = difficulty.lower()

    # Retrieve the scripted episodes for this profile
    profile = SCRIPTED_PROFILES.get(model_label, {})
    episodes: List[Dict[str, Any]] = profile.get(diff_key, [])

    if not episodes:
        # Fallback: no data for this combo — show an error note
        outputs = _blank_outputs()
        return outputs + (0, [])

    # Start at episode 0
    new_index = 0
    frame = episodes[new_index]
    outputs = _frame_to_outputs(frame, new_index, len(episodes))

    # Return outputs + updated state values (index and episodes list)
    return outputs + (new_index, episodes)


def on_next_episode(
    ep_index_state: int,
    episodes_state: List[Dict[str, Any]],
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_next_episode
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Advance to the next episode frame.  Wraps around at the end.

    Args:
        ep_index_state  : Current 0-based episode index (gr.State).
        episodes_state  : List of episode frame dicts (gr.State).

    Returns:
        Flat tuple matching the same component + state ordering.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not episodes_state:
        # No simulation loaded yet — return blanks unchanged
        return _blank_outputs() + (0, [])

    # Advance index with wrap-around
    new_index = (ep_index_state + 1) % len(episodes_state)
    frame = episodes_state[new_index]
    outputs = _frame_to_outputs(frame, new_index, len(episodes_state))
    return outputs + (new_index, episodes_state)


def on_prev_episode(
    ep_index_state: int,
    episodes_state: List[Dict[str, Any]],
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_prev_episode
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Step back to the previous episode frame.  Wraps around at the start.

    Args:
        ep_index_state  : Current 0-based episode index (gr.State).
        episodes_state  : List of episode frame dicts (gr.State).

    Returns:
        Flat tuple matching the same component + state ordering.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not episodes_state:
        return _blank_outputs() + (0, [])

    # Step back with wrap-around
    new_index = (ep_index_state - 1) % len(episodes_state)
    frame = episodes_state[new_index]
    outputs = _frame_to_outputs(frame, new_index, len(episodes_state))
    return outputs + (new_index, episodes_state)


def on_auto_tick(
    auto_active: bool,
    ep_index_state: int,
    episodes_state: List[Dict[str, Any]],
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_auto_tick
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Called by gr.Timer on each tick when auto-play is active.

    If auto-play is off or no episodes are loaded, returns current
    state unchanged to avoid unnecessary re-renders.

    Args:
        auto_active     : Whether auto-play is currently enabled (gr.State).
        ep_index_state  : Current episode index (gr.State).
        episodes_state  : Episode frames list (gr.State).

    Returns:
        Same flat tuple as on_next_episode.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not auto_active or not episodes_state:
        # Nothing to do — return blanks or current unchanged state
        if not episodes_state:
            return _blank_outputs() + (0, [])
        # Return current frame without advancing
        frame = episodes_state[ep_index_state]
        outputs = _frame_to_outputs(frame, ep_index_state, len(episodes_state))
        return outputs + (ep_index_state, episodes_state)

    # Advance to next episode
    return on_next_episode(ep_index_state, episodes_state)


# ===========================================================================
# TAB BUILDER
# ===========================================================================

def build_demo_tab() -> gr.Tab:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    build_demo_tab
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Construct and return the full "Demo Mode" gr.Tab component.

    All internal component wiring (event handlers, state) is done here.
    The caller (gradio_app.py) just inserts the returned Tab into the
    gr.Blocks layout.

    Returns:
        A configured gr.Tab component ready to be placed in gr.Blocks.
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

            Each episode shows one task from the environment.  As the agent encounters
            the same tool-call patterns, it proposes macros that collapse those patterns
            into a single reusable action — saving tokens and earning efficiency bonuses.
            """
        )

        # -------------------------------------------------------------------
        # CONTROLS ROW
        # -------------------------------------------------------------------
        with gr.Row():
            # Model selector dropdown
            model_selector = gr.Dropdown(
                choices=MODEL_OPTIONS,          # available simulated profiles
                value=MODEL_OPTIONS[0],         # default: GPT-4o
                label="Model Profile",
                scale=3,
            )

            # Difficulty selector (only Easy is scripted for now)
            difficulty_selector = gr.Dropdown(
                choices=DIFFICULTY_OPTIONS,     # ["Easy"]
                value=DIFFICULTY_OPTIONS[0],
                label="Difficulty",
                scale=1,
            )

            # Trigger to load / reload the simulation
            run_btn = gr.Button(
                "▶ Run Simulation",
                variant="primary",
                scale=2,
            )

        # Disclaimer below controls
        gr.Markdown(f"*{DISCLAIMER_TEXT}*")

        gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

        # -------------------------------------------------------------------
        # MAIN CONTENT — two-column layout
        # -------------------------------------------------------------------
        with gr.Row():

            # ===============================================================
            # LEFT COLUMN
            # ===============================================================
            with gr.Column(scale=1):

                # Current task prompt
                current_task_box = gr.Textbox(
                    label="📋 Current Task",
                    value="Click 'Run Simulation' to load a scripted episode.",
                    lines=3,
                    interactive=False,
                )

                # Available tools (HTML for colour-coded macros)
                gr.Markdown("**🔧 Available Tools** *(purple = macro)*")
                available_tools_html = gr.HTML(
                    value="<div style='color:#9ca3af;padding:8px;'>Run simulation to see tools.</div>"
                )

                # Episode progress counter
                episode_progress_box = gr.Textbox(
                    label="📊 Episode Progress",
                    value="Episode — of —",
                    interactive=False,
                )

            # ===============================================================
            # RIGHT COLUMN
            # ===============================================================
            with gr.Column(scale=1):

                # Agent plan (HTML with macro highlighting)
                gr.Markdown("**🤖 Agent Plan**")
                agent_plan_html = gr.HTML(
                    value="<p style='color:#9ca3af;'>No plan yet.</p>"
                )

                # Reward display — large styled number
                gr.Markdown("**🏆 Step Reward**")
                reward_html = gr.HTML(value=render_reward_html(0.0))

                # Turn counters row
                with gr.Row():
                    turns_used_num = gr.Number(
                        label="LLM Turns Used",
                        value=0,
                        interactive=False,
                        precision=0,
                    )
                    turns_saved_num = gr.Number(
                        label="Turns Saved vs Baseline",
                        value=0,
                        interactive=False,
                        precision=0,
                    )

                # Macro library panel
                gr.Markdown("**⚙ Macro Library**")
                macro_library_html = gr.HTML(value=render_macro_library_html([]))

        # Agent note / narrative (full width, below columns)
        agent_note_box = gr.Textbox(
            label="📝 Agent Note",
            value="Notes will appear here after simulation.",
            lines=3,
            interactive=False,
        )

        gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

        # -------------------------------------------------------------------
        # NAVIGATION ROW
        # -------------------------------------------------------------------
        with gr.Row():
            prev_btn = gr.Button("◀ Previous Episode", scale=1)

            # Thinking indicator — shown briefly when auto-play is active
            # TODO: Wire this to show during the auto-advance delay
            thinking_html = gr.HTML(
                value="",
                visible=True,
                elem_id="demo_thinking_indicator",
            )

            next_btn = gr.Button("Next Episode ▶", variant="secondary", scale=1)

            auto_play_btn = gr.Button(
                "▶ Auto Play",
                variant="secondary",
                scale=1,
            )

        # -------------------------------------------------------------------
        # HIDDEN STATE
        # -------------------------------------------------------------------

        # Holds the 0-based index of the currently displayed episode
        ep_index_state = gr.State(value=0)

        # Holds the full list of episode frame dicts for the active profile
        episodes_state = gr.State(value=[])

        # Tracks whether auto-play is currently running
        auto_active_state = gr.State(value=False)

        # -------------------------------------------------------------------
        # AUTO-PLAY TIMER
        # Ticks every 3 seconds when auto-play is active.
        # gr.Timer was introduced in Gradio 4.4+.
        # TODO: Gracefully degrade for older Gradio versions.
        # -------------------------------------------------------------------
        try:
            auto_timer = gr.Timer(value=3.0, active=False)
            _has_timer = True
        except AttributeError:
            logger.warning(
                "gr.Timer not available in this Gradio version. "
                "Auto-play will not function. Please upgrade to Gradio 4.4+."
            )
            _has_timer = False

        # -------------------------------------------------------------------
        # CANONICAL OUTPUT LIST
        # The order here must match _frame_to_outputs() and _blank_outputs().
        # -------------------------------------------------------------------
        _all_display_outputs = [
            current_task_box,
            available_tools_html,
            episode_progress_box,
            agent_plan_html,
            reward_html,
            turns_used_num,
            turns_saved_num,
            macro_library_html,
            agent_note_box,
        ]
        # State outputs always appended after display outputs
        _state_outputs = [ep_index_state, episodes_state]

        # -------------------------------------------------------------------
        # EVENT: Run Simulation button
        # -------------------------------------------------------------------
        run_btn.click(
            fn=on_run_simulation,
            inputs=[model_selector, difficulty_selector, ep_index_state, episodes_state],
            outputs=_all_display_outputs + _state_outputs,
        )

        # -------------------------------------------------------------------
        # EVENT: Next Episode button
        # -------------------------------------------------------------------
        next_btn.click(
            fn=on_next_episode,
            inputs=[ep_index_state, episodes_state],
            outputs=_all_display_outputs + _state_outputs,
        )

        # -------------------------------------------------------------------
        # EVENT: Previous Episode button
        # -------------------------------------------------------------------
        prev_btn.click(
            fn=on_prev_episode,
            inputs=[ep_index_state, episodes_state],
            outputs=_all_display_outputs + _state_outputs,
        )

        # -------------------------------------------------------------------
        # EVENT: Auto Play button — toggles auto_active_state and timer
        # -------------------------------------------------------------------
        def toggle_auto_play(current_active: bool):
            """
            Toggle the auto-play state.  Returns the new state value and
            updates the button label to reflect the current mode.
            """
            new_active = not current_active
            new_label  = "⏹ Stop Auto Play" if new_active else "▶ Auto Play"

            # Show/hide the thinking spinner based on auto state
            thinking_content = (
                '<span class="thinking-spinner"></span><em style="color:#a78bfa;">Auto advancing…</em>'
                if new_active else ""
            )

            if _has_timer:
                # Activate/deactivate the timer component
                return new_active, gr.update(value=new_label), gr.update(value=thinking_content), gr.Timer(active=new_active)
            else:
                return new_active, gr.update(value=new_label), gr.update(value=thinking_content)

        if _has_timer:
            auto_play_btn.click(
                fn=toggle_auto_play,
                inputs=[auto_active_state],
                outputs=[auto_active_state, auto_play_btn, thinking_html, auto_timer],
            )

            # Timer tick — advance one episode
            auto_timer.tick(
                fn=on_auto_tick,
                inputs=[auto_active_state, ep_index_state, episodes_state],
                outputs=_all_display_outputs + _state_outputs,
            )
        else:
            # Fallback: toggle button still works but no timer-driven advance
            auto_play_btn.click(
                fn=toggle_auto_play,
                inputs=[auto_active_state],
                outputs=[auto_active_state, auto_play_btn, thinking_html],
            )

    return tab
