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
    - "Run Simulation" calls env_reset() on the running ToolForge server
    and shows the first task prompt in "Next Task" while "Current Task"
    stays as a placeholder.
    - "Next Step" sends the scripted plan via env_step() and displays the
      actual reward, task prompt, and macro state returned by the server.

This means the env is the source of truth for:
    - Task prompts
    - Rewards
    - Available tools (including approved macros)
    - Episode done state
    - Episode length (number of tasks in the group)

The scripted profiles (SCRIPTED_PROFILES) define WHAT PLAN each simulated
model would choose at each step.  When the scripted plan list is shorter
than the env's task group, a *fallback step* is derived by recycling the
last scripted plan (with macro substitution if a matching macro exists).

Navigation model:
    Step    = one task prompt within an episode (one env.step() call).
    Episode = one task group (one env.reset() call).
    "Next Step ▶"    — advance within episode (calls env.step()).
    "🔄 Next Episode" — at last step; calls env.reset() with next episode_id.
    "◀ Previous Step" — steps back through already-executed frames (no re-calling env).

Stepping lifecycle:
    1. User clicks "Run Simulation" → env_reset() → display task, no plan.
       Internal state: step_idx = -1 (reset done, no step executed).
    2. User clicks "Next Step" → env_step(scripted_step_0) → display plan + reward.
       Internal state: step_idx = 0.
    3. Repeat until env returns done=True.

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
    parse_total_tasks_from_obs,
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

DISCLAIMER_TEXT: str = (
    "ℹ️  Agent plans are pre-scripted behavioral profiles. "
    "Rewards and task prompts come from the real ToolForge environment server."
)

# Note shown below Next button at episode boundary
_EPISODE_END_NOTE: str = (
    '<em style="color:#a78bfa;font-size:0.85em;">'
    '📌 Episode complete — clicking again starts next episode and resets macros.'
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

def _derive_fallback_step(
    last_scripted_step: Dict[str, Any],
    macros: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Derive a fallback step when the scripted profile is exhausted.

    Strategy:
        - If a macro exists whose steps exactly match the last plan,
          substitute the macro name as a single-call plan.
        - Otherwise, reuse the last plan as-is.
        - Never propose a new macro in fallback (already proposed).

    Args:
        last_scripted_step : The last defined scripted step dict.
        macros             : Current accepted macros from the env observation.

    Returns:
        A new step dict suitable for env_step().
    """
    last_plan = last_scripted_step.get("plan", [])

    # Check if any accepted macro exactly matches the last plan
    for macro in macros:
        if macro.get("steps") == last_plan:
            return {
                "plan":           [macro["name"]],
                "macro_proposed": None,
                "note":           f"Agent reuses macro '{macro['name']}' (derived from learned pattern).",
            }

    return {
        "plan":           list(last_plan),  # shallow copy
        "macro_proposed": None,
        "note":           "Agent continues with the established pattern (plan recycled).",
    }


def _get_scripted_step(
    episode: Dict[str, Any],
    step_idx: int,
    macros: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the scripted step at step_idx, or derive a fallback.

    If step_idx is within the scripted steps list, return it directly.
    Otherwise, derive a fallback from the last scripted step.
    """
    steps = episode.get("steps", [])
    if step_idx < len(steps):
        return steps[step_idx]
    if steps:
        return _derive_fallback_step(steps[-1], macros)
    # Degenerate case — no scripted steps at all
    return {
        "plan":           ["deploy", "healthcheck", "notify"],
        "macro_proposed": None,
        "note":           "No scripted plan available — using default atomic plan.",
    }


def _compute_btn_label_and_note(
    step_idx:    int,
    total_tasks: int,
    is_last_ep:  bool,
    env_done:    bool,
) -> Tuple[str, str]:
    """Return (next_button_label, hint_html) based on progress.

    Episode-done logic uses the UI-side step counter vs total_tasks
    returned by the env at reset time.  This is authoritative even when
    the REST env returns done=True prematurely (which happens in stateless
    mode where each HTTP request creates a fresh env instance).

    An additional ``env_done`` guard is kept as a safety net for when the
    env correctly signals end-of-episode before the scripted step count is
    exhausted (e.g. task count mismatch).

    Args:
        step_idx    : Current 0-based step index just executed.
        total_tasks : Total tasks in this episode (from reset metadata).
        is_last_ep  : Whether this is the last scripted episode.
        env_done    : Whether the env itself signalled done (belt-and-suspenders).
    """
    # Episode is complete when we've executed the last task (step_idx is
    # 0-based, so last valid index is total_tasks - 1).
    ui_episode_done = (
        (total_tasks > 0 and step_idx >= total_tasks - 1)
        or env_done
    )
    if ui_episode_done and is_last_ep:
        return "🔄 Restart Simulation", _ALL_DONE_NOTE
    elif ui_episode_done:
        return "🔄 Next Episode ▶", _EPISODE_END_NOTE
    return "Next Step ▶", ""


def _build_reset_outputs(
    result:    Dict[str, Any],
    ep_idx:    int,
    total_tasks: int,
    episodes:  List[Dict[str, Any]],
) -> Tuple:
    """Build the 13 display values shown after env_reset() — no plan yet.

    This is the state after "Run Simulation" or after crossing an episode
    boundary.  The task prompt and tools are shown, but no plan or reward.
    """
    task      = parse_task_from_obs(result)
    all_tools = parse_tools_from_obs(result)
    macros    = extract_macros(all_tools)
    atomic_names = [t["name"] for t in all_tools if not t.get("is_macro")]

    current_task_text = "No tasks executed yet"
    next_task_text     = task.get("prompt", "— no task prompt —")
    tools_html        = render_tools_html(atomic_names or ATOMIC_TOOLS, macros)
    ep_label       = episodes[ep_idx].get("episode_label", episodes[ep_idx]["episode_id"])
    total_eps      = len(episodes)
    progress_text  = f"Ready (0 of {total_tasks})  |  Episode {ep_idx + 1} of {total_eps}: {ep_label}"
    plan_html      = '<p style="color:#9ca3af;font-style:italic;">Waiting for first step…</p>'
    reward_html    = render_reward_html(0.0)
    macro_lib_html = render_macro_library_html(macros)

    btn_label, btn_note = "Next Step ▶", ""

    return (
        current_task_text,  # 0
        next_task_text,     # 1
        tools_html,         # 2
        progress_text,      # 3
        plan_html,          # 4
        reward_html,        # 5
        0,                  # 6  turns_used
        0,                  # 7  turns_saved
        macro_lib_html,     # 8
        "Click 'Next Step' to execute the agent's first plan.",  # 9 note
        btn_label,          # 10
        btn_note,           # 11
        "",                 # 12 status
    )


def _build_step_outputs(
    result:        Dict[str, Any],
    scripted_step: Dict[str, Any],
    current_task_text: str,
    ep_idx:        int,
    step_idx:      int,
    total_tasks:   int,
    episodes:      List[Dict[str, Any]],
    env_done:      bool,
) -> Tuple:
    """Build the 13 display values after an env_step() call.

    Args:
        result        : Raw dict from env_step().
        scripted_step : The step dict used (plan, note, macro_proposed).
        ep_idx        : Current episode index.
        current_task_text: Prompt displayed as the current task.
        step_idx      : Current step index (0-based, post-increment).
        episodes      : Full episode list.
        env_done      : Whether the env flagged the episode as done.
    """
    task      = parse_task_from_obs(result)
    all_tools = parse_tools_from_obs(result)
    macros    = extract_macros(all_tools)
    atomic_names = [t["name"] for t in all_tools if not t.get("is_macro")]

    # The env returns the prompt to execute after this step, so it becomes
    # the next task while the previously displayed next task becomes current.
    next_task_text = task.get("prompt", "— no task prompt —")
    tools_html      = render_tools_html(atomic_names or ATOMIC_TOOLS, macros)

    # Progress
    ep_label  = episodes[ep_idx].get("episode_label", episodes[ep_idx]["episode_id"])
    total_eps = len(episodes)
    progress_text = (
        f"Step {step_idx + 1} of {total_tasks}  |  "
        f"Episode {ep_idx + 1} of {total_eps}: {ep_label}"
    )
    if env_done:
        progress_text += "  ✅ Complete"

    # Plan display
    plan       = scripted_step.get("plan", [])
    macro_prop = scripted_step.get("macro_proposed")
    plan_html  = render_plan_html(plan, macros, macro_proposal=macro_prop)

    # Reward from env
    raw_reward   = result.get("reward") or 0.0
    reward_html  = render_reward_html(float(raw_reward))

    # Turn counters
    baseline    = task.get("baseline_call_count") or task.get("baseline_token_cost") or len(plan)
    turns_used  = len(plan)
    turns_saved = max(0, baseline - turns_used)

    # Macro library
    macro_lib_html = render_macro_library_html(macros)

    # Agent note
    note_text = scripted_step.get("note", "")

    # Button state — uses UI-side step counter to avoid premature "Next Episode"
    is_last_ep = ep_idx >= len(episodes) - 1
    btn_label, btn_note = _compute_btn_label_and_note(
        step_idx, total_tasks, is_last_ep, env_done
    )

    return (
        current_task_text,  # 0
        next_task_text,     # 1
        tools_html,         # 2
        progress_text,      # 3
        plan_html,          # 4
        reward_html,        # 5
        turns_used,         # 6
        turns_saved,        # 7
        macro_lib_html,     # 8
        note_text,          # 9
        btn_label,          # 10
        btn_note,           # 11
        "",                 # 12 status
    )


def _blank_outputs() -> Tuple:
    """13-element blank tuple shown before any simulation is loaded."""
    return (
        "No tasks executed yet",
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
    """13-element error tuple shown when env call fails."""
    err_html = (
        f'<div style="color:#f87171;padding:10px;border-radius:6px;'
        f'background:#450a0a;margin:4px 0;">{msg}</div>'
    )
    return (
        "— env error —",
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
    env_url:     str,
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_run_simulation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Called when the user clicks "Run Simulation".

    1. Loads the scripted episode list for the selected model (Easy only).
    2. Calls env_reset() on the env server with the first episode_id.
     3. Returns display outputs showing the first prompt in Next Task but
         **no plan yet**.

    Internal state after this call:
        step_idx    = -1  (reset complete, no step executed)
        total_tasks = N   (number of tasks in this episode, from env metadata)

    Returns:
        13 display outputs + episodes_state + ep_idx + step_idx
        + history_state + env_done_state + total_tasks_state.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    logger.info("Demo run | model=%s env=%s", model_label, env_url)

    # Demo mode only uses Easy profiles
    profile  = SCRIPTED_PROFILES.get(model_label, {})
    episodes: List[Dict[str, Any]] = profile.get("easy", [])

    if not episodes or not episodes[0].get("steps"):
        blank = _blank_outputs()
        err_note = (
            f'<div style="color:#f87171;padding:8px;">'
            f'No scripted episodes for {model_label}.</div>'
        )
        # Error: (episodes, ep_idx, step_idx, total_tasks, history, env_done)
        return blank[:-1] + (err_note,) + ([], 0, -1, 0, [], False)

    ep_idx   = 0
    step_idx = -1  # No step executed yet
    episode  = episodes[ep_idx]

    # Call the real env via /web/reset (stateful WebInterfaceManager)
    result, err = env_reset(env_url, episode["episode_id"])
    if err or result is None:
        return _error_outputs(f"env_reset failed: {err}") + ([], 0, -1, 0, [], False)

    # Extract episode length from obs metadata so UI can track progress
    # without depending on env.done (unreliable in stateless REST mode).
    total_tasks: int = parse_total_tasks_from_obs(result)
    logger.info("Episode '%s' loaded | total_tasks=%d", episode["episode_id"], total_tasks)

    outputs = _build_reset_outputs(result, ep_idx, total_tasks, episodes)
    reset_current_task = "No tasks executed yet"
    reset_next_task = parse_task_from_obs(result).get("prompt", "— no task prompt —")

    # Seed history with the reset frame (no plan executed)
    history = [{
        "result":           result,
        "scripted":         None,
        "current_task_text": reset_current_task,
        "next_task_text":    reset_next_task,
        "ep_idx":           ep_idx,
        "step_idx":         -1,
        "env_done":         False,
        "total_tasks":      total_tasks,
    }]

    # Return order must match: [episodes_state] + _step_state
    # i.e.: episodes, ep_idx, step_idx, total_tasks, history, env_done
    return outputs + (episodes, ep_idx, step_idx, total_tasks, history, False)


def on_next_step(
    ep_idx:      int,
    step_idx:    int,
    total_tasks: int,
    episodes:    List[Dict[str, Any]],
    env_url:     str,
    history:     List[Dict[str, Any]],
    env_done:    bool,
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_next_step
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Advance one step forward.  Sends the scripted plan to the real env.

    Episode-done detection uses the UI-side step counter (new_step_idx
    vs total_tasks) so the button label is correct even when the env
    prematurely returns done=True (stateless REST mode bug).

    Cases:
        env_done or step >= total_tasks : Advance to next episode (env_reset).
        step_idx == -1                  : Execute scripted step 0.
        else                            : Normal step advance (env_step).

    Returns:
        13 display outputs + new ep_idx + new step_idx + updated history
        + new env_done + new total_tasks.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not episodes:
        return _blank_outputs() + (0, -1, 0, [], False)

    is_last_ep = ep_idx >= len(episodes) - 1

    # Determine whether the CURRENT state is "episode complete" by checking
    # both the env flag and the UI-side counter.
    ui_episode_done = (
        (total_tasks > 0 and step_idx >= total_tasks - 1)
        or env_done
    )

    # ---- Case: episode was done — advance to next episode ---------------
    if ui_episode_done:
        new_ep_idx = 0 if is_last_ep else ep_idx + 1

        episode = episodes[new_ep_idx]
        result, err = env_reset(env_url, episode["episode_id"])
        if err or result is None:
            return _error_outputs(f"env_reset failed: {err}") + (new_ep_idx, -1, 0, history, False)

        new_total_tasks: int = parse_total_tasks_from_obs(result)
        outputs = _build_reset_outputs(result, new_ep_idx, new_total_tasks, episodes)
        next_task_text = parse_task_from_obs(result).get("prompt", "— no task prompt —")
        new_history = history + [{
            "result":           result,
            "scripted":         None,
            "current_task_text": "No tasks executed yet",
            "next_task_text":    next_task_text,
            "ep_idx":           new_ep_idx,
            "step_idx":         -1,
            "env_done":         False,
            "total_tasks":      new_total_tasks,
        }]
        # Return order: ep_idx, step_idx, total_tasks, history, env_done
        return outputs + (new_ep_idx, -1, new_total_tasks, new_history, False)

    # ---- Case: normal step advance --------------------------------------
    new_step_idx = 0 if step_idx == -1 else step_idx + 1

    episode = episodes[ep_idx]

    # Derive current macros from last history frame (for fallback scripted step)
    last_frame = history[-1] if history else None
    current_task_text = "No tasks executed yet"
    current_macros: List[Dict[str, Any]] = []
    if last_frame and last_frame.get("result"):
        current_task_text = last_frame.get("next_task_text", current_task_text)
        current_macros = extract_macros(parse_tools_from_obs(last_frame["result"]))

    scripted_step = _get_scripted_step(episode, new_step_idx, current_macros)

    # Call env_step via /web/step (stateful WebInterfaceManager)
    result, err = env_step(
        env_url,
        scripted_step["plan"],
        scripted_step.get("macro_proposed"),
    )
    if err or result is None:
        return _error_outputs(f"Env step failed: {err}") + (ep_idx, new_step_idx, total_tasks, history, False)

    # Use env.done only as a belt-and-suspenders guard; primary done signal
    # is the UI-side counter checked at the TOP of this function.
    new_env_done = bool(result.get("done", False))

    outputs = _build_step_outputs(
        result,
        scripted_step,
        current_task_text,
        ep_idx,
        new_step_idx,
        total_tasks,
        episodes,
        new_env_done,
    )

    next_task_text = parse_task_from_obs(result).get("prompt", "— no task prompt —")

    new_history = history + [{
        "result":           result,
        "scripted":         scripted_step,
        "current_task_text": current_task_text,
        "next_task_text":    next_task_text,
        "ep_idx":           ep_idx,
        "step_idx":         new_step_idx,
        "env_done":         new_env_done,
        "total_tasks":      total_tasks,
    }]

    # Return order: ep_idx, step_idx, total_tasks, history, env_done
    return outputs + (ep_idx, new_step_idx, total_tasks, new_history, new_env_done)


def on_prev_step(
    ep_idx:      int,
    step_idx:    int,
    total_tasks: int,
    episodes:    List[Dict[str, Any]],
    history:     List[Dict[str, Any]],
    env_done:    bool,
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_prev_step
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Step backwards through already-executed frames using the history
    buffer.  Does NOT re-call the env (env state is not reversible).

    Returns:
        13 display outputs + new ep_idx + new step_idx + new total_tasks
        + trimmed history + restored env_done + total_tasks.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not history or len(history) < 2:
        # Already at the start — return unchanged
        if not history:
            return _blank_outputs() + (ep_idx, step_idx, total_tasks, history, env_done)
        frame = history[-1]
        frame_total = frame.get("total_tasks", total_tasks)
        if frame.get("scripted") is None:
            outputs = _build_reset_outputs(frame["result"], frame["ep_idx"], frame_total, episodes)
        else:
            outputs = _build_step_outputs(
                frame["result"], frame["scripted"],
                frame.get("current_task_text", "No tasks executed yet"),
                frame["ep_idx"], frame["step_idx"],
                frame_total, episodes, frame.get("env_done", False),
            )
        # Return order: ep_idx, step_idx, total_tasks, history, env_done
        return outputs + (
            frame["ep_idx"], frame["step_idx"], frame_total,
            history, frame.get("env_done", False),
        )

    # Pop the latest frame and show the previous one
    prev_frame  = history[-2]
    trimmed     = history[:-1]
    frame_total = prev_frame.get("total_tasks", total_tasks)

    if prev_frame.get("scripted") is None:
        outputs = _build_reset_outputs(prev_frame["result"], prev_frame["ep_idx"], frame_total, episodes)
    else:
        outputs = _build_step_outputs(
            prev_frame["result"], prev_frame["scripted"],
            prev_frame.get("current_task_text", "No tasks executed yet"),
            prev_frame["ep_idx"], prev_frame["step_idx"],
            frame_total, episodes, prev_frame.get("env_done", False),
        )
    # Return order: ep_idx, step_idx, total_tasks, history, env_done
    return outputs + (
        prev_frame["ep_idx"], prev_frame["step_idx"], frame_total,
        trimmed, prev_frame.get("env_done", False),
    )


def on_auto_tick(
    auto_active: bool,
    ep_idx:      int,
    step_idx:    int,
    total_tasks: int,
    episodes:    List[Dict[str, Any]],
    env_url:     str,
    history:     List[Dict[str, Any]],
    env_done:    bool,
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
            return _blank_outputs() + (0, -1, 0, [], False)
        # Return current frame unchanged
        if history:
            frame = history[-1]
            frame_total = frame.get("total_tasks", total_tasks)
            if frame.get("scripted") is None:
                outputs = _build_reset_outputs(frame["result"], frame["ep_idx"], frame_total, episodes)
            else:
                outputs = _build_step_outputs(
                    frame["result"], frame["scripted"],
                    frame.get("current_task_text", "No tasks executed yet"),
                    frame["ep_idx"], frame["step_idx"],
                    frame_total, episodes, frame.get("env_done", False),
                )
            # Return order: ep_idx, step_idx, total_tasks, history, env_done
            return outputs + (
                frame["ep_idx"], frame["step_idx"], frame_total,
                history, frame.get("env_done", False),
            )
        return _blank_outputs() + (ep_idx, step_idx, total_tasks, history, env_done)

    return on_next_step(ep_idx, step_idx, total_tasks, episodes, env_url, history, env_done)


def on_test_env(env_url: str) -> str:
    """Check env health and return a status HTML string."""
    _, msg = check_env_health(env_url)
    if msg.startswith("✅"):
        return (
            f'<div style="color:#4ade80;padding:8px;border-radius:6px;background:#052e16;">{msg}</div>'
        )
    if msg.startswith("⚠"):
        return (
            f'<div style="color:#fbbf24;padding:8px;border-radius:6px;background:#451a03;">{msg}</div>'
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
                    value="No tasks executed yet",
                    lines=3,
                    interactive=False,
                )

                next_task_box = gr.Textbox(
                    label="📌 Next Task",
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

        # Index of current step within the episode (-1 = reset, no step yet)
        step_index_state = gr.State(value=-1)

        # Total tasks in the current episode — from env reset() metadata.
        # Used to determine "episode done" without relying on env.done,
        # which is unreliable when the HTTP server creates a fresh env
        # instance per request (stateless REST mode).
        total_tasks_state = gr.State(value=0)

        # Full episode+step list for the active profile
        episodes_state   = gr.State(value=[])

        # History of executed frames for Prev navigation
        # Each entry: {"result": dict, "scripted": dict|None,
        #              "current_task_text": str, "next_task_text": str,
        #              "ep_idx": int, "step_idx": int,
        #              "env_done": bool, "total_tasks": int}
        history_state    = gr.State(value=[])

        # Whether auto-play is active
        auto_active_state = gr.State(value=False)

        # Whether the env signalled episode done (belt-and-suspenders only;
        # primary done signal is total_tasks_state vs step_index_state)
        env_done_state = gr.State(value=False)

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
        # CANONICAL OUTPUT LIST (13 display + state components)
        # -------------------------------------------------------------------
        _display = [
            current_task_box,      # 0
            next_task_box,         # 1
            available_tools_html,  # 2
            episode_progress_box,  # 3
            agent_plan_html,       # 4
            reward_html,           # 5
            turns_used_num,        # 6
            turns_saved_num,       # 7
            macro_library_html,    # 8
            agent_note_box,        # 9
            next_btn,              # 10 — label update
            btn_note_html,         # 11
            step_status_html,      # 12
        ]
        # _step_state: components updated by all navigation handlers.
        # Order must match the tail of every handler's return tuple:
        #   ep_index, step_index, total_tasks, history, env_done
        _step_state = [
            ep_index_state,
            step_index_state,
            total_tasks_state,
            history_state,
            env_done_state,
        ]

        # -------------------------------------------------------------------
        # EVENTS
        # -------------------------------------------------------------------

        # Test env connection
        test_env_btn.click(
            fn=on_test_env,
            inputs=[env_url_field],
            outputs=[env_status_html],
        )

        # Run Simulation → reset env, show first prompt in Next Task (no plan)
        # Inputs: model, env_url  (difficulty removed — always Easy)
        run_btn.click(
            fn=on_run_simulation,
            inputs=[model_selector, env_url_field],
            outputs=_display + [episodes_state] + _step_state,
        )

        # Next Step (also handles episode boundary and restart)
        next_btn.click(
            fn=on_next_step,
            inputs=[ep_index_state, step_index_state, total_tasks_state,
                    episodes_state, env_url_field, history_state, env_done_state],
            outputs=_display + _step_state,
        )

        # Previous Step (navigates history buffer, no env call)
        prev_btn.click(
            fn=on_prev_step,
            inputs=[ep_index_state, step_index_state, total_tasks_state,
                    episodes_state, history_state, env_done_state],
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
                inputs=[auto_active_state, ep_index_state, step_index_state,
                        total_tasks_state, episodes_state, env_url_field,
                        history_state, env_done_state],
                outputs=_display + _step_state,
            )
        else:
            auto_play_btn.click(
                fn=toggle_auto_play,
                inputs=[auto_active_state],
                outputs=[auto_active_state, auto_play_btn, thinking_html],
            )

    return tab
