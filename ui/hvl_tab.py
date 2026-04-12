# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tab 3 — Human vs LLM
======================

An interactive game where the user and an LLM both propose plans for the
same ToolForge task.  After both submit, scores are compared side-by-side.

Flow:
    1. User configures LLM connection (API key or ngrok) and clicks "Start Game".
    2. The env is reset → first real task prompt is shown.
    3. User types their plan (one tool per line) and clicks "Submit My Plan".
    4. Simultaneously:
          a. Human plan → env_step() → gets REAL reward from the env.
          b. LLM plan   → openai API call → scored with local heuristic
             (the env advances after the human step, so we can't replay the
              same task position for LLM; heuristic scoring keeps the game fair).
    5. Both plans and scores are shown side-by-side. Winner announced.
    6. User clicks "Next Task" — env is already on the next task (advanced by
       the human's env_step in step 4a), so we just read the new task from obs.
    7. A running scoreboard tracks cumulative Human vs LLM scores.
    8. After all tasks are exhausted (env returns done=True), a final summary
       is shown.

Scoring:
    - Human  : REAL reward from env_step() — same pipeline as the benchmark.
    - LLM    : Local heuristic (slot coverage + efficiency) — labelled clearly
               in the UI so the comparison is honest.

Import pattern (runs from toolforge_env/ dir):
    from ui.hvl_tab import build_hvl_tab
"""

import json
import logging
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import httpx

from ui.shared import (
    ATOMIC_TOOLS,
    SYSTEM_PROMPT,
    TOOL_DESCRIPTIONS,
    render_macro_library_html,
    render_plan_html,
    render_reward_html,
)
from ui.env_client import (
    DEFAULT_ENV_URL,
    env_reset,
    env_step,
    extract_macros,
    parse_task_from_obs,
    parse_tools_from_obs,
    parse_total_tasks_from_obs,
)

# ---------------------------------------------------------------------------
# Module-level logger — NEVER log api_key values
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inference constants (matches byoa_tab / inference.py)
# ---------------------------------------------------------------------------
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 500
TASK_ID: str = "easy"   # HvL game always uses easy difficulty

# ---------------------------------------------------------------------------
# Slot→tool hint map used for local LLM plan heuristic scoring
# ---------------------------------------------------------------------------
_SLOT_TOOL_HINTS: Dict[str, List[str]] = {
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
    "patch_execution":         ["patch"],
    "test_execution":          ["run_tests"],
    "alert_notification":      ["pagerduty_alert", "notify"],
}


# ===========================================================================
# SECTION 1: CONNECTION HELPERS
# ===========================================================================

def _make_llm_client(mode: str, base_url: str, api_key: str, ngrok_url: str):
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _make_llm_client
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Instantiate an openai.OpenAI client from the HvL connection settings.

    Args:
        mode     : "API Key (OpenAI-compatible)" or "Local Model via ngrok"
        base_url : API endpoint (API key mode)
        api_key  : User API key (API key mode) — NEVER logged
        ngrok_url: ngrok tunnel URL (ngrok mode)

    Returns:
        openai.OpenAI client instance
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    if mode == "Local Model via ngrok":
        return OpenAI(base_url=ngrok_url.rstrip("/"), api_key="local")
    else:
        url = base_url.rstrip("/") or "https://api.openai.com/v1"
        return OpenAI(base_url=url, api_key=api_key)


def on_hvl_mode_change(mode: str) -> Tuple:
    """Show/hide API key vs ngrok connection sections."""
    is_api = mode == "API Key (OpenAI-compatible)"
    return gr.update(visible=is_api), gr.update(visible=not is_api)


def on_test_connection_api(base_url: str, model_name: str, api_key: str) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_test_connection_api
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Test API key connection by calling GET /models, with fallback to
    a 1-token completion.

    Returns HTML status string.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not api_key.strip():
        return _status_html("❌ API key is empty.", error=True)
    if not base_url.strip():
        return _status_html("❌ Base URL is empty.", error=True)
    if not model_name.strip():
        return _status_html("❌ Model name is empty.", error=True)

    try:
        client = _make_llm_client("API Key (OpenAI-compatible)", base_url, api_key, "")
        try:
            client.models.list()
            return _status_html(f"✅ Connected to {base_url} — model list OK. Ready to play.")
        except Exception:
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0.0,
            )
            return _status_html(f"✅ Connected ({model_name}) — completion test passed.")
    except Exception as exc:
        return _status_html(f"❌ Connection failed: {exc}", error=True)


def on_test_connection_ngrok(ngrok_url: str, model_name: str) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_test_connection_ngrok
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Ping the ngrok /v1/models endpoint to verify the local server is
    reachable.

    Returns HTML status string.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not ngrok_url.strip():
        return _status_html("❌ ngrok URL is empty.", error=True)
    if not model_name.strip():
        return _status_html("❌ Model name is empty.", error=True)

    probe_url = ngrok_url.rstrip("/") + "/models"
    try:
        resp = httpx.get(probe_url, timeout=8.0)
        if resp.status_code == 200:
            return _status_html(f"✅ ngrok server reachable — models endpoint OK.")
        else:
            return _status_html(
                f"⚠️ Server responded HTTP {resp.status_code}. May still work — proceed with caution.",
                warn=True,
            )
    except Exception as exc:
        return _status_html(f"❌ Cannot reach {ngrok_url}: {exc}", error=True)


def _status_html(msg: str, error: bool = False, warn: bool = False) -> str:
    """Return a coloured status <div> string."""
    if error:
        bg, fg = "#450a0a", "#f87171"
    elif warn:
        bg, fg = "#451a03", "#fbbf24"
    else:
        bg, fg = "#052e16", "#4ade80"
    return f'<div style="color:{fg};padding:8px;border-radius:6px;background:{bg};">{msg}</div>'


# ===========================================================================
# SECTION 2: LLM PLAN FETCHER
# ===========================================================================

def _get_llm_plan(
    client,
    model_name: str,
    task_prompt: str,
    available_tools: List[Dict[str, Any]],
) -> Tuple[List[str], Optional[str]]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _get_llm_plan
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Ask the LLM for a plan for the given task prompt.
    Returns (plan_list, error_str).  On failure returns a sensible
    fallback plan so the game can continue.

    Args:
        client          : openai.OpenAI client
        model_name      : LLM model identifier
        task_prompt     : Task description shown to the LLM
        available_tools : List of tool dicts (name, description)

    Returns:
        Tuple (plan_names: List[str], error: str | None)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    try:
        from models import ToolForgeAction, ToolCall  # type: ignore
    except ImportError:
        # Fallback to a heuristic plan if models can't be imported
        return _heuristic_llm_plan(task_prompt), None

    user_prompt = textwrap.dedent(f"""
        Task: {task_prompt}
        Available tools: {json.dumps([t['name'] for t in available_tools])}

        Choose the minimal plan (ordered list of tool names) that best completes this task.
        Return ONLY valid JSON matching the ToolForgeAction schema — no markdown, no explanation.
    """).strip()

    system = (
        SYSTEM_PROMPT
        + "\nRespond ONLY with valid JSON. No markdown. No explanation."
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        action = ToolForgeAction(**json.loads(raw))
        plan = [tc.tool_name for tc in action.plan]
        return plan, None
    except Exception as exc:
        logger.warning("LLM plan parse failed: %s — using heuristic fallback", exc)
        return _heuristic_llm_plan(task_prompt), str(exc)


def _heuristic_llm_plan(task_prompt: str) -> List[str]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _heuristic_llm_plan
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Fallback LLM plan derived from keyword matching on the task prompt.
    Used when the API call fails or the response cannot be parsed.

    Args:
        task_prompt : The task description text.

    Returns:
        List of tool name strings.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    prompt_lower = task_prompt.lower()
    if "rollback" in prompt_lower:
        return ["rollback", "healthcheck", "notify"]
    if "scale" in prompt_lower or "replicas" in prompt_lower:
        return ["scale", "ping", "notify"]
    if "restart" in prompt_lower:
        return ["restart", "healthcheck", "notify"]
    if "patch" in prompt_lower:
        return ["patch", "healthcheck", "notify"]
    # Default: deploy → healthcheck → notify
    return ["deploy", "healthcheck", "notify"]


# ===========================================================================
# SECTION 3: LOCAL HEURISTIC SCORER (for LLM plan only)
# ===========================================================================

def _heuristic_score(
    plan: List[str],
    required_slots: List[str],
    baseline: int,
    valid_tool_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _heuristic_score
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Score a plan locally using slot-coverage heuristics.
    Used ONLY for the LLM plan — the human plan is scored by the real env.

    Args:
        plan             : List of tool name strings.
        required_slots   : Slot names the task requires.
        baseline         : baseline_call_count for the task.
        valid_tool_names : Tool names accepted by the env (atomic + macros).
                           Falls back to ATOMIC_TOOLS if not supplied.

    Returns:
        Dict with keys: reward (float), score_100 (int), note (str).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # Use env-supplied tool names (includes accepted macros) if available;
    # fall back to the static list only when env data is absent.
    known_names    = valid_tool_names if valid_tool_names else ATOMIC_TOOLS
    all_tools_lower = {t.lower() for t in known_names}
    valid_plan = [t for t in plan if t.lower() in all_tools_lower]

    if not valid_plan:
        return {"reward": -0.20, "score_100": 0, "note": "No valid tools in plan."}

    filled = sum(
        1 for slot in required_slots
        if any(t in valid_plan for t in _SLOT_TOOL_HINTS.get(slot, []))
    )
    slot_ratio = filled / len(required_slots) if required_slots else 0.0

    if slot_ratio == 1.0:
        eff  = max(0.0, min(0.5, (baseline - len(valid_plan)) / max(baseline, 1) * 0.5 + 0.25))
        reward = min(1.0, 0.40 + eff)
    elif slot_ratio >= 0.65:
        reward = 0.25 * slot_ratio
    else:
        reward = max(-0.15, -0.15 + slot_ratio * 0.23)

    reward    = max(-0.20, min(1.0, reward))
    score_100 = int((reward + 0.20) / 1.20 * 100)
    note      = f"Filled {filled}/{len(required_slots)} required slots. Plan length: {len(valid_plan)} (baseline: {baseline}). *(local estimate)*"

    return {"reward": round(reward, 2), "score_100": score_100, "note": note}


def _parse_human_plan(text: str, valid_tool_names: Optional[List[str]] = None) -> List[str]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _parse_human_plan
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Parse tool names from the human's plain-text input (one per line).

    Validation is done against `valid_tool_names` (env-supplied, includes
    accepted macros) so the human can use macro names they've seen created.
    Falls back to ATOMIC_TOOLS when env data is not available.

    Args:
        text             : Raw textarea content.
        valid_tool_names : Tool names accepted by the env (atomic + macros).

    Returns:
        Ordered list of valid tool name strings (preserving original casing
        from the env list).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # Build a lower→original mapping from env tools (or static fallback)
    known_names     = valid_tool_names if valid_tool_names else ATOMIC_TOOLS
    all_tools_lower = {t.lower(): t for t in known_names}
    lines = [line.strip().lower() for line in text.strip().splitlines()]
    return [all_tools_lower[l] for l in lines if l in all_tools_lower]


# ===========================================================================
# SECTION 4: RENDERING HELPERS
# ===========================================================================

def _render_tool_reference(tools_list: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _render_tool_reference
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Render the available tool list as an HTML reference card.

    Uses `tools_list` from the env observation (atomic tools + any
    accepted macros) when available.  Falls back to the static
    ATOMIC_TOOLS + TOOL_DESCRIPTIONS when the env has not been called
    yet (e.g. initial widget render before Start Game).

    Args:
        tools_list : List of tool dicts from parse_tools_from_obs().
                     Each dict has keys: name, description, is_macro, steps.

    Returns:
        HTML string.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    rows = ""

    if tools_list:
        # Use env-supplied tools — separates atomics from macros visually
        for tool in tools_list:
            name    = tool.get("name", "")
            desc    = tool.get("description", "")
            is_macro = tool.get("is_macro", False)
            if is_macro:
                steps_preview = " → ".join(
                    s.get("tool_name", s) if isinstance(s, dict) else str(s)
                    for s in tool.get("steps", [])
                )
                rows += (
                    f'<div style="margin:4px 0;padding:4px 10px;border-radius:6px;'
                    f'background:#2e1065;border-left:3px solid #7c3aed;">'
                    f'<code style="color:#c4b5fd;">⚙ {name}</code>'
                    f'<span style="color:#a78bfa;font-size:0.78em;margin-left:8px;">{steps_preview}</span>'
                    f'</div>'
                )
            else:
                rows += (
                    f'<div style="margin:4px 0;padding:4px 10px;border-radius:6px;background:#1f2937;">'
                    f'<code style="color:#a78bfa;">{name}</code>'
                    f'<span style="color:#9ca3af;font-size:0.82em;margin-left:8px;">{desc}</span>'
                    f'</div>'
                )
    else:
        # Fallback: static list (shown before env is reset)
        for tool in ATOMIC_TOOLS:
            desc = TOOL_DESCRIPTIONS.get(tool, "")
            rows += (
                f'<div style="margin:4px 0;padding:4px 10px;border-radius:6px;background:#1f2937;">'
                f'<code style="color:#a78bfa;">{tool}</code>'
                f'<span style="color:#9ca3af;font-size:0.82em;margin-left:8px;">{desc}</span>'
                f'</div>'
            )

    return f'<div style="padding:4px 0;">{rows}</div>'


def _render_score_card(score_100: int, reward: float, player: str) -> str:
    """Render a score card HTML block."""
    colour = "#22c55e" if score_100 >= 70 else ("#f59e0b" if score_100 >= 45 else "#ef4444")
    return (
        f'<div style="text-align:center;padding:12px;">'
        f'<div style="font-size:2.6em;font-weight:800;color:{colour};">'
        f'{score_100}<span style="font-size:0.45em;opacity:0.6;">/100</span></div>'
        f'<div style="color:#9ca3af;font-size:0.78em;margin-top:4px;">'
        f'raw reward: {reward:+.2f}</div>'
        f'</div>'
    )


def _render_slot_breakdown(required_slots: List[str], human_plan: List[str], llm_plan: List[str]) -> str:
    """Render slot breakdown table HTML."""
    rows = ""
    for slot in required_slots:
        hints = _SLOT_TOOL_HINTS.get(slot, [])
        h = "✅" if any(t in human_plan for t in hints) else "❌"
        l = "✅" if any(t in llm_plan  for t in hints) else "❌"
        rows += (
            f"<tr>"
            f"<td style='padding:5px 8px;color:#d1d5db;'>{slot}</td>"
            f"<td style='text-align:center;padding:5px 8px;'>{h}</td>"
            f"<td style='text-align:center;padding:5px 8px;'>{l}</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:0.83em;'>"
        "<thead><tr style='color:#9ca3af;background:#1f2937;'>"
        "<th style='text-align:left;padding:5px 8px;'>Slot</th>"
        "<th style='padding:5px 8px;'>You</th>"
        "<th style='padding:5px 8px;'>LLM</th>"
        "</tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def _winner_html(human_score: int, llm_score: int) -> str:
    """Render the winner banner HTML for this round."""
    if human_score > llm_score:
        return '<div class="winner-banner" style="background:#14532d;color:#4ade80;padding:16px;border-radius:12px;font-size:1.4em;font-weight:800;text-align:center;">🏆 You Win This Round!</div>'
    if llm_score > human_score:
        return '<div class="winner-banner" style="background:#450a0a;color:#f87171;padding:16px;border-radius:12px;font-size:1.4em;font-weight:800;text-align:center;">🤖 LLM Wins This Round!</div>'
    return '<div class="winner-banner" style="background:#1e3a5f;color:#93c5fd;padding:16px;border-radius:12px;font-size:1.4em;font-weight:800;text-align:center;">🤝 It\'s a Tie!</div>'


# ===========================================================================
# SECTION 5: GAME EVENT HANDLERS
# ===========================================================================

def on_start_game(
    mode: str,
    base_url: str,
    api_model: str,
    api_key: str,
    ngrok_url: str,
    ngrok_model: str,
    env_url: str,
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_start_game
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Reset the environment, get the first task, and show the game area.

    Accepts both connection modes so the handler has everything needed
    if the user later calls the LLM.

    Returns:
        Tuple updating: status, game_area, task_display, tool_reference,
        scoreboard, results visibility, and all hidden state components.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    logger.info("HvL game started | mode=%s env=%s", mode, env_url)

    # Reset env to get the first task
    result, err = env_reset(env_url, TASK_ID)
    if err or result is None:
        return (
            _status_html(f"❌ Cannot start game — env_reset failed: {err}", error=True),
            gr.update(visible=False),   # game_area stays hidden
            "", "", "", "",              # task, tools, scoreboard, llm_placeholder
            gr.update(visible=False),   # results_row
            None, None, None, None,     # obs, total_tasks, human_score, llm_score
        )

    total_tasks = parse_total_tasks_from_obs(result)
    task_info   = parse_task_from_obs(result)
    tools_list  = parse_tools_from_obs(result)   # from env — includes atomics + macros
    task_prompt = task_info.get("prompt", "No task found.")

    status_msg  = _status_html(
        f"✅ Game started — {total_tasks} tasks loaded. Good luck!"
    )

    return (
        status_msg,
        gr.update(visible=True),             # show game_area
        task_prompt,                         # task_display
        _render_tool_reference(tools_list),  # tool_reference_html — from env
        "You: 0  |  LLM: 0",           # scoreboard_box
        "Waiting for your submission…", # llm_placeholder
        gr.update(visible=False),       # results_row — hide at start
        result,                         # last_obs_state (full obs dict)
        total_tasks,                    # total_tasks_state
        0,                              # human_score_state
        0,                              # llm_score_state
    )


def on_submit_human_plan(
    human_plan_text: str,
    # LLM connection fields
    mode: str,
    base_url: str,
    api_model: str,
    api_key: str,
    ngrok_url: str,
    ngrok_model: str,
    env_url: str,
    # Game state
    last_obs_state: Optional[Dict],
    total_tasks_state: int,
    human_score_state: int,
    llm_score_state: int,
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_submit_human_plan
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Called when the user clicks "Submit My Plan".

    Steps:
        1. Parse and validate the human's plan.
        2. Call env_step(human_plan) → get REAL human reward.
        3. Call LLM API → get LLM's plan for the same task.
        4. Score LLM plan with local heuristic.
        5. Update scoreboard, show winner banner, reveal results.
        6. Store the updated observation (next task) in state.

    Returns:
        Large tuple updating all result + state components.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # ---- guard: no game in progress ----------------------------------------
    if last_obs_state is None:
        return _blank_submit_returns(human_score_state, llm_score_state, last_obs_state, total_tasks_state)

    task_info   = parse_task_from_obs(last_obs_state)
    tools_list  = parse_tools_from_obs(last_obs_state)   # env-supplied — atomics + macros
    task_prompt = task_info.get("prompt", "")
    required_slots    = task_info.get("required_slots", [])
    baseline          = task_info.get("baseline_call_count", 3)
    tools_for_llm     = [{"name": t.get("name",""), "description": t.get("description","")} for t in tools_list]
    # All valid tool names from env (atomics + accepted macros) for validation
    valid_tool_names  = [t.get("name","") for t in tools_list if t.get("name")]

    # ---- parse human plan (validated against env's tool list) ---------------
    human_plan = _parse_human_plan(human_plan_text, valid_tool_names)
    if not human_plan:
        # Nothing valid — show a warning but don't advance
        error_banner = '<div style="color:#fbbf24;padding:10px;background:#451a03;border-radius:8px;">⚠️ No valid tool names found. Enter one tool name per line.</div>'
        return (
            gr.update(visible=True),    # results_row
            "<p style='color:#f87171;'>No valid plan submitted.</p>",
            '<div style="text-align:center;font-size:2em;font-weight:800;color:#ef4444;">0<span style="font-size:0.45em;">/100</span></div>',
            "<p style='color:#9ca3af;'>—</p>",
            '<div style="text-align:center;font-size:2em;font-weight:800;color:#9ca3af;">—</div>',
            error_banner,               # winner_banner_html
            "",                         # slot_table
            f"You: {human_score_state}  |  LLM: {llm_score_state}",
            last_obs_state,             # obs unchanged
            total_tasks_state,
            human_score_state,
            llm_score_state,
            gr.update(interactive=True),
        )

    # ---- call env_step with human plan → real human reward ------------------
    step_result, step_err = env_step(env_url, human_plan)
    if step_err or step_result is None:
        error_banner = _status_html(f"❌ env_step failed: {step_err}", error=True)
        return (
            gr.update(visible=True),
            render_plan_html(human_plan, []),
            '<div style="text-align:center;color:#ef4444;">Error</div>',
            "<p>—</p>",
            '<div style="text-align:center;color:#9ca3af;">—</div>',
            error_banner,
            "",
            f"You: {human_score_state}  |  LLM: {llm_score_state}",
            last_obs_state,
            total_tasks_state,
            human_score_state,
            llm_score_state,
            gr.update(interactive=True),
        )

    human_reward = float(step_result.get("reward", 0.0))
    human_score_100 = int((human_reward + 0.20) / 1.20 * 100)
    new_obs = step_result   # after step, obs has the NEXT task ready

    # ---- call LLM for its plan (async in sync context) ----------------------
    llm_plan: List[str] = []
    llm_err_note: Optional[str] = None
    model_name = ngrok_model if mode == "Local Model via ngrok" else api_model

    if model_name.strip():
        try:
            client = _make_llm_client(mode, base_url, api_key, ngrok_url)
            llm_plan, llm_err_note = _get_llm_plan(client, model_name, task_prompt, tools_for_llm)
        except Exception as exc:
            llm_err_note = str(exc)
            llm_plan = _heuristic_llm_plan(task_prompt)
    else:
        # No model configured — use heuristic fallback silently
        llm_plan = _heuristic_llm_plan(task_prompt)
        llm_err_note = "No model configured — using heuristic fallback."

    # ---- score LLM plan locally (pass env tool names so macros are recognised)
    llm_result     = _heuristic_score(llm_plan, required_slots, baseline, valid_tool_names)
    llm_score_100  = llm_result["score_100"]
    llm_reward_est = llm_result["reward"]

    # ---- scoreboard update --------------------------------------------------
    new_human_score = human_score_state + human_score_100
    new_llm_score   = llm_score_state   + llm_score_100
    scoreboard_text = f"You: {new_human_score}  |  LLM: {new_llm_score}"

    # ---- render outputs -----------------------------------------------------
    macros = extract_macros(tools_list)

    human_plan_html  = render_plan_html(human_plan, macros)
    human_score_card = _render_score_card(human_score_100, human_reward, "You")
    llm_plan_html    = render_plan_html(llm_plan, macros)
    llm_score_note   = (" *(local heuristic)*" if not llm_err_note else f" *(heuristic fallback: {llm_err_note[:60]})*")
    llm_score_card   = _render_score_card(llm_score_100, llm_reward_est, "LLM") + f'<div style="font-size:0.72em;color:#6b7280;text-align:center;margin-top:4px;">{llm_score_note}</div>'

    banner    = _winner_html(human_score_100, llm_score_100)
    slot_html = _render_slot_breakdown(required_slots, human_plan, llm_plan)

    return (
        gr.update(visible=True),    # results_row visible
        human_plan_html,            # human_plan_result_html
        human_score_card,           # human_score_html
        llm_plan_html,              # llm_plan_result_html
        llm_score_card,             # llm_score_html
        banner,                     # winner_banner_html
        slot_html,                  # slot_table_html
        scoreboard_text,            # scoreboard_box
        new_obs,                    # last_obs_state ← updated obs (next task)
        total_tasks_state,
        new_human_score,
        new_llm_score,
        gr.update(interactive=True),
    )


def on_next_task(
    last_obs_state: Optional[Dict],
    total_tasks_state: int,
    human_score_state: int,
    llm_score_state: int,
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_next_task
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Advance to the next task.  The env already advanced when the human
    submitted (via env_step), so we just read task info from last_obs_state.

    If the env signalled done=True, show the final summary screen.

    Returns:
        Tuple updating task display, human plan input, llm placeholder,
        results visibility, and state.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if last_obs_state is None:
        return (
            "No game in progress.", "", "Waiting for your submission…",
            gr.update(visible=False),
            _render_tool_reference(),   # fallback static list — no obs
            last_obs_state, total_tasks_state, human_score_state, llm_score_state,
        )

    env_done    = bool(last_obs_state.get("done", False))
    task_info   = parse_task_from_obs(last_obs_state)
    tools_list  = parse_tools_from_obs(last_obs_state)   # refresh from new obs
    task_prompt = task_info.get("prompt", "")

    # Treat done if env says done OR task_prompt looks like a terminal placeholder
    is_terminal = env_done or not task_prompt or task_prompt == "Default task"

    if is_terminal:
        if human_score_state > llm_score_state:
            final_msg = f"🏆 **You Win!** Final: You {human_score_state} – LLM {llm_score_state}"
        elif llm_score_state > human_score_state:
            final_msg = f"🤖 **LLM Wins!** Final: You {human_score_state} – LLM {llm_score_state}"
        else:
            final_msg = f"🤝 **It's a Tie!** Final: You {human_score_state} – LLM {llm_score_state}"

        return (
            f"🎉 Game Over! {final_msg}",
            "",
            "Game over — no more tasks.",
            gr.update(visible=False),
            _render_tool_reference(),   # fallback — game over
            None,                       # clear obs state
            total_tasks_state,
            human_score_state,
            llm_score_state,
        )

    return (
        task_prompt,
        "",                                       # clear human plan input
        "Waiting for your submission…",
        gr.update(visible=False),                 # hide previous results
        _render_tool_reference(tools_list),       # refresh tool list from new obs
        last_obs_state,
        total_tasks_state,
        human_score_state,
        llm_score_state,
    )


def _blank_submit_returns(human_score: int, llm_score: int, obs, total_tasks: int) -> Tuple:
    """Safe blank return tuple for on_submit_human_plan when game not started."""
    return (
        gr.update(visible=True),
        "<p style='color:#9ca3af;'>Start the game first.</p>",
        '<div style="text-align:center;color:#9ca3af;">—</div>',
        "<p style='color:#9ca3af;'>—</p>",
        '<div style="text-align:center;color:#9ca3af;">—</div>',
        _status_html("⚠️ Click 'Start Game' first.", warn=True),
        "",
        f"You: {human_score}  |  LLM: {llm_score}",
        obs,
        total_tasks,
        human_score,
        llm_score,
        gr.update(interactive=True),
    )


# ===========================================================================
# SECTION 6: TAB BUILDER
# ===========================================================================

def build_hvl_tab() -> gr.Tab:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    build_hvl_tab
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Construct and return the "Human vs LLM" gr.Tab component.
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
            You and an LLM both propose a DevOps tool plan for the same task.
            Your plan is scored by the **real ToolForge evaluation pipeline**.
            The LLM plan is scored with a local heuristic (labelled in the results).
            """
        )
        gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

        # -------------------------------------------------------------------
        # CONNECTION SECTION
        # -------------------------------------------------------------------
        gr.Markdown("### Connect Your LLM  *(optional — needed for real LLM opponent)*")

        hvl_connection_mode = gr.Radio(
            choices=["API Key (OpenAI-compatible)", "Local Model via ngrok"],
            value="API Key (OpenAI-compatible)",
            label="Connection Mode",
        )

        # --- API KEY SUB-SECTION ---
        with gr.Group(visible=True) as hvl_api_section:
            with gr.Row():
                hvl_base_url = gr.Textbox(
                    value="https://api.openai.com/v1",
                    label="Base URL",
                    placeholder="https://api.openai.com/v1",
                    scale=3,
                )
                hvl_api_model = gr.Textbox(
                    value="",
                    label="Model Name",
                    placeholder="e.g. gpt-4o, mistral-small",
                    scale=2,
                )
            hvl_api_key = gr.Textbox(
                value="",
                label="API Key  (used only in your session — never stored or logged)",
                placeholder="sk-...",
                type="password",
            )
            with gr.Row():
                test_api_btn     = gr.Button("Test Connection", variant="secondary", scale=1)
                api_status_html  = gr.HTML(value="", scale=3)

        # --- NGROK SUB-SECTION ---
        with gr.Group(visible=False) as hvl_ngrok_section:
            with gr.Row():
                hvl_ngrok_url   = gr.Textbox(
                    value="",
                    label="ngrok Tunnel URL",
                    placeholder="https://xxxx.ngrok-free.app/v1",
                    scale=3,
                )
                hvl_ngrok_model = gr.Textbox(
                    value="",
                    label="Model Name",
                    placeholder="e.g. llama3, phi3",
                    scale=2,
                )
            with gr.Row():
                test_ngrok_btn    = gr.Button("Test Connection", variant="secondary", scale=1)
                ngrok_status_html = gr.HTML(value="", scale=3)

        gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

        # -------------------------------------------------------------------
        # ENV URL + START
        # -------------------------------------------------------------------
        with gr.Row():
            hvl_env_url = gr.Textbox(
                value=DEFAULT_ENV_URL,
                label="ToolForge Env URL",
                placeholder="http://localhost:8000",
                scale=3,
            )
            start_btn = gr.Button("🎮 Start Game", variant="primary", scale=2)

        start_status_html = gr.HTML(value="")

        # -------------------------------------------------------------------
        # GAME AREA (hidden until Start Game is clicked)
        # -------------------------------------------------------------------
        with gr.Column(visible=False) as game_area:

            # Running scoreboard
            scoreboard_box = gr.Textbox(
                label="📊 Scoreboard",
                value="You: 0  |  LLM: 0",
                interactive=False,
            )

            gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

            # Current task
            task_display = gr.Textbox(
                label="📋 Current Task",
                value="",
                lines=4,
                interactive=False,
            )

            # Two-column layout
            with gr.Row():

                # ===== LEFT: Human's Plan =====
                with gr.Column(scale=1):
                    gr.Markdown("### 🧑 Your Plan")
                    gr.Markdown("**Available Tools** *(enter names one per line)*")
                    tool_reference_html = gr.HTML(value=_render_tool_reference())

                    human_plan_input = gr.Textbox(
                        label="Your plan (one tool name per line)",
                        placeholder="deploy\nhealthcheck\nnotify",
                        lines=6,
                    )
                    submit_plan_btn = gr.Button("Submit My Plan ▶", variant="primary")

                # ===== RIGHT: LLM's Plan =====
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
            with gr.Column(visible=False) as results_row:

                gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")
                gr.Markdown("### Results")

                winner_banner_html = gr.HTML(value="")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**🧑 Your Score**  *(real env reward)*")
                        human_score_html = gr.HTML(
                            value='<div style="text-align:center;font-size:2em;font-weight:800;">—</div>'
                        )
                        gr.Markdown("**Your Plan:**")
                        human_plan_result_html = gr.HTML(value="<p>—</p>")

                    with gr.Column(scale=1):
                        gr.Markdown("**🤖 LLM Score**  *(local heuristic)*")
                        llm_score_html = gr.HTML(
                            value='<div style="text-align:center;font-size:2em;font-weight:800;">—</div>'
                        )
                        gr.Markdown("**LLM Plan:**")
                        llm_plan_result_html = gr.HTML(value="<p>—</p>")

                gr.Markdown("**Slot Breakdown:**")
                slot_table_html = gr.HTML(value="")

                next_task_btn = gr.Button("Next Task ▶", variant="secondary")

        # -------------------------------------------------------------------
        # HIDDEN STATE
        # -------------------------------------------------------------------

        # Full observation dict returned by env_reset or env_step
        last_obs_state    = gr.State(value=None)

        # Total tasks in the episode (from reset metadata)
        total_tasks_state = gr.State(value=0)

        # Cumulative scores (sum of score_100 per round)
        human_score_state = gr.State(value=0)
        llm_score_state   = gr.State(value=0)

        # -------------------------------------------------------------------
        # EVENT WIRING
        # -------------------------------------------------------------------

        # Connection mode toggle
        hvl_connection_mode.change(
            fn=on_hvl_mode_change,
            inputs=[hvl_connection_mode],
            outputs=[hvl_api_section, hvl_ngrok_section],
        )

        # Test connection — API key mode
        test_api_btn.click(
            fn=on_test_connection_api,
            inputs=[hvl_base_url, hvl_api_model, hvl_api_key],
            outputs=[api_status_html],
        )

        # Test connection — ngrok mode
        test_ngrok_btn.click(
            fn=on_test_connection_ngrok,
            inputs=[hvl_ngrok_url, hvl_ngrok_model],
            outputs=[ngrok_status_html],
        )

        # Start Game — reset env, load first task
        _start_outputs = [
            start_status_html,
            game_area,
            task_display,
            tool_reference_html,
            scoreboard_box,
            llm_plan_placeholder,
            results_row,
            last_obs_state,
            total_tasks_state,
            human_score_state,
            llm_score_state,
        ]

        start_btn.click(
            fn=on_start_game,
            inputs=[
                hvl_connection_mode,
                hvl_base_url,
                hvl_api_model,
                hvl_api_key,
                hvl_ngrok_url,
                hvl_ngrok_model,
                hvl_env_url,
            ],
            outputs=_start_outputs,
        )

        # Submit human plan — score both, reveal results
        _submit_inputs = [
            human_plan_input,
            hvl_connection_mode,
            hvl_base_url,
            hvl_api_model,
            hvl_api_key,
            hvl_ngrok_url,
            hvl_ngrok_model,
            hvl_env_url,
            last_obs_state,
            total_tasks_state,
            human_score_state,
            llm_score_state,
        ]

        _submit_outputs = [
            results_row,
            human_plan_result_html,
            human_score_html,
            llm_plan_result_html,
            llm_score_html,
            winner_banner_html,
            slot_table_html,
            scoreboard_box,
            last_obs_state,
            total_tasks_state,
            human_score_state,
            llm_score_state,
            submit_plan_btn,
        ]

        submit_plan_btn.click(
            fn=on_submit_human_plan,
            inputs=_submit_inputs,
            outputs=_submit_outputs,
        )

        # Next Task — read next task from obs (env already advanced);
        # also refreshes tool_reference_html since new obs may contain new macros.
        next_task_btn.click(
            fn=on_next_task,
            inputs=[last_obs_state, total_tasks_state, human_score_state, llm_score_state],
            outputs=[
                task_display,
                human_plan_input,
                llm_plan_placeholder,
                results_row,
                tool_reference_html,    # refreshed from new obs (may have macros)
                last_obs_state,
                total_tasks_state,
                human_score_state,
                llm_score_state,
            ],
        )

    return tab
