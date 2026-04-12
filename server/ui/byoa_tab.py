# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tab 2 — Bring Your Own Agent (BYOA)
====================================

Lets users connect their own LLM and run it against the real ToolForge
environment using a real inference loop that mirrors inference.py.

Connection modes:
    1. API Key (OpenAI-compatible) — base_url + model_name + api_key
    2. Local Model via ngrok       — ngrok_url + model_name

Results:
    - Episode-by-episode table (Episode | Task Summary | Plan | Reward | Turns | Macro used)
    - Live training data JSON block + real tempfile download
    - Macro library display

Import pattern (runs from toolforge_env/ dir):
    from server.ui.byoa_tab import build_byoa_tab
"""

import json
import logging
import os
import tempfile
import textwrap
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
import httpx

# ---------------------------------------------------------------------------
# Inline-style HTML helpers — no CSS class dependency, always visible even
# when this child app is mounted inside OpenEnv's root TabbedInterface.
# ---------------------------------------------------------------------------
_DIVIDER = (
    "<div style='border:none;border-top:3px solid #7c3aed;"
    "margin:22px 0 18px;opacity:0.65;'></div>"
)

def _sec_hdr(text: str) -> str:
    """Bold section title with purple left-bar accent."""
    return (
        f"<div style='border-left:4px solid #7c3aed;padding:5px 14px;"
        f"margin:4px 0 14px;font-size:1.05em;font-weight:700;color:#e2e8f0;'>"
        f"{text}</div>"
    )

from server.ui.shared import (
    ATOMIC_TOOLS,
    SYSTEM_PROMPT,
    render_macro_library_html,
    render_plan_html,
    render_reward_html,
    render_tools_html,
)
from server.ui.env_client import (
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
# Inference constants (mirrors inference.py)
# ---------------------------------------------------------------------------
MAX_STEPS_PER_EPISODE: int = 20   # max LLM turns before force-advancing
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 500
TASK_ID: str = "easy"             # BYOA always runs easy (matches demo mode)


# ===========================================================================
# SECTION 1: CONNECTION HELPERS
# ===========================================================================

def _make_openai_client(mode: str, base_url: str, api_key: str, ngrok_url: str):
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _make_openai_client
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Instantiate an openai.OpenAI client from the current connection settings.

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
        url = ngrok_url.rstrip("/")
        return OpenAI(base_url=url, api_key="local")
    else:
        url = base_url.rstrip("/") or "https://api.openai.com/v1"
        return OpenAI(base_url=url, api_key=api_key)


def on_mode_change(mode: str) -> Tuple:
    """Show/hide API key section vs ngrok section."""
    is_api = mode == "API Key (OpenAI-compatible)"
    return gr.update(visible=is_api), gr.update(visible=not is_api)


def on_test_connection_api(base_url: str, model_name: str, api_key: str) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_test_connection_api
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Test an API key connection by calling GET /models (lightweight probe).
    Falls back to a minimal completion if models endpoint is unavailable.

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
        client = _make_openai_client("API Key (OpenAI-compatible)", base_url, api_key, "")
        # Try listing models as a lightweight test
        try:
            client.models.list()
            return _status_html(f"✅ Connected to {base_url} — model list OK. Ready to run.")
        except Exception:
            # Fall back: minimal 1-token completion
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
    Ping the ngrok URL (GET /v1/models) and verify a 200 response.

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
            return _status_html(f"✅ ngrok server reachable at {ngrok_url} — models endpoint OK.")
        else:
            return _status_html(
                f"⚠️ Server responded HTTP {resp.status_code}. It may still work — proceed with caution.",
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
# SECTION 2: INFERENCE HELPERS (mirrors inference.py logic)
# ===========================================================================

def _build_user_prompt(
    step: int,
    task_prompt: str,
    available_tools: List[Dict[str, Any]],
    last_reward: float,
    history: List[str],
) -> str:
    """Build the per-step user prompt (mirrors inference.py build_user_prompt)."""
    from models import ToolForgeAction, ToolCall, Tool  # type: ignore
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current task: {task_prompt!r}
        Available tools: {json.dumps(available_tools)}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}

        Return a ToolForgeAction whose `plan` uses only currently available tools.
        """
    ).strip()


def _get_model_action(
    client,
    model_name: str,
    step: int,
    task_prompt: str,
    available_tools: List[Dict[str, Any]],
    last_reward: float,
    history: List[str],
    system_prompt: str,
) -> Any:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _get_model_action
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Call the LLM and parse the response into a ToolForgeAction.
    Returns (action, raw_text, error_str).
    On parse failure, returns a single-tool fallback action.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    from models import ToolForgeAction, ToolCall, Tool  # type: ignore

    user_prompt = _build_user_prompt(step, task_prompt, available_tools, last_reward, history)
    system = (
        system_prompt
        + "Respond ONLY with valid JSON matching the ToolForgeAction schema.\n"
        + "No markdown, no explanation, no commentary outside the JSON object."
    )
    schema_suffix = """

Expected JSON Structure:
{
  "action_type": "propose_plan", // or "propose_plan_with_macro"
  "plan": [
    { "tool_name": "deploy" },
    { "tool_name": "healthcheck" }
  ],
  "macro_proposal": null // or an object if action_type is propose_plan_with_macro
}

Example of proposing a macro:
{
  "action_type": "propose_plan_with_macro",
  "plan": [
    { "tool_name": "deploy" }
  ],
  "macro_proposal": {
    "name": "deploy_and_verify",
    "description": "Deploys code and checks health",
    "steps": [
      { "tool_name": "deploy" },
      { "tool_name": "healthcheck" }
    ]
  }
}
"""
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt + schema_suffix},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        action = ToolForgeAction(**json.loads(raw))
        return action, raw, None
    except Exception as exc:
        fallback_tool = available_tools[0]["name"] if available_tools else "deploy"
        action = ToolForgeAction(
            action_type="propose_plan",
            plan=[ToolCall(tool_name=fallback_tool)],
            macro_proposal=None,
        )
        return action, None, str(exc)


# ===========================================================================
# SECTION 3: EPISODE TABLE RENDERING
# ===========================================================================

def _render_episode_table(rows: List[Dict[str, Any]]) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _render_episode_table
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Render the episode-by-episode results table as HTML.

    Each row dict has keys:
        episode     int
        task        str   — truncated task prompt
        plan        str   — "tool1, tool2, ..." (comma-joined)
        reward      float
        turns       int
        macro       str   — macro name or "—"

    Returns HTML table string.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not rows:
        return (
            '<div style="padding:16px;color:#9ca3af;font-style:italic;text-align:center;">'
            "No episodes completed yet."
            "</div>"
        )

    header = (
        "<table style='width:100%;border-collapse:collapse;font-size:0.83em;'>"
        "<thead><tr style='background:#1f2937;color:#9ca3af;text-align:left;'>"
        "<th style='padding:6px 10px;'>Ep</th>"
        "<th style='padding:6px 10px;'>Task</th>"
        "<th style='padding:6px 10px;'>Plan</th>"
        "<th style='padding:6px 10px;'>Reward</th>"
        "<th style='padding:6px 10px;'>Turns</th>"
        "<th style='padding:6px 10px;'>Macro used</th>"
        "</tr></thead><tbody>"
    )

    body = ""
    total_reward = 0.0
    total_turns = 0
    for i, row in enumerate(rows):
        bg = "#111827" if i % 2 == 0 else "#1a2332"
        reward = row.get("reward", 0.0)
        total_reward += reward
        total_turns += row.get("turns", 0)
        reward_colour = "#22c55e" if reward >= 0.75 else ("#f59e0b" if reward >= 0.5 else ("#f97316" if reward >= 0 else "#ef4444"))
        task_text = str(row.get("task", ""))[:60] + ("…" if len(str(row.get("task", ""))) > 60 else "")
        body += (
            f"<tr style='background:{bg};color:#d1d5db;'>"
            f"<td style='padding:6px 10px;font-weight:600;'>{row.get('episode', i+1)}</td>"
            f"<td style='padding:6px 10px;'>{task_text}</td>"
            f"<td style='padding:6px 10px;font-family:monospace;color:#a5b4fc;'>{row.get('plan','')}</td>"
            f"<td style='padding:6px 10px;font-weight:700;color:{reward_colour};'>{reward:+.2f}</td>"
            f"<td style='padding:6px 10px;'>{row.get('turns', 0)}</td>"
            f"<td style='padding:6px 10px;color:#c4b5fd;'>{row.get('macro', '—')}</td>"
            f"</tr>"
        )

    n = len(rows)
    avg_reward = total_reward / n if n else 0.0
    footer = (
        f"<tr style='background:#374151;color:#e5e7eb;font-weight:700;border-top:2px solid #4b5563;'>"
        f"<td style='padding:6px 10px;' colspan='3'>Totals ({n} ep)</td>"
        f"<td style='padding:6px 10px;'>avg {avg_reward:+.2f}</td>"
        f"<td style='padding:6px 10px;'>{total_turns}</td>"
        f"<td style='padding:6px 10px;'></td>"
        f"</tr>"
    )
    return header + body + footer + "</tbody></table>"


# ===========================================================================
# SECTION 4: MAIN INFERENCE LOOP (generator — yields UI updates per step)
# ===========================================================================

def run_agent_episode(
    mode: str,
    base_url: str,
    model_name: str,
    api_key: str,
    ngrok_url: str,
    env_url: str,
    system_prompt: str,
    # State inputs
    episode_rows_state: List[Dict[str, Any]],
    training_data_state: List[Dict[str, Any]],
    episode_number_state: int,
) -> Generator:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    run_agent_episode
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Generator that runs one full episode and yields UI updates after each step.

    Yields a tuple matching the outputs list defined in build_byoa_tab():
        (status_html, table_html, training_json, macro_html,
         episode_rows_state, training_data_state, episode_number_state)

    Args:
        mode                 : Connection mode radio value
        base_url             : API base URL (API key mode)
        model_name           : LLM model identifier (both modes)
        api_key              : API key (API key mode) — NEVER logged
        ngrok_url            : ngrok tunnel URL (ngrok mode)
        env_url              : ToolForge env server URL
        system_prompt        : The editable system prompt provided by the user
        episode_rows_state   : Accumulated episode row dicts (across episodes)
        training_data_state  : Accumulated training data records
        episode_number_state : Current episode number (1-based)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # --- Input validation --------------------------------------------------
    if not model_name.strip():
        yield (
            _status_html("❌ Model name is required.", error=True),
            _render_episode_table(episode_rows_state),
            _json_block(training_data_state),
            render_macro_library_html([]),
            episode_rows_state, training_data_state, episode_number_state,
            gr.update(interactive=True)
        )
        return

    effective_base = ngrok_url if mode == "Local Model via ngrok" else base_url
    if not effective_base.strip():
        yield (
            _status_html("❌ URL is required.", error=True),
            _render_episode_table(episode_rows_state),
            _json_block(training_data_state),
            render_macro_library_html([]),
            episode_rows_state, training_data_state, episode_number_state,
            gr.update(interactive=True)
        )
        return

    # --- Build client -------------------------------------------------------
    try:
        client = _make_openai_client(mode, base_url, api_key, ngrok_url)
    except Exception as exc:
        yield (
            _status_html(f"❌ Could not create LLM client: {exc}", error=True),
            _render_episode_table(episode_rows_state),
            _json_block(training_data_state),
            render_macro_library_html([]),
            episode_rows_state, training_data_state, episode_number_state,
            gr.update(interactive=True)
        )
        return

    ep_num = episode_number_state
    yield (
        _status_html(f"⏳ Starting episode {ep_num} — resetting environment…", warn=True),
        _render_episode_table(episode_rows_state),
        _json_block(training_data_state),
        render_macro_library_html([]),
        episode_rows_state, training_data_state, episode_number_state,
        gr.update(interactive=False)
    )

    # --- Reset environment --------------------------------------------------
    result, err = env_reset(env_url, TASK_ID)
    if err or result is None:
        yield (
            _status_html(f"❌ env_reset failed: {err}", error=True),
            _render_episode_table(episode_rows_state),
            _json_block(training_data_state),
            render_macro_library_html([]),
            episode_rows_state, training_data_state, episode_number_state,
            gr.update(interactive=True)
        )
        return

    total_tasks = parse_total_tasks_from_obs(result)
    history: List[str] = []
    last_reward = 0.0
    macros: List[Dict[str, Any]] = []
    episode_reward_sum = 0.0
    episode_turns = 0
    macro_created_name: Optional[str] = None
    done = False
    task_step_idx = 0

    # --- Step loop ----------------------------------------------------------
    for step in range(1, MAX_STEPS_PER_EPISODE + 1):
        if done:
            break

        task_info  = parse_task_from_obs(result)
        tools_list = parse_tools_from_obs(result)
        macros     = extract_macros(tools_list)

        task_prompt   = task_info.get("prompt", "")
        task_id_str   = task_info.get("id", "unknown")
        tools_for_llm = [{"name": t.get("name", ""), "description": t.get("description", "")} for t in tools_list]

        yield (
            _status_html(
                f"⏳ Episode {ep_num} | Task {task_step_idx+1}/{total_tasks or '?'} | "
                f"Step {step} — asking {model_name}…",
                warn=True,
            ),
            _render_episode_table(episode_rows_state),
            _json_block(training_data_state),
            render_macro_library_html(macros),
            episode_rows_state, training_data_state, episode_number_state,
            gr.update(interactive=False)
        )

        # LLM call
        action, raw_text, llm_err = _get_model_action(
            client, model_name, step,
            task_prompt, tools_for_llm, last_reward, history, system_prompt
        )

        if llm_err and raw_text is None:
            # Hard LLM failure — stop this episode
            yield (
                _status_html(f"❌ LLM call failed: {llm_err}", error=True),
                _render_episode_table(episode_rows_state),
                _json_block(training_data_state),
                render_macro_library_html(macros),
                episode_rows_state, training_data_state, episode_number_state,
                gr.update(interactive=True)
            )
            return

        plan_names = [tc.tool_name for tc in action.plan]
        mp = action.macro_proposal
        if mp and mp.name:
            macro_created_name = mp.name

        # Build macro_proposal in the format env_step expects: {"name": str, "steps": List[str]}
        mp_dict: Optional[Dict[str, Any]] = None
        if mp and mp.name and mp.steps:
            mp_dict = {
                "name":  mp.name,
                "steps": [tc.tool_name for tc in mp.steps],
            }

        # env_step
        step_result, step_err = env_step(env_url, plan_names, macro_proposal=mp_dict)
        if step_err or step_result is None:
            yield (
                _status_html(f"❌ env_step failed: {step_err}", error=True),
                _render_episode_table(episode_rows_state),
                _json_block(training_data_state),
                render_macro_library_html(macros),
                episode_rows_state, training_data_state, episode_number_state,
                gr.update(interactive=True)
            )
            return

        reward = float(step_result.get("reward", 0.0))
        done   = bool(step_result.get("done", False))
        meta   = step_result.get("observation", {}).get("metadata", {}) or {}

        episode_reward_sum += reward
        episode_turns      += 1
        last_reward         = reward
        task_step_idx      += 1

        history.append(
            f"{action.model_dump_json()}|Step {step}|reward {reward:+.2f}|task_id {task_id_str}"
        )

        # Build training record for this step
        training_record = {
            "episode":           ep_num,
            "task":              task_prompt,
            "plan":              plan_names,
            "macro_proposed":    mp.model_dump() if mp else None,
            "reward":            reward,
            "turns_used":        step,
            "turns_saved":       max(0, task_info.get("baseline_call_count", len(plan_names)) - len(plan_names)),
            "slots_filled":      [],    # populated from metadata when available
            "slots_missing":     [],
            "validation_passed": bool(meta.get("task_complete", reward > 0)),
        }
        training_data_state = list(training_data_state) + [training_record]

        # Update the result for next iteration
        result = step_result

        # Check UI-side episode completion
        ui_ep_done = done or (total_tasks > 0 and task_step_idx >= total_tasks)
        if ui_ep_done:
            break

    # --- Episode complete — add summary row ---------------------------------
    row: Dict[str, Any] = {
        "episode": ep_num,
        "task":    task_prompt,
        "plan":    ", ".join(plan_names) if plan_names else "—",
        "reward":  episode_reward_sum / max(episode_turns, 1),
        "turns":   episode_turns,
        "macro":   macro_created_name or "—",
    }
    new_rows = list(episode_rows_state) + [row]
    new_ep_num = ep_num + 1

    yield (
        _status_html(
            f"✅ Episode {ep_num} complete — {episode_turns} turns, "
            f"avg reward {episode_reward_sum/max(episode_turns,1):+.2f}. "
            f"Click 'Next Episode' to continue."
        ),
        _render_episode_table(new_rows),
        _json_block(training_data_state),
        render_macro_library_html(macros),
        new_rows, training_data_state, new_ep_num,
        gr.update(interactive=True)
    )


def _json_block(data: List[Dict[str, Any]]) -> str:
    """Render training data list as a preformatted JSON block."""
    if not data:
        return '<pre style="color:#9ca3af;font-size:0.8em;">No training data yet.</pre>'
    return (
        '<pre style="background:#111827;color:#a5b4fc;padding:12px;border-radius:8px;'
        'font-size:0.78em;overflow:auto;max-height:300px;">'
        + json.dumps(data[-10:], indent=2)   # show last 10 records
        + f'\n... ({len(data)} total records)</pre>'
    )


# ===========================================================================
# SECTION 5: DOWNLOAD HANDLER
# ===========================================================================

def on_download_training_data(training_data_state: List[Dict[str, Any]]):
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_download_training_data
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Write training data to a real temp file and return it for Gradio download.

    Args:
        training_data_state : List of training record dicts.

    Returns:
        File path string (Gradio gr.File picks this up).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    if not training_data_state:
        return None

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="toolforge_training_",
        delete=False,
        encoding="utf-8",
    )
    json.dump(training_data_state, tmp, indent=2)
    tmp.close()
    logger.info("Training data written to %s (%d records)", tmp.name, len(training_data_state))
    return tmp.name


# ===========================================================================
# SECTION 6: TAB BUILDER
# ===========================================================================

def build_byoa_tab() -> gr.Tab:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    build_byoa_tab
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Construct and return the "Bring Your Own Agent" gr.Tab component.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    with gr.Tab("🔌 Bring Your Own Agent") as tab:

        # -------------------------------------------------------------------
        # HEADER
        # -------------------------------------------------------------------
        gr.Markdown(
            """
            # Bring Your Own Agent
            Connect your own LLM and run it against the **real ToolForge environment**.
            Your agent receives actual task prompts, proposes plans, earns rewards,
            and is scored by the same pipeline as the benchmark.
            """
        )
        gr.HTML(_DIVIDER)

        # -------------------------------------------------------------------
        # CONNECTION MODE
        # -------------------------------------------------------------------
        connection_mode = gr.Radio(
            choices=["API Key (OpenAI-compatible)", "Local Model via ngrok"],
            value="API Key (OpenAI-compatible)",
            label="Connection Mode",
        )

        # --- API KEY SECTION ------------------------------------------------
        with gr.Group(visible=True) as api_key_section:
            gr.HTML(_sec_hdr("API Key Connection"))
            with gr.Row():
                base_url_field = gr.Textbox(
                    value="https://api.openai.com/v1",
                    label="Base URL",
                    placeholder="https://api.openai.com/v1",
                    scale=3,
                )
                api_model_field = gr.Textbox(
                    value="",
                    label="Model Name",
                    placeholder="e.g. gpt-4o, mistral-small, claude-3-5-sonnet",
                    scale=2,
                )
            api_key_field = gr.Textbox(
                value="",
                label="API Key  (used only in your session — never stored or logged)",
                placeholder="sk-...",
                type="password",
            )
            with gr.Row():
                test_api_btn   = gr.Button("Test Connection", variant="secondary", scale=1)
                api_status_html = gr.HTML(value="", scale=3)

        # --- NGROK SECTION --------------------------------------------------
        with gr.Group(visible=False) as ngrok_section:
            gr.HTML(_sec_hdr("Local Model via ngrok"))
            with gr.Row():
                ngrok_url_field   = gr.Textbox(
                    value="",
                    label="ngrok Tunnel URL",
                    placeholder="https://xxxx.ngrok-free.app/v1",
                    scale=3,
                )
                ngrok_model_field = gr.Textbox(
                    value="",
                    label="Model Name",
                    placeholder="e.g. llama3, mistral, phi3",
                    scale=2,
                )
            with gr.Row():
                test_ngrok_btn   = gr.Button("Test Connection", variant="secondary", scale=1)
                ngrok_status_html = gr.HTML(value="", scale=3)

            # ngrok setup instructions accordion
            with gr.Accordion("📖 Setup Instructions — Local Model via ngrok", open=False):
                gr.Markdown(
                    """
                    **Step 1 — Install and run a local model server**

                    Option A — **Ollama** (easiest):
                    ```bash
                    # macOS / Linux
                    curl -fsSL https://ollama.com/install.sh | sh
                    ollama pull llama3
                    ollama serve          # starts on http://localhost:11434
                    ```

                    Option B — **LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai),
                    load a model, and enable the local server (port 1234 by default).

                    ---

                    **Step 2 — Download and run the ToolForge proxy server**

                    Download `local_agent_server.py` (button below), then:
                    ```bash
                    pip install fastapi uvicorn httpx
                    # For Ollama:
                    python local_agent_server.py --model llama3 --port 8080 --backend-url http://localhost:11434
                    # For LM Studio:
                    python local_agent_server.py --model your-model --port 8080 --backend-url http://localhost:1234
                    ```

                    ---

                    **Step 3 — Expose via ngrok**
                    ```bash
                    # Install ngrok: https://ngrok.com/download
                    ngrok config add-authtoken YOUR_TOKEN   # one-time setup
                    ngrok http 8080
                    ```
                    Copy the **Forwarding** URL (e.g. `https://abcd1234.ngrok-free.app`)
                    and paste it into the **ngrok Tunnel URL** field above (include `/v1` suffix).

                    ---

                    **Step 4** — Click **Test Connection** to verify, then **Run Agent**.
                    """
                )
                download_local_server_btn = gr.Button(
                    "⬇ Download local_agent_server.py", variant="secondary"
                )
                local_server_file = gr.File(label="", visible=False)

        gr.HTML(_DIVIDER)

        # -------------------------------------------------------------------
        # ENV URL + SYSTEM PROMPT
        # -------------------------------------------------------------------
        gr.HTML(_sec_hdr("Settings"))
        env_url_field = gr.Textbox(
            value=DEFAULT_ENV_URL,
            label="ToolForge Env URL",
            placeholder="http://localhost:8000",
        )
        system_prompt_box = gr.Textbox(
            value=SYSTEM_PROMPT,
            label="Agent System Prompt  (editable)",
            lines=10,
        )

        gr.HTML(_DIVIDER)

        # -------------------------------------------------------------------
        # RUN CONTROLS ROW
        # -------------------------------------------------------------------
        with gr.Row():
            run_btn        = gr.Button("Run Agent",      variant="primary",   scale=2)
            next_ep_btn    = gr.Button("Next Episode",   variant="secondary", scale=2, interactive=False)
            pause_btn      = gr.Button("Pause",          variant="secondary", scale=1)
            autoplay_btn   = gr.Button("Autoplay: OFF",  variant="secondary", scale=1)

        run_status_html = gr.HTML(value="")

        gr.HTML(_DIVIDER)

        # -------------------------------------------------------------------
        # RESULTS AREA
        # -------------------------------------------------------------------
        gr.HTML(_sec_hdr("Episode Results"))
        episode_table_html = gr.HTML(value=_render_episode_table([]))

        gr.HTML(_DIVIDER)

        gr.HTML(_sec_hdr("Macro Library"))
        byoa_macro_html = gr.HTML(value=render_macro_library_html([]))

        gr.HTML(_DIVIDER)

        # -------------------------------------------------------------------
        # TRAINING DATA
        # -------------------------------------------------------------------
        gr.HTML(_sec_hdr("Training Data Export"))
        gr.Markdown(
            "Live JSON log of every step — ready for fine-tuning or analysis. "
            "Shows the last 10 records. Download to get all records."
        )
        training_json_html = gr.HTML(value=_json_block([]))
        with gr.Row():
            download_training_btn  = gr.Button("Download Training Data", variant="secondary", scale=1)
            training_download_file = gr.File(label="", visible=False, scale=2)

        # -------------------------------------------------------------------
        # HIDDEN STATE
        # -------------------------------------------------------------------
        episode_rows_state   = gr.State(value=[])   # List[Dict] — episode summary rows
        training_data_state  = gr.State(value=[])   # List[Dict] — training records
        episode_number_state = gr.State(value=1)    # int — current episode number (1-based)
        autoplay_state       = gr.State(value=False) # bool — autoplay active

        # -------------------------------------------------------------------
        # EVENT WIRING
        # -------------------------------------------------------------------

        # Connection mode toggle
        connection_mode.change(
            fn=on_mode_change,
            inputs=[connection_mode],
            outputs=[api_key_section, ngrok_section],
        )

        # API key test
        test_api_btn.click(
            fn=on_test_connection_api,
            inputs=[base_url_field, api_model_field, api_key_field],
            outputs=[api_status_html],
        )

        # ngrok test
        test_ngrok_btn.click(
            fn=on_test_connection_ngrok,
            inputs=[ngrok_url_field, ngrok_model_field],
            outputs=[ngrok_status_html],
        )

        # Unified model name: API mode uses api_model_field, ngrok uses ngrok_model_field.
        # The run handler reads both and picks the relevant one based on mode.

        def _get_model_name(mode: str, api_model: str, ngrok_model: str) -> str:
            """Return the active model name based on connection mode."""
            if mode == "Local Model via ngrok":
                return ngrok_model
            return api_model

        # Run agent (generator → streams updates)
        def run_agent_wrapper(
            mode, base_url, api_model, api_key, ngrok_url, ngrok_model,
            env_url, system_prompt, episode_rows, training_data, ep_num,
        ):
            model = _get_model_name(mode, api_model, ngrok_model)
            yield from run_agent_episode(
                mode=mode,
                base_url=base_url,
                model_name=model,
                api_key=api_key,
                ngrok_url=ngrok_url,
                env_url=env_url,
                system_prompt=system_prompt,
                episode_rows_state=episode_rows,
                training_data_state=training_data,
                episode_number_state=ep_num,
            )

        _run_outputs = [
            run_status_html,
            episode_table_html,
            training_json_html,
            byoa_macro_html,
            episode_rows_state,
            training_data_state,
            episode_number_state,
            system_prompt_box,
        ]

        run_btn.click(
            fn=run_agent_wrapper,
            inputs=[
                connection_mode,
                base_url_field,
                api_model_field,
                api_key_field,
                ngrok_url_field,
                ngrok_model_field,
                env_url_field,
                system_prompt_box,
                episode_rows_state,
                training_data_state,
                episode_number_state,
            ],
            outputs=_run_outputs,
        )

        # Next Episode — same as run but episode_number carries over from state
        next_ep_btn.click(
            fn=run_agent_wrapper,
            inputs=[
                connection_mode,
                base_url_field,
                api_model_field,
                api_key_field,
                ngrok_url_field,
                ngrok_model_field,
                env_url_field,
                system_prompt_box,
                episode_rows_state,
                training_data_state,
                episode_number_state,
            ],
            outputs=_run_outputs,
        )

        # Enable Next Episode button after first run completes
        run_btn.click(
            fn=lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[next_ep_btn],
        )

        # Autoplay toggle (visual only — actual loop left to user clicking)
        def toggle_autoplay(active: bool):
            new_active = not active
            label = "Autoplay: ON" if new_active else "Autoplay: OFF"
            return gr.update(value=label), new_active

        autoplay_btn.click(
            fn=toggle_autoplay,
            inputs=[autoplay_state],
            outputs=[autoplay_btn, autoplay_state],
        )

        # Download training data
        def _do_download(data):
            path = on_download_training_data(data)
            if path:
                return gr.update(value=path, visible=True)
            return gr.update(visible=False)

        download_training_btn.click(
            fn=_do_download,
            inputs=[training_data_state],
            outputs=[training_download_file],
        )

        # Download local_agent_server.py
        def _serve_local_agent_server():
            """Return the path to local_agent_server.py for download."""
            candidate = os.path.join(
                os.path.dirname(__file__), "..", "ui", "local_agent_server.py"
            )
            alt = os.path.join(os.path.dirname(__file__), "local_agent_server.py")
            path = candidate if os.path.exists(candidate) else (alt if os.path.exists(alt) else None)
            if path:
                return gr.update(value=path, visible=True)
            return gr.update(visible=False)

        download_local_server_btn.click(
            fn=_serve_local_agent_server,
            inputs=[],
            outputs=[local_server_file],
        )

    return tab
