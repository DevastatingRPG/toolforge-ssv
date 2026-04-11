# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tab 2 — Bring Your Own Agent (BYOA)
====================================

Lets users connect their own LLM and run it against the real ToolForge
environment.  Supports two connection modes:

    1. API Key (OpenAI-compatible)  — user supplies a base URL + API key.
    2. Local Model via ngrok        — user supplies an ngrok tunnel URL.

The agent system prompt is pre-filled from shared.SYSTEM_PROMPT but is
fully editable.  A warning is shown that the JSON response format must
not be changed (it would break the server-side parser).

Results layout mirrors the Demo Mode right column:
    - Agent Plan HTML
    - Reward display
    - LLM Turns Used / Turns Saved
    - Macro Library
    - Download Results JSON button (placeholder)

TODO: Wire the Run button to the real ToolforgeEnv client (inference.py
      logic).  The current handler returns placeholder results.
TODO: Implement the "Download Results JSON" button to serialise the run
      log and trigger a Gradio file download.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from toolforge_env.ui.shared import (
    ATOMIC_TOOLS,
    SYSTEM_PROMPT,
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
# Difficulty options (all three levels are available in BYOA mode)
# ---------------------------------------------------------------------------
DIFFICULTY_OPTIONS: List[str] = ["Easy", "Medium", "Hard"]

# Provider presets — defines the default base URL per provider
PROVIDER_PRESETS: Dict[str, str] = {
    "OpenAI":    "https://api.openai.com/v1",
    "Anthropic": "https://api.anthropic.com/v1",
    "Other":     "",
}

# Default model name placeholder shown in the model field
DEFAULT_MODEL_NAME: str = "gpt-4o"

# Placeholder ngrok URL hint
NGROK_URL_PLACEHOLDER: str = "https://xxxx-xx-xx-xxx.ngrok-free.app/v1"


# ===========================================================================
# EVENT HANDLERS
# ===========================================================================

def on_provider_change(provider: str) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_provider_change
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Pre-fill the Base URL field when the user selects a known provider.

    Args:
        provider : Selected provider name ("OpenAI", "Anthropic", "Other").

    Returns:
        The preset base URL string for that provider.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    return PROVIDER_PRESETS.get(provider, "")


def on_mode_change(mode: str) -> Tuple[bool, bool]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_mode_change
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Show or hide the API key section vs the ngrok section depending on
    which connection mode the user selected.

    Args:
        mode : "API Key (OpenAI-compatible)" or "Local Model via ngrok".

    Returns:
        Tuple (api_key_visible, ngrok_visible) as gr.update dicts.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    is_api_mode   = (mode == "API Key (OpenAI-compatible)")
    is_ngrok_mode = not is_api_mode
    return gr.update(visible=is_api_mode), gr.update(visible=is_ngrok_mode)


def on_test_connection_api(
    provider: str,
    base_url: str,
    model_name: str,
    api_key: str,
) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_test_connection_api
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Placeholder connection test for API key mode.

    TODO: Perform a real lightweight LLM call (e.g., list models or a
          minimal completion) to verify credentials before the full run.

    Args:
        provider   : Selected provider name.
        base_url   : API base URL.
        model_name : Model identifier string.
        api_key    : User's API key (not stored or logged).

    Returns:
        Status message HTML string.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    logger.info("Connection test requested | provider=%s model=%s", provider, model_name)

    # TODO: Replace with a real openai.OpenAI(base_url=base_url, api_key=api_key)
    #       test call.  For now we return a success placeholder.
    if not api_key.strip():
        return (
            '<div style="color:#f87171;padding:8px;border-radius:6px;background:#450a0a;">'
            '❌ API key is empty. Please enter your key.</div>'
        )
    if not base_url.strip():
        return (
            '<div style="color:#f87171;padding:8px;border-radius:6px;background:#450a0a;">'
            '❌ Base URL is empty. Please enter the API endpoint.</div>'
        )

    return (
        '<div style="color:#4ade80;padding:8px;border-radius:6px;background:#052e16;">'
        f'✅ Connection to <strong>{provider}</strong> ({model_name}) looks valid. '
        'Ready to run.</div>'
    )


def on_test_connection_ngrok(ngrok_url: str) -> str:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_test_connection_ngrok
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Placeholder connection test for the ngrok / local model mode.

    TODO: Actually ping the ngrok URL (e.g., GET /v1/models) and verify
          a 200 response before allowing the user to start a run.

    Args:
        ngrok_url : The ngrok tunnel URL for the local model server.

    Returns:
        Status message HTML string.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    logger.info("ngrok connection test requested | url=%s", ngrok_url)

    if not ngrok_url.strip():
        return (
            '<div style="color:#f87171;padding:8px;border-radius:6px;background:#450a0a;">'
            '❌ ngrok URL is empty.</div>'
        )

    return (
        '<div style="color:#4ade80;padding:8px;border-radius:6px;background:#052e16;">'
        f'✅ Placeholder success — ngrok URL accepted. '
        'TODO: Add real connectivity ping.</div>'
    )


def on_run_byoa(
    mode: str,
    # API key mode fields
    provider: str,
    base_url: str,
    model_name: str,
    api_key: str,
    # ngrok mode fields
    ngrok_url: str,
    # Common fields
    difficulty: str,
    system_prompt: str,
) -> Tuple:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_run_byoa
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Entry point for running the user's agent against the ToolForge env.

    TODO: This function should:
        1. Instantiate an openai.OpenAI client from the supplied credentials.
        2. Connect to the ToolForge environment server (local or HF Space).
        3. Run the inference loop (mirroring inference.py) for the chosen
           difficulty task group.
        4. Stream intermediate results back to the UI via a generator
           (use `yield` for streaming updates).
        5. Call the EpisodeGrader at the end to compute the final score.

    For now, returns static placeholder data after a fake "run" notice.

    Args:
        mode         : Connection mode string.
        provider     : LLM provider (API key mode).
        base_url     : API endpoint (API key mode).
        model_name   : Model identifier (API key mode).
        api_key      : User's API key (API key mode) — NEVER log this.
        ngrok_url    : ngrok tunnel URL (local mode).
        difficulty   : Difficulty level string.
        system_prompt: Editable system prompt for the agent.

    Returns:
        Tuple of display values for the results area + a run log JSON string.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    logger.info(
        "BYOA run requested | mode=%s difficulty=%s model=%s",
        mode, difficulty, model_name if mode == "API Key (OpenAI-compatible)" else "local",
    )

    # ------------------------------------------------------------------
    # TODO: Replace everything below with a real inference run.
    # The placeholder simulates one successful step so the UI is visually
    # complete even without a live backend.
    # ------------------------------------------------------------------

    # Placeholder plan
    placeholder_plan     = ["deploy", "healthcheck", "notify"]
    placeholder_macros   = []                # No macros created in placeholder run
    placeholder_reward   = 0.65
    placeholder_baseline = 3

    plan_html    = render_plan_html(placeholder_plan, placeholder_macros)
    reward_html  = render_reward_html(placeholder_reward)
    turns_used   = len(placeholder_plan)
    turns_saved  = max(0, placeholder_baseline - turns_used)
    macro_html   = render_macro_library_html(placeholder_macros)

    # Placeholder run log for the download button
    run_log = {
        "status":     "placeholder",
        "difficulty": difficulty,
        "note":       "TODO: Replace with real inference run output.",
        "plan":       placeholder_plan,
        "reward":     placeholder_reward,
        "macros":     placeholder_macros,
    }
    run_log_str = json.dumps(run_log, indent=2)

    return (
        gr.update(visible=True),   # Make results area visible
        plan_html,
        reward_html,
        turns_used,
        turns_saved,
        macro_html,
        run_log_str,               # Raw JSON for download button
    )


def on_download_results(run_log_json: str):
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    on_download_results
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Placeholder for the Download Results JSON button.

    TODO: Write run_log_json to a temp file and return it as a
          gr.File component so Gradio triggers a browser download.

    Args:
        run_log_json : JSON string of the run log.

    Returns:
        Status message string (placeholder).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # TODO: implement file download
    return (
        '<div style="color:#fbbf24;padding:8px;border-radius:6px;background:#451a03;">'
        '⚠️ Download not yet implemented. TODO: Write to temp file and return gr.File.</div>'
    )


# ===========================================================================
# TAB BUILDER
# ===========================================================================

def build_byoa_tab() -> gr.Tab:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    build_byoa_tab
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Construct and return the "Bring Your Own Agent" gr.Tab component.

    Returns:
        A configured gr.Tab component.
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
            Your agent will receive actual task prompts, propose plans, earn rewards,
            and can create macro tools — all scored by the same pipeline as the benchmark.
            """
        )

        gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

        # -------------------------------------------------------------------
        # CONNECTION MODE RADIO
        # -------------------------------------------------------------------
        connection_mode = gr.Radio(
            choices=["API Key (OpenAI-compatible)", "Local Model via ngrok"],
            value="API Key (OpenAI-compatible)",
            label="Connection Mode",
        )

        # -------------------------------------------------------------------
        # API KEY SECTION (visible by default)
        # -------------------------------------------------------------------
        with gr.Group(visible=True) as api_key_section:
            gr.Markdown("### API Key Connection")

            with gr.Row():
                provider_dropdown = gr.Dropdown(
                    choices=list(PROVIDER_PRESETS.keys()),
                    value="OpenAI",
                    label="Provider",
                    scale=1,
                )
                base_url_field = gr.Textbox(
                    value=PROVIDER_PRESETS["OpenAI"],   # pre-filled for OpenAI
                    label="Base URL",
                    placeholder="https://api.openai.com/v1",
                    scale=3,
                )

            with gr.Row():
                model_name_field = gr.Textbox(
                    value=DEFAULT_MODEL_NAME,
                    label="Model Name",
                    placeholder="gpt-4o",
                    scale=2,
                )
                api_key_field = gr.Textbox(
                    value="",
                    label="API Key  (used only in your session — never stored)",
                    placeholder="sk-...",
                    type="password",        # masks the key in the UI
                    scale=3,
                )

            test_api_btn = gr.Button("Test Connection", variant="secondary")
            api_status_html = gr.HTML(value="")

        # -------------------------------------------------------------------
        # NGROK SECTION (hidden by default)
        # -------------------------------------------------------------------
        with gr.Group(visible=False) as ngrok_section:
            gr.Markdown("### Local Model via ngrok")

            ngrok_url_field = gr.Textbox(
                value="",
                label="ngrok Tunnel URL",
                placeholder=NGROK_URL_PLACEHOLDER,
            )

            test_ngrok_btn  = gr.Button("Test Connection", variant="secondary")
            ngrok_status_html = gr.HTML(value="")

            # Collapsible setup instructions
            with gr.Accordion("📖 Setup Instructions", open=False):
                gr.Markdown(
                    """
                    TODO: Add step-by-step instructions for:
                    1. Installing and authenticating ngrok.
                    2. Running a local OpenAI-compatible server (e.g. LM Studio, Ollama).
                    3. Exposing the local server via `ngrok http 11434`.
                    4. Pasting the resulting tunnel URL here.
                    """
                )

        gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

        # -------------------------------------------------------------------
        # COMMON SETTINGS (shown for both connection modes)
        # -------------------------------------------------------------------
        gr.Markdown("### Run Settings")

        with gr.Row():
            difficulty_selector = gr.Dropdown(
                choices=DIFFICULTY_OPTIONS,
                value="Easy",
                label="Difficulty",
                scale=1,
            )

        # Editable system prompt — pre-filled from inference.py
        system_prompt_box = gr.Textbox(
            value=SYSTEM_PROMPT,
            label="Agent System Prompt",
            lines=14,
            info=(
                "You may edit the strategy section above the ⚠️ warning line. "
                "Do NOT remove the JSON-only response requirement — it will break the parser."
            ),
        )

        run_btn = gr.Button("▶ Run Agent", variant="primary")

        gr.HTML("<hr style='border-color:#374151;margin:8px 0;'>")

        # -------------------------------------------------------------------
        # RESULTS AREA (hidden until a run completes)
        # -------------------------------------------------------------------
        with gr.Group(visible=False) as results_group:
            gr.Markdown("### Results")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**🤖 Agent Plan**")
                    byoa_plan_html = gr.HTML(
                        value="<p style='color:#9ca3af;'>No plan yet.</p>"
                    )

                    gr.Markdown("**⚙ Macro Library**")
                    byoa_macro_html = gr.HTML(value=render_macro_library_html([]))

                with gr.Column(scale=1):
                    gr.Markdown("**🏆 Step Reward**")
                    byoa_reward_html = gr.HTML(value=render_reward_html(0.0))

                    with gr.Row():
                        byoa_turns_used = gr.Number(
                            label="LLM Turns Used",
                            value=0,
                            interactive=False,
                            precision=0,
                        )
                        byoa_turns_saved = gr.Number(
                            label="Turns Saved vs Baseline",
                            value=0,
                            interactive=False,
                            precision=0,
                        )

            # Run log storage (hidden) and download button
            byoa_run_log_state = gr.State(value="")

            with gr.Row():
                download_btn    = gr.Button("⬇ Download Results JSON", scale=1)
                download_status = gr.HTML(value="", scale=2)

        # -------------------------------------------------------------------
        # EVENT WIRING
        # -------------------------------------------------------------------

        # Connection mode toggle
        connection_mode.change(
            fn=on_mode_change,
            inputs=[connection_mode],
            outputs=[api_key_section, ngrok_section],
        )

        # Provider changes → pre-fill base URL
        provider_dropdown.change(
            fn=on_provider_change,
            inputs=[provider_dropdown],
            outputs=[base_url_field],
        )

        # API key test button
        test_api_btn.click(
            fn=on_test_connection_api,
            inputs=[provider_dropdown, base_url_field, model_name_field, api_key_field],
            outputs=[api_status_html],
        )

        # ngrok test button
        test_ngrok_btn.click(
            fn=on_test_connection_ngrok,
            inputs=[ngrok_url_field],
            outputs=[ngrok_status_html],
        )

        # Run agent button
        run_btn.click(
            fn=on_run_byoa,
            inputs=[
                connection_mode,
                provider_dropdown,
                base_url_field,
                model_name_field,
                api_key_field,
                ngrok_url_field,
                difficulty_selector,
                system_prompt_box,
            ],
            outputs=[
                results_group,      # make visible
                byoa_plan_html,
                byoa_reward_html,
                byoa_turns_used,
                byoa_turns_saved,
                byoa_macro_html,
                byoa_run_log_state,
            ],
        )

        # Download button
        download_btn.click(
            fn=on_download_results,
            inputs=[byoa_run_log_state],
            outputs=[download_status],
        )

    return tab
