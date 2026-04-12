# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ToolForge Gradio UI — Main Entry Point
=======================================

Assembles the three-tab Gradio interface and exposes it in two ways:

    1. Standalone dev server (port 7860):
           python gradio_ui.py
           gradio toolforge_env/gradio_ui.py

    2. Mounted at /web on the FastAPI server (OpenEnv standard, HF Spaces):
           Imported by server/app.py via the `gradio_builder` callable.
           The FastAPI server (port 8000) uses create_web_interface_app()
           which calls gradio_builder() and mounts the result at /web as
           a "Custom" tab alongside the OpenEnv default Playground tab.

Tabs:
    Tab 1 — 🎬 Demo Mode          : Pre-scripted agent simulation (no API key needed)
    Tab 2 — 🔌 Bring Your Own Agent: Connect a real LLM to the live environment
    Tab 3 — 🎮 Human vs LLM       : Interactive game — human plans vs LLM plans

Environment Variables (standalone mode only):
    GRADIO_SERVER_PORT  : Override the default port (7860).
    GRADIO_SERVER_NAME  : Override the bind address (default "0.0.0.0").
    GRADIO_SHARE        : Set to "true" to create a public share link.

HF Spaces Deployment:
    The canonical deployment path is via server/app.py → create_web_interface_app()
    → gradio_builder().  This mounts the UI at /web on port 8000 alongside
    the REST API, satisfying OpenEnv's single-port deployment requirement.
    The root URL (/) redirects to /web/ automatically.
"""

import logging
import os

import gradio as gr

from server.ui.shared   import CUSTOM_CSS
from server.ui.demo_tab import build_demo_tab
from server.ui.byoa_tab import build_byoa_tab
from server.ui.hvl_tab  import build_hvl_tab

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server configuration from environment variables
# ---------------------------------------------------------------------------

# Port the Gradio server listens on
GRADIO_PORT: int = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

# Bind address (0.0.0.0 = all interfaces, use 127.0.0.1 for local-only)
GRADIO_HOST: str = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")

# Whether to create a public share link via Gradio's tunnelling service
GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"


# ===========================================================================
# APP BUILDER
# ===========================================================================

def build_app() -> gr.Blocks:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    build_app
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Construct and return the top-level gr.Blocks application.

    Assembles the three tabs by calling each tab's builder function.
    Injects global CSS and metadata.

    Returns:
        A fully configured gr.Blocks instance.  Call .launch() on it
        to start the server, or use it as a WSGI app via .queue().
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    logger.info("Building ToolForge Gradio UI…")

    # css= on gr.Blocks is deprecated in Gradio 6 and is NOT propagated when
    # this child app is wrapped inside OpenEnv's root TabbedInterface at /web.
    # Injecting a <style> tag via gr.HTML is the reliable path: the component
    # renders into the live DOM in both standalone and mounted deployments.
    with gr.Blocks(title="ToolForge — Macro Tool Learning Benchmark") as app:

        # Inject CUSTOM_CSS into the DOM so styles (spinner, winner-banner)
        # apply whether running standalone or mounted under OpenEnv's
        # TabbedInterface.  gr.HTML renders raw HTML directly into the page.
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # -------------------------------------------------------------------
        # TOP-LEVEL HEADER
        # -------------------------------------------------------------------
        gr.Markdown(
            """
            <div style="text-align:center;padding:8px 0 4px;">
                <h1 style="margin:0;font-size:1.8em;">⚒ ToolForge</h1>
                <p style="margin:4px 0 0;color:#a78bfa;font-size:0.95em;">
                    Agentic Efficiency through Macro Tool Learning
                </p>
            </div>
            """
        )

        # -------------------------------------------------------------------
        # THREE TABS
        # -------------------------------------------------------------------
        with gr.Tabs():
            build_demo_tab()   # Tab 1 — Demo Mode
            build_byoa_tab()   # Tab 2 — Bring Your Own Agent
            build_hvl_tab()    # Tab 3 — Human vs LLM

    logger.info("ToolForge Gradio UI built successfully.")
    return app


# ===========================================================================
# MODULE-LEVEL DEMO OBJECT
# ---------------------------------------------------------------------------
# Gradio expects a top-level `demo` variable when running via `gradio app.py`.
# We build it once at import time so both `gradio gradio_ui.py` and
# `python gradio_ui.py` work correctly.
#
# The same object is also returned by `gradio_builder` (below) so that
# server/app.py can mount it at /web without a second build pass.
# ===========================================================================

demo: gr.Blocks = build_app()


# ===========================================================================
# OPENENV-STANDARD GRADIO BUILDER
# ---------------------------------------------------------------------------
# `create_web_interface_app` in openenv.core.env_server.web_interface accepts
# an optional `gradio_builder` callable with this exact signature:
#
#     (web_manager, action_fields, metadata, is_chat_env, title, quick_start_md)
#     -> gr.Blocks
#
# When provided, the returned gr.Blocks is wrapped in a gr.TabbedInterface
# alongside the default OpenEnv Playground tab, then mounted at /web on the
# FastAPI server.  This satisfies the HF Spaces single-port requirement.
# ===========================================================================

def gradio_builder(
    web_manager,       # openenv WebInterfaceManager (unused — UI calls REST API)
    action_fields,     # List[Dict] of action field metadata (unused by custom UI)
    metadata,          # EnvironmentMetadata (unused — UI has its own header)
    is_chat_env,       # bool — False for ToolForge (unused)
    title,             # str — environment display name (unused)
    quick_start_md,    # str — quick-start markdown (unused)
) -> gr.Blocks:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gradio_builder  (OpenEnv-standard callable)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Return the pre-built ToolForge three-tab gr.Blocks UI.

    Called by create_web_interface_app() in server/app.py to mount the
    custom UI at /web as the "Custom" tab alongside the OpenEnv Playground.

    We return the pre-built `demo` object (already constructed at import
    time) rather than calling build_app() again, so the UI is only built
    once regardless of the execution path.

    Args:
        web_manager   : OpenEnv WebInterfaceManager (not used — Gradio
                        event handlers call the REST API via env_client.py).
        action_fields : Pydantic field metadata from ToolForgeAction (unused).
        metadata      : EnvironmentMetadata (unused).
        is_chat_env   : Whether env is chat-style (always False here).
        title         : Environment display name (unused).
        quick_start_md: Quick-start connection guide markdown (unused).

    Returns:
        The fully configured gr.Blocks instance for the ToolForge UI.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    logger.info("gradio_builder called — returning pre-built ToolForge UI.")
    return demo


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    logger.info(
        "Launching ToolForge UI | host=%s port=%d share=%s",
        GRADIO_HOST, GRADIO_PORT, GRADIO_SHARE,
    )

    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        # CSS is injected via gr.HTML(<style>) inside build_app() so it works
        # in both standalone and mounted /web paths.  Theme still goes here.
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        # Enable the queue so streaming / long-running handlers work correctly
        # when Tab 2 (BYOA) real inference is wired in.
        # TODO: Tune max_size and concurrency_count after real backend is live.
    )
