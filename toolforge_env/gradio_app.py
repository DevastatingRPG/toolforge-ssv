# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ToolForge Gradio UI — Main Entry Point
=======================================

Assembles the three-tab Gradio interface and launches the server.

Tabs:
    Tab 1 — 🎬 Demo Mode          : Pre-scripted agent simulation (no API key needed)
    Tab 2 — 🔌 Bring Your Own Agent: Connect a real LLM to the live environment
    Tab 3 — 🎮 Human vs LLM       : Interactive game — human plans vs LLM plans

Usage:
    # From the toolforge_env package root:
    python gradio_app.py

    # Or with uvicorn-style hot-reload via Gradio CLI:
    gradio toolforge_env/gradio_app.py

    # Or import the `demo` object for programmatic mounting:
    from gradio_app import demo
    demo.launch(server_port=7860)

Environment Variables:
    GRADIO_SERVER_PORT  : Override the default port (7860).
    GRADIO_SERVER_NAME  : Override the bind address (default "0.0.0.0").
    GRADIO_SHARE        : Set to "true" to create a public share link.

Note:
    This file does NOT modify or import from server/app.py.  It is a
    standalone UI layer that can run independently of the FastAPI server.
    When a real backend connection is needed (Tab 2 & 3), the UI calls
    the environment via ToolforgeEnv client (inference.py logic) — not
    by importing the server module directly.
"""

import logging
import os

import gradio as gr

from ui.shared import CUSTOM_CSS
from ui.demo_tab import build_demo_tab
from ui.byoa_tab import build_byoa_tab
from ui.hvl_tab  import build_hvl_tab

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

    # Gradio 6+ moved theme and css to launch() — keep Blocks minimal
    with gr.Blocks(title="ToolForge — Macro Tool Learning Benchmark") as app:

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
# We build it once at import time so both `gradio gradio_app.py` and
# `python gradio_app.py` work correctly.
# ===========================================================================

demo: gr.Blocks = build_app()


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
        # Gradio 6+: theme and css are passed here
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
        # Enable the queue so streaming / long-running handlers work correctly
        # when Tab 2 (BYOA) real inference is wired in.
        # TODO: Tune max_size and concurrency_count after real backend is live.
    )
