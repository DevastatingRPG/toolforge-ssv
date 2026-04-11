# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Toolforge Env Environment.

This module creates an HTTP server that exposes the ToolforgeEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

The Gradio UI is mounted at /web (OpenEnv standard) so that the full
deployment runs on a single port — a requirement for HuggingFace Spaces.
Visiting the root URL (/) redirects automatically to /web/.

Endpoints (REST API):
    POST /reset          — Reset the environment
    POST /step           — Execute an action
    GET  /state          — Get current environment state
    GET  /schema         — Get action/observation schemas
    WS   /ws             — WebSocket endpoint for persistent sessions

Endpoints (Web interface, added by create_web_interface_app):
    GET  /               — Redirect → /web/
    GET  /web/           — Gradio UI (OpenEnv Playground + ToolForge Custom tabs)
    POST /web/reset      — Web-facing reset (used by Gradio handlers)
    POST /web/step       — Web-facing step  (used by Gradio handlers)
    GET  /web/state      — Web-facing state (used by Gradio handlers)
    GET  /web/metadata   — Environment metadata JSON
    WS   /ws/ui          — WebSocket for real-time Gradio state updates

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenEnv web-interface factory — mounts Gradio at /web on the FastAPI app
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.web_interface import create_web_interface_app
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies with '\n    uv sync\n'"
    ) from e

# ---------------------------------------------------------------------------
# ToolForge environment models and implementation
# ---------------------------------------------------------------------------
try:
    from models import ToolForgeAction, ToolForgeObservation
    from server.toolforge_env_environment import ToolforgeEnvironment
except ModuleNotFoundError:
    from ..models import ToolForgeAction, ToolForgeObservation           # type: ignore[no-redef]
    from .toolforge_env_environment import ToolforgeEnvironment           # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Custom Gradio UI builder (OpenEnv-standard callable)
# ---------------------------------------------------------------------------
# gradio_builder has the signature:
#   (web_manager, action_fields, metadata, is_chat_env, title, quick_start_md)
#   -> gr.Blocks
#
# create_web_interface_app() wraps it in a TabbedInterface alongside the
# default OpenEnv Playground tab, then mounts the combined UI at /web.
# ---------------------------------------------------------------------------
try:
    from gradio_app import gradio_builder
except ModuleNotFoundError:
    from ..gradio_app import gradio_builder                               # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Create the FastAPI app with Gradio mounted at /web
# ---------------------------------------------------------------------------
logger.info("Creating ToolForge FastAPI app with Gradio UI at /web …")

app = create_app(
    ToolforgeEnvironment,
    ToolForgeAction,
    ToolForgeObservation,
    env_name="toolforge_env",
    # Limit concurrent WebSocket env sessions; increase for parallel eval runs.
    max_concurrent_envs=1,
    # Our three-tab ToolForge UI is injected as the "Custom" tab alongside
    # the default OpenEnv "Playground" tab.
    gradio_builder=gradio_builder,
)

logger.info("ToolForge app ready.  Gradio UI accessible at /web/")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main(host: str = "0.0.0.0", port: int = 8000):
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    main
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m toolforge_env.server.app

    Args:
        host : Host address to bind to (default: "0.0.0.0").
        port : Port number to listen on (default: 8000).

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn toolforge_env.server.app:app --workers 4
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the ToolForge environment server."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    main(host=args.host, port=args.port)
