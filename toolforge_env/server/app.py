"""
FastAPI application for the ToolForge Environment.

Exposes the ToolForgeEnvironment over HTTP and WebSocket endpoints
compatible with the OpenEnv EnvClient.

Endpoints (registered automatically by create_app):
    POST /reset   — Reset episode, return initial observation
    POST /step    — Execute action, return observation + reward + done
    GET  /state   — Return current ToolForgeState
    GET  /schema  — Return action/observation JSON schemas
    GET  /health  — Liveness check
    WS   /ws      — Persistent WebSocket session (preferred by EnvClient)

Usage:
    # Development:
    uvicorn toolforge_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn toolforge_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Via entry point (defined in pyproject.toml):
    python -m toolforge_env.server.app
"""

import logging

from openenv.core.env_server import create_app

from toolforge_env.models import ToolForgeAction, ToolForgeObservation
from toolforge_env.server.toolforge_environment import ToolForgeEnvironment

# ──────────────────────────────────────────────────────────────────────────────
# Module logger.
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Application factory.
#
# create_app() registers all OpenEnv-standard routes and the WebSocket
# session manager. We pass the *class* (not an instance) so that the server
# can call ToolForgeEnvironment() as a factory for each new WebSocket session.
#
# max_concurrent_envs=4 allows up to 4 simultaneous benchmark agents.
# This is safe because SUPPORTS_CONCURRENT_SESSIONS is True on the class —
# each session receives its own ToolForgeEnvironment instance with no shared
# mutable state.
# ──────────────────────────────────────────────────────────────────────────────
app = create_app(
    ToolForgeEnvironment,
    ToolForgeAction,
    ToolForgeObservation,
    env_name="toolforge_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Entry point for starting the ToolForge server directly.

    Can be invoked via:
        python -m toolforge_env.server.app
        uv run --project . server

    Args:
        host: Bind address (default: all interfaces).
        port: TCP port to listen on (default: 8000).
    """

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    # ── CLI argument parsing ────────────────────────────────────────────────
    # Allows overriding the port at launch without editing source.
    parser = argparse.ArgumentParser(description="ToolForge Environment Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    main(host=args.host, port=args.port)
