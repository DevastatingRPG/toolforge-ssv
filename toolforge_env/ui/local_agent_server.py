#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
local_agent_server.py — OpenAI-compatible proxy for local LLM backends.
========================================================================

Exposes a POST /v1/chat/completions endpoint that forwards requests to
a local model server (Ollama, LM Studio, etc.) and returns an
OpenAI-compatible response. This lets you point the BYOA tab's ngrok
tunnel at this server instead of the raw Ollama/LM Studio port, giving
you a clean OpenAI-compatible endpoint with request logging.

Usage:
    pip install fastapi uvicorn httpx

    # Ollama backend (default):
    python local_agent_server.py --model llama3 --port 8080

    # LM Studio backend:
    python local_agent_server.py --model your-model --port 8080 \\
        --backend-url http://localhost:1234

    # Custom model and backend:
    python local_agent_server.py \\
        --model phi3:mini \\
        --port 8080 \\
        --backend-url http://localhost:11434

Then expose with ngrok:
    ngrok http 8080

Paste the resulting https://xxxx.ngrok-free.app/v1 URL into the BYOA tab.
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("local_agent_server")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ToolForge Local Agent Proxy",
    description="OpenAI-compatible proxy for local LLM backends (Ollama, LM Studio).",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Runtime configuration (populated by CLI args in __main__)
# ---------------------------------------------------------------------------
_config: Dict[str, Any] = {
    "model":       "llama3",
    "backend_url": "http://localhost:11434",
    "port":        8080,
}


# ===========================================================================
# SECTION 1: ENDPOINTS
# ===========================================================================

@app.get("/")
async def root():
    """Health check / welcome."""
    return {
        "status":  "ok",
        "model":   _config["model"],
        "backend": _config["backend_url"],
        "note":    "POST /v1/chat/completions to call the local model.",
    }


@app.get("/v1/models")
async def list_models():
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    list_models
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Return a minimal OpenAI-compatible models list containing just the
    configured local model.  The BYOA Test Connection button calls this.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    return {
        "object": "list",
        "data": [
            {
                "id":      _config["model"],
                "object":  "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    chat_completions
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Receive an OpenAI-format chat completion request, log the incoming
    prompt to stdout, forward to the configured backend, and return
    the backend response verbatim.

    The model field in the incoming request is overridden with
    _config["model"] so that whatever the client sends, the correct
    local model name is used.

    Args:
        request : Raw FastAPI Request (we parse body manually to log it).

    Returns:
        JSONResponse with the backend's chat completion response.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    try:
        body: Dict[str, Any] = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")

    # Override model name with configured local model
    body["model"] = _config["model"]

    # Print incoming prompt to stdout so you can watch the agent's reasoning
    messages: List[Dict[str, str]] = body.get("messages", [])
    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    logger.info(
        "─── Incoming request ───\n"
        "  model   : %s\n"
        "  messages: %d\n"
        "  last_user_prompt (first 300 chars):\n%s",
        _config["model"],
        len(messages),
        last_user[:300],
    )

    # Forward to backend
    backend_url = _config["backend_url"].rstrip("/") + "/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(backend_url, json=body)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        logger.error("Cannot reach backend at %s", backend_url)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Cannot connect to local backend at {_config['backend_url']}. "
                f"Is Ollama / LM Studio running?"
            ),
        )
    except httpx.HTTPStatusError as exc:
        logger.error("Backend returned HTTP %d", exc.response.status_code)
        raise HTTPException(
            status_code=502,
            detail=f"Backend error {exc.response.status_code}: {exc.response.text[:300]}",
        )
    except Exception as exc:
        logger.error("Unexpected backend error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))

    # Log the response content
    choices = data.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        logger.info("  → response (first 300 chars): %s", content[:300])

    return JSONResponse(content=data)


# ===========================================================================
# SECTION 2: CLI ENTRY POINT
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ToolForge local agent proxy server (OpenAI-compatible).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python local_agent_server.py --model llama3 --port 8080
              python local_agent_server.py --model phi3:mini --backend-url http://localhost:11434
              python local_agent_server.py --model local-model --backend-url http://localhost:1234 --port 8081
            """
        ) if False else "",   # textwrap not imported here — plain epilog
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Local model name to forward requests to (default: llama3)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:11434",
        help="Base URL of the local model server, e.g. http://localhost:11434 for Ollama (default: http://localhost:11434)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import textwrap  # noqa: F811 (only needed for __main__ block)

    args = parse_args()

    # Populate runtime config
    _config["model"]       = args.model
    _config["backend_url"] = args.backend_url.rstrip("/")
    _config["port"]        = args.port

    logger.info(
        "Starting ToolForge local agent proxy\n"
        "  model       : %s\n"
        "  backend     : %s\n"
        "  listening on: http://localhost:%d\n"
        "\n"
        "Expose with ngrok:\n"
        "  ngrok http %d\n"
        "Then paste the forwarding URL (+ /v1) into the BYOA tab.",
        _config["model"],
        _config["backend_url"],
        _config["port"],
        _config["port"],
    )

    uvicorn.run(app, host="0.0.0.0", port=_config["port"], log_level="info")
