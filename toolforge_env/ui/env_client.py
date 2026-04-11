# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ui/env_client.py — Synchronous HTTP client for the ToolForge environment server.

The ToolForge environment runs as a FastAPI server (server/app.py) exposing:
    POST /reset    Reset episode, get first observation
    POST /step     Submit an action, get next observation + reward + done
    GET  /health   Liveness probe

This module wraps those endpoints with plain synchronous httpx calls so that
Gradio event handlers (which run synchronously) can interact with the env
without asyncio complexity.

The demo tab, BYOA tab, and HvL tab all import from here rather than talking
to the env directly.

Import pattern (matches project convention — runs from toolforge_env/ dir):
    from ui.env_client import env_reset, env_step, check_env_health

TODO: Add retry logic with exponential back-off for HF Space cold starts.
TODO: Add request caching / debounce for rapid button presses.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default env server address (localhost when running locally)
# ---------------------------------------------------------------------------
DEFAULT_ENV_URL: str = "http://localhost:8000"

# Request timeout in seconds for each REST call
REQUEST_TIMEOUT: float = 15.0


# ===========================================================================
# SECTION 1: CONNECTION HELPERS
# ===========================================================================

def check_env_health(base_url: str) -> Tuple[bool, str]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    check_env_health
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Ping the environment server to verify it is reachable.

    Args:
        base_url : Root URL of the env server (e.g. "http://localhost:8000").

    Returns:
        Tuple (is_healthy: bool, status_message: str).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    # Strip trailing slash for clean URL building
    base_url = base_url.rstrip("/")
    url = f"{base_url}/health"

    try:
        resp = httpx.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            logger.info("Env health OK | url=%s", base_url)
            return True, f"✅ Environment server is running at {base_url}"
        logger.warning("Env health check returned %d | url=%s", resp.status_code, base_url)
        return False, f"❌ Server at {base_url} returned HTTP {resp.status_code}"
    except httpx.ConnectError:
        return False, f"❌ Cannot connect to environment at {base_url} — is the server running?"
    except Exception as exc:
        logger.error("Env health check failed | url=%s error=%s", base_url, exc)
        return False, f"❌ Unexpected error: {exc}"


# ===========================================================================
# SECTION 2: ENVIRONMENT INTERACTION
# ===========================================================================

def env_reset(base_url: str, task_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    env_reset
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Call POST /reset on the environment server.

    The server resets its internal state, loads the task group identified
    by task_id, and returns the first observation.

    Args:
        base_url : Root URL of the env server.
        task_id  : Task group identifier — e.g. "easy-deployment-sprints",
                   "medium-traffic-readiness", or the alias "easy" / "medium" / "hard".

    Returns:
        Tuple (result_dict, error_msg).
        result_dict keys on success:
            observation.current_task.prompt     str
            observation.current_task.id         str
            observation.available_tools         list[dict]
            observation.grading                 dict | None
            reward                              float | None
            done                                bool
        error_msg is empty string on success, non-empty on failure.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    base_url = base_url.rstrip("/")
    url      = f"{base_url}/reset"

    # Payload matches the env reset() signature: task_id kwarg
    payload: Dict[str, Any] = {"task_id": task_id}

    try:
        resp = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        logger.info("env_reset OK | task_id=%s", task_id)
        return data, ""
    except httpx.HTTPStatusError as exc:
        msg = f"Reset failed (HTTP {exc.response.status_code}): {exc.response.text[:200]}"
        logger.error(msg)
        return None, msg
    except Exception as exc:
        msg = f"Reset error: {exc}"
        logger.error(msg)
        return None, msg


def env_step(
    base_url:       str,
    plan:           List[str],
    macro_proposal: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    env_step
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Call POST /step on the environment server.

    Constructs a ToolForgeAction payload from the given plan and optional
    macro proposal, submits it, and returns the parsed result.

    Args:
        base_url       : Root URL of the env server.
        plan           : Ordered list of tool name strings for the plan
                         (e.g. ["deploy", "healthcheck", "notify"]).
        macro_proposal : If the agent proposes a new macro this step, pass
                         a dict with keys "name" and "steps" (list of str).
                         Pass None for a plain propose_plan action.

    Returns:
        Tuple (result_dict, error_msg).
        result_dict keys on success:
            observation.current_task            dict
            observation.available_tools         list[dict]
            observation.grading                 dict | None
            reward                              float
            done                                bool
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    base_url = base_url.rstrip("/")
    url      = f"{base_url}/step"

    # Build the plan as a list of ToolCall dicts
    plan_payload: List[Dict[str, str]] = [{"tool_name": t} for t in plan]

    # Determine action_type
    action_type: str = "propose_plan_with_macro" if macro_proposal else "propose_plan"

    # Build macro_proposal payload if present
    macro_payload: Optional[Dict[str, Any]] = None
    if macro_proposal:
        macro_payload = {
            "name":        macro_proposal["name"],
            "description": f"Macro: {' -> '.join(macro_proposal['steps'])}",
            "is_macro":    True,
            "steps":       [{"tool_name": s} for s in macro_proposal["steps"]],
        }

    payload: Dict[str, Any] = {
        "action_type":    action_type,
        "plan":           plan_payload,
        "macro_proposal": macro_payload,
    }

    try:
        resp = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "env_step OK | action_type=%s plan=%s reward=%s done=%s",
            action_type, plan, data.get("reward"), data.get("done"),
        )
        return data, ""
    except httpx.HTTPStatusError as exc:
        msg = f"Step failed (HTTP {exc.response.status_code}): {exc.response.text[:200]}"
        logger.error(msg)
        return None, msg
    except Exception as exc:
        msg = f"Step error: {exc}"
        logger.error(msg)
        return None, msg


# ===========================================================================
# SECTION 3: OBSERVATION PARSERS
# ===========================================================================

def parse_task_from_obs(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parse_task_from_obs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Extract the current_task dict from an env response.

    Args:
        result : Raw response dict from env_reset or env_step.

    Returns:
        current_task dict with keys: id, prompt, difficulty, required_slots,
        baseline_call_count.  Returns a safe empty dict on parse failure.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    try:
        return result.get("observation", {}).get("current_task", {})
    except Exception:
        return {}


def parse_tools_from_obs(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parse_tools_from_obs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Extract available_tools list from an env response.

    Args:
        result : Raw response dict from env_reset or env_step.

    Returns:
        List of tool dicts, each with keys: name, description, is_macro, steps.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    try:
        return result.get("observation", {}).get("available_tools", [])
    except Exception:
        return []


def extract_macros(available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    extract_macros
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Filter available_tools to return only macro entries.

    The env returns macros with is_macro=True and a steps list.  We
    normalise steps from either List[ToolCall-dict] or List[str] so
    the UI renderers always receive List[str].

    Args:
        available_tools : List of tool dicts from parse_tools_from_obs.

    Returns:
        List of macro dicts with keys "name" and "steps" (list of str).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    macros: List[Dict[str, Any]] = []
    for tool in available_tools:
        if not tool.get("is_macro"):
            continue
        raw_steps = tool.get("steps", [])
        # Normalise steps: may be list of {"tool_name": "..."} dicts or plain strings
        steps: List[str] = []
        for s in raw_steps:
            if isinstance(s, dict):
                steps.append(s.get("tool_name", str(s)))
            else:
                steps.append(str(s))
        macros.append({"name": tool["name"], "steps": steps})
    return macros
