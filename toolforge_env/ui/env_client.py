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

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment-aware URL resolution
# ---------------------------------------------------------------------------

def resolve_env_url() -> str:
    """Resolve the environment server URL from context.

    Priority order:
        1. ``TOOLFORGE_ENV_URL`` env var (explicit override).
        2. ``SPACE_HOST`` env var (set automatically by Hugging Face Spaces)
           → ``https://{SPACE_HOST}``.
        3. Fallback to ``http://localhost:8000`` for local dev.
    """
    explicit = os.getenv("TOOLFORGE_ENV_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")

    space_host = os.getenv("SPACE_HOST", "").strip()
    if space_host:
        return f"https://{space_host}"

    return "http://localhost:8000"


DEFAULT_ENV_URL: str = resolve_env_url()

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
    Two-level readiness probe for the environment server.

    Level 1 — ``GET /health``    → server alive.
    Level 2 — ``POST /reset``    → ToolForge API shape confirmed.

    Args:
        base_url : Root URL of the env server (e.g. "http://localhost:8000").

    Returns:
        Tuple (is_healthy: bool, status_message: str).
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    base_url = base_url.rstrip("/")

    # --- Level 1: server alive -------------------------------------------
    health_url = f"{base_url}/health"
    try:
        resp = httpx.get(health_url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("Health check HTTP %d | url=%s", resp.status_code, base_url)
            return False, f"❌ Server at {base_url} returned HTTP {resp.status_code}"
    except httpx.ConnectError:
        return False, f"❌ Cannot connect to environment at {base_url} — is the server running?"
    except Exception as exc:
        logger.error("Health check failed | url=%s error=%s", base_url, exc)
        return False, f"❌ Unexpected error: {exc}"

    logger.info("Level-1 health OK | url=%s", base_url)

    # --- Level 2: API shape (lightweight reset) --------------------------
    reset_url = f"{base_url}/reset"
    try:
        resp = httpx.post(reset_url, json={"task_id": "easy"}, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("API shape check HTTP %d | url=%s", resp.status_code, base_url)
            return (
                False,
                f"⚠️ Server alive but POST /reset returned HTTP {resp.status_code}. "
                f"ToolForge API may not be configured correctly.",
            )
        data = resp.json()
        obs = data.get("observation", {})
        if "current_task" not in obs or "available_tools" not in obs:
            return (
                False,
                f"⚠️ Server alive but /reset response is missing expected fields. "
                f"Check that the ToolForge environment is loaded.",
            )
    except Exception as exc:
        logger.warning("API shape probe failed | url=%s error=%s", base_url, exc)
        return (
            False,
            f"⚠️ Server alive (health OK) but API probe failed: {exc}",
        )

    logger.info("Level-2 API shape OK | url=%s", base_url)
    return (
        True,
        f"✅ Environment server ready at {base_url} — health OK, API verified.",
    )


# ===========================================================================
# SECTION 2: ENVIRONMENT INTERACTION
# ===========================================================================

def env_reset(base_url: str, task_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    env_reset
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Call POST /web/reset on the environment server.

    Uses the /web/reset endpoint (provided by create_web_interface_app)
    rather than the bare /reset endpoint.  The key difference is that
    /web/reset is backed by WebInterfaceManager which holds a PERSISTENT
    env instance — state is preserved between reset and subsequent step
    calls.  The bare /reset endpoint creates a fresh env per request
    (stateless), so /step would run on an un-initialized env, returning
    "Default task" and done=True immediately.

    Args:
        base_url : Root URL of the env server (e.g. "http://localhost:8000").
        task_id  : Task group identifier — e.g. "easy-deployment-sprints"
                   or the alias "easy" / "medium" / "hard".

    Returns:
        Tuple (result_dict, error_msg).
        result_dict keys on success:
            observation.current_task.prompt     str
            observation.current_task.id         str
            observation.available_tools         list[dict]
            observation.metadata.total_tasks    int   ← episode length
            observation.grading                 dict | None
            reward                              float | None
            done                                bool
        error_msg is empty string on success, non-empty on failure.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    base_url = base_url.rstrip("/")
    url      = f"{base_url}/web/reset"   # WebInterfaceManager — stateful env

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
    Call POST /web/step on the environment server.

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
    url      = f"{base_url}/web/step"   # WebInterfaceManager — stateful env

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

    # OpenEnv's HTTP /step endpoint expects the action nested under
    # a top-level "action" field, not the raw action object at the root.
    payload: Dict[str, Any] = {
        "action": {
            "action_type":    action_type,
            "plan":           plan_payload,
            "macro_proposal": macro_payload,
        }
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


def parse_total_tasks_from_obs(result: Dict[str, Any]) -> int:
    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parse_total_tasks_from_obs
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Extract the total number of tasks in the current episode from the
    observation metadata returned by env_reset().

    The environment sets ``observation.metadata.total_tasks`` in
    reset() so the UI can track episode progress without relying on
    env.done (which is unreliable in stateless REST mode).

    Args:
        result : Raw response dict from env_reset().

    Returns:
        Total task count as a positive int, or 0 on parse failure.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    try:
        meta = result.get("observation", {}).get("metadata", {}) or {}
        count = meta.get("total_tasks", 0)
        return int(count) if count else 0
    except Exception:
        return 0


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
