"""
ToolForge Environment Client
-----------------------------
Client-side wrapper for the ToolForge environment server.

This client maintains a persistent WebSocket connection to the environment
server via the base EnvClient, enabling efficient multi-step interactions.

Usage (async):
    >>> async with ToolForgeEnv(base_url="ws://localhost:8000") as env:
    ...     result = await env.reset(task_id="easy-deploy-notify")
    ...     result = await env.step(action)

Usage (sync wrapper):
    >>> env = ToolForgeEnv(base_url="ws://localhost:8000").sync()
    >>> with env:
    ...     result = env.reset(task_id="easy-deploy-notify")
    ...     result = env.step(action)
"""

import logging

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from toolforge_env.models import ToolForgeAction, ToolForgeObservation, ToolForgeState

# ──────────────────────────────────────────────────────────────────────────────
# Module-level logger for client-side diagnostics.
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


class ToolForgeEnv(EnvClient[ToolForgeAction, ToolForgeObservation, ToolForgeState]):
    """
    Client for the ToolForge environment server.

    Subclasses the core OpenEnv EnvClient and provides environment-specific
    payload shaping and response parsing. Communicates with the server over
    a persistent WebSocket connection.

    Three abstract hooks from EnvClient must be implemented:
        _step_payload  — Action  → JSON dict sent as WebSocket "data"
        _parse_result  — JSON dict → StepResult[ToolForgeObservation]
        _parse_state   — JSON dict → ToolForgeState
    """

    # ──────────────────────────────────────────────────────────────────────
    # *** _step_payload ***
    # Converts a ToolForgeAction into the flat JSON dict that EnvClient
    # will embed under {"type": "step", "data": <this dict>}.
    #
    # CRITICAL: Do NOT wrap in an extra "action" key here.
    # EnvClient.step() already places this payload at "data" in the
    # WebSocket message. The server reads "data" and deserializes it
    # directly via deserialize_action(data, action_cls).
    # Adding an extra "action" layer causes a Pydantic ValidationError
    # because the server receives {"action": {...}} instead of the raw
    # action fields.
    #
    # Reference: envs/coding_env/client.py — _step_payload returns action
    # fields directly without any outer wrapper.
    # ──────────────────────────────────────────────────────────────────────
    def _step_payload(self, action: ToolForgeAction) -> dict:
        """
        Convert a ToolForgeAction into the JSON payload for the step message.

        The returned dict is placed directly under the WebSocket message's
        ``data`` key by the EnvClient base class, so it must be a flat
        representation of the action fields (no extra nesting).

        Args:
            action: The ToolForgeAction the agent is submitting.

        Returns:
            dict: Action fields suitable for JSON serialization.
        """

        # --- Core action fields (always present) ---
        payload: dict = {
            # The discriminating field telling the server which action variant this is
            "action_type": action.action_type,

            # Serialised list of ToolCall objects
            "plan": [call.model_dump() for call in action.plan],

            # Agent's free-form reasoning string
            "reasoning": action.reasoning,
        }

        # --- Optional macro proposal (only when action_type == "propose_plan_with_macro") ---
        if action.macro_proposal is not None:
            payload["macro_proposal"] = action.macro_proposal.model_dump()

        return payload

    # ──────────────────────────────────────────────────────────────────────
    # *** _parse_result ***
    # Converts the server's step/reset response JSON into a typed
    # StepResult[ToolForgeObservation]. The server serialises the
    # observation into the "observation" sub-key inside "data".
    # ──────────────────────────────────────────────────────────────────────
    def _parse_result(self, payload: Dict) -> StepResult[ToolforgeObservation]:
        """
        Parse server response into StepResult[ToolforgeObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ToolforgeObservation
        """
        obs_data = payload.get("observation", {})
        observation = ToolforgeObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
