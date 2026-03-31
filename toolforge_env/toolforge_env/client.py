from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from toolforge_env.models import ToolForgeAction, ToolForgeObservation, ToolForgeState

class ToolForgeEnv(EnvClient[ToolForgeAction, ToolForgeObservation, ToolForgeState]):
    """
    Client-side wrapper for the ToolForge environment server.
    """

    def _step_payload(self, action: ToolForgeAction) -> dict:
        """
        Format the action into the expected JSON payload shape for the server step endpoint.
        """
        payload = {
            "action_type": action.action_type,
            "plan": [call.model_dump() for call in action.plan],
            "reasoning": action.reasoning
        }
        
        if action.macro_proposal:
            payload["macro_proposal"] = action.macro_proposal.model_dump()
            
        return dict(action=payload)

    def _parse_result(self, payload: dict) -> StepResult[ToolForgeObservation]:
        """
        Parse the step endpoint response into a StepResult.
        """
        obs = ToolForgeObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=bool(payload.get("done", False))
        )

    def _parse_state(self, payload: dict) -> ToolForgeState:
        """
        Parse the state endpoint response into a ToolForgeState.
        """
        return ToolForgeState(**payload)
