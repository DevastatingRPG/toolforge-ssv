import logging
from typing import List

# Internal imports for building Tool models
from models import Tool


# Logger instance for the tools module
logger = logging.getLogger(__name__)



def build_atomic_tools() -> List[Tool]:
    """
    Returns a fresh list of fundamental atomic DevOps tools available 
    in the environment. Each tool is represented as a Tool Pydantic model.
    """
    
    logger.info("Building fresh atomic tools list.")
    
    tool_deploy = Tool(
        name="deploy",
        description="Deploys a specified service or application version to a target environment.",
    )
    
    tool_patch = Tool(
        name="patch",
        description="Applies a specific security patch or hotfix to a running service.",
    )

    tool_healthcheck = Tool(
        name="healthcheck",
        description="Checks the current health status of a running service.",
    )
    
    tool_run_tests = Tool(
        name="run_tests",
        description="Executes a test suite against a deployed service to verify it is functioning correctly.",
    )

    tool_ping = Tool(
        name="ping",
        description="Performs a quick network ping to an endpoint to verify basic connectivity.",
    )

    tool_notify = Tool(
        name="notify",
        description="Sends a notification to a specific channel (e.g., Slack, Email) regarding system status.",
    )
    
    tool_pagerduty_alert = Tool(
        name="pagerduty_alert",
        description="Triggers a high-priority incident alert via PagerDuty.",
    )

    tool_rollback = Tool(
        name="rollback",
        description="Reverts a deployed service to the previous stable version if something goes wrong.",
    )

    tool_scale = Tool(
        name="scale",
        description="Adjusts the number of running replicas for a given service.",
    )

    tool_restart = Tool(
        name="restart",
        description="Performs a rolling restart on a specified service.",
    )

    # Collection list holding all the instantiated Tool models
    tools_list = [
        tool_deploy,
        tool_patch,
        tool_healthcheck,
        tool_run_tests,
        tool_ping,
        tool_notify,
        tool_pagerduty_alert,
        tool_rollback,
        tool_scale,
        tool_restart,
    ]
    
    logger.debug(f"Successfully constructed {len(tools_list)} tools.")
    
    return tools_list
