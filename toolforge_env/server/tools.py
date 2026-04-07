import logging
from typing import List

# Internal imports for building Tool models
from models import Tool


# Logger instance for the tools module
logger = logging.getLogger(__name__)



def build_atomic_tools() -> List[Tool]:
    """
    Returns a fresh list of the 6 fundamental atomic DevOps tools available 
    in the environment. Each tool is represented as a Tool Pydantic model.
    """
    
    logger.info("Building fresh atomic tools list.")
    
    # ---------------------------------------------------------
    # 1. Deploy Tool
    # ---------------------------------------------------------
    # Description: Deploys a specified service
    # Parameters: service_name (string), version (string)
    tool_deploy = Tool(
        name="deploy",
        description="Deploys a specified service or application version to a target environment.",
    )
    
    # ---------------------------------------------------------
    # 2. Healthcheck Tool
    # ---------------------------------------------------------
    # Description: Checks if a service is healthy
    # Parameters: service_name (string)
    tool_healthcheck = Tool(
        name="healthcheck",
        description="Checks the current health status of a running service.",
    )

    # ---------------------------------------------------------
    # 3. Notify Tool
    # ---------------------------------------------------------
    # Description: Sends an alert or notification
    # Parameters: channel (string), message (string)
    tool_notify = Tool(
        name="notify",
        description="Sends a notification to a specific channel (e.g., Slack, Email) regarding system status.",
    )

    # ---------------------------------------------------------
    # 4. Rollback Tool
    # ---------------------------------------------------------
    # Description: Reverts a deployment to its previous stable state
    # Parameters: service_name (string)
    tool_rollback = Tool(
        name="rollback",
        description="Reverts a deployed service to the previous stable version if something goes wrong.",
    )

    # ---------------------------------------------------------
    # 5. Scale Tool
    # ---------------------------------------------------------
    # Description: Adjusts the replica count for a service
    # Parameters: service_name (string), replicas (integer)
    tool_scale = Tool(
        name="scale",
        description="Adjusts the number of running replicas for a given service.",
    )

    # ---------------------------------------------------------
    # 6. Restart Tool
    # ---------------------------------------------------------
    # Description: Bounces a service
    # Parameters: service_name (string)
    tool_restart = Tool(
        name="restart",
        description="Performs a rolling restart on a specified service.",
    )

    # Collection list holding all the instantiated Tool models
    tools_list = [
        tool_deploy,
        tool_healthcheck,
        tool_notify,
        tool_rollback,
        tool_scale,
        tool_restart,
    ]
    
    logger.debug(f"Successfully constructed {len(tools_list)} tools.")
    
    return tools_list

