"""Semantic slot definitions for the ToolForge judge.

Each slot represents a granular action phase (execution, verification, notification)
within a DevOps workflow. Mapped by the Stage-2 semantic judge.
"""

DEVOPS_SLOTS = {
    "deployment_execution": "Execute a deployment or release of a new version to a target environment.",
    "deployment_verification": "Check health, ping, or verify a newly deployed service is healthy.",
    "deployment_notification": "Notify or alert a channel/person that a deployment has finished.",
    "rollback_execution": "Revert or restore a service to a previous stable state.",
    "rollback_verification": "Verify the health or stability of a rolled-back service.",
    "rollback_notification": "Send an alert or message indicating a rollback event occurred.",
    "scaling_execution": "Increase or decrease the replica count or scale a service.",
    "scaling_verification": "Verify that the service is healthy and stabilized after scaling.",
    "scaling_notification": "Notify stakeholders that a service scaling event occurred.",
    "restart_execution": "Restart, bounce, or cycle a service instance.",
    "restart_verification": "Verify the service is running properly after being restarted.",
    "restart_notification": "Notify stakeholders that a service has been restarted.",
}
