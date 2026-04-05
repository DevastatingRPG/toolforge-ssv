from models import Task

# Tool constraints:
# Available tools: deploy, healthcheck, notify, rollback, scale, restart
# Slots format: [action]_execution, [action]_verification, [action]_notification

# ======================== EASY TASKS ========================

EASY_TASKS = [
    {
        "list_id": "easy-deployment-basic",
        "description": "Basic deployment and validation tasks",
        "tasks": [
            Task(
                id="easy-deploy-notify",
                prompt="Deploy the 'frontend-web' service version 'v2.1.0', check its health, and notify the '#deployments' channel that it is done.",
                difficulty="easy",
                required_slots=[
                    "deployment_execution",
                    "deployment_verification",
                    "deployment_notification"
                ],
                baseline_call_count=3 # deploy, healthcheck, notify
            ),
            Task(
                id="easy-deploy-restart",
                prompt="Deploy 'backend-api' version 'v1.4.2', check its health. It usually needs a restart after deployment, so restart it, check health again, and then notify '#backend-ops' that the deploy is complete.",
                difficulty="easy",
                required_slots=[
                    "deployment_execution",
                    "deployment_verification",
                    "restart_execution",
                    "restart_verification",
                    "deployment_notification"
                ],
                baseline_call_count=5 # deploy, healthcheck, restart, healthcheck, notify
            ),
            Task(
                id="easy-deploy-scale",
                prompt="Deploy 'analytics-engine' version 'v3.0.0', then quickly scale it to 10 replicas for the incoming load test. Check health to verify and notify '#data-team'.",
                difficulty="easy",
                required_slots=[
                    "deployment_execution",
                    "scaling_execution",
                    "deployment_verification",
                    "deployment_notification"
                ],
                baseline_call_count=4 # deploy, scale, healthcheck, notify
            ),
        ]
    },
    {
        "list_id": "easy-scaling-ops",
        "description": "Scaling and resource management tasks",
        "tasks": [
            Task(
                id="easy-scale-notify",
                prompt="Scale the 'queue-worker' service to 5 replicas, check its health to ensure all are up, and notify '#ops-alerts' with the updated scaling status.",
                difficulty="easy",
                required_slots=[
                    "scaling_execution",
                    "scaling_verification",
                    "scaling_notification"
                ],
                baseline_call_count=3 # scale, healthcheck, notify
            ),
            Task(
                id="easy-scale-down",
                prompt="Scale down the 'batch-processor' service from 8 to 3 replicas to save costs during off-peak hours. Verify health and notify '#cost-ops' when complete.",
                difficulty="easy",
                required_slots=[
                    "scaling_execution",
                    "scaling_verification",
                    "scaling_notification"
                ],
                baseline_call_count=3 # scale, healthcheck, notify
            ),
            Task(
                id="easy-scale-verify",
                prompt="Scale 'user-service' to 4 replicas and perform a thorough health check on all replicas to ensure they are fully ready.",
                difficulty="easy",
                required_slots=[
                    "scaling_execution",
                    "scaling_verification"
                ],
                baseline_call_count=2 # scale, healthcheck
            ),
        ]
    },
    {
        "list_id": "easy-maintenance-recovery",
        "description": "Service maintenance and recovery tasks",
        "tasks": [
            Task(
                id="easy-restart-notify",
                prompt="Restart the 'cache-redis' service because it has been acting up. Check its health afterwards and notify '#db-admins' that the bounce is complete.",
                difficulty="easy",
                required_slots=[
                    "restart_execution",
                    "restart_verification",
                    "restart_notification"
                ],
                baseline_call_count=3 # restart, healthcheck, notify
            ),
            Task(
                id="easy-rollback-notify",
                prompt="The recent deployment of 'auth-service' caused a spike in errors. Rollback the deployment, check the health, and notify '#on-call' that the service has been rolled back.",
                difficulty="easy",
                required_slots=[
                    "rollback_execution",
                    "rollback_verification",
                    "rollback_notification"
                ],
                baseline_call_count=3 # rollback, healthcheck, notify
            ),
            Task(
                id="easy-restart-verify",
                prompt="Restart 'payment-service' and verify it comes back healthy with all its dependent connections restored.",
                difficulty="easy",
                required_slots=[
                    "restart_execution",
                    "restart_verification"
                ],
                baseline_call_count=2 # restart, healthcheck
            ),
        ]
    },
]

# ======================== MEDIUM TASKS ========================

MEDIUM_TASKS = [
    {
        "list_id": "medium-multi-service-deploy",
        "description": "Coordinated deployment of multiple services with dependencies",
        "tasks": [
            Task(
                id="medium-deploy-dependent-services",
                prompt="Deploy 'data-pipeline' v2.0.0, then deploy 'analytics-ui' v1.8.0 which depends on it. Verify each with health checks, and notify '#data-platform' when both are stable.",
                difficulty="medium",
                required_slots=[
                    "deployment_execution",
                    "deployment_verification",
                    "deployment_notification"
                ],
                baseline_call_count=5 # deploy, healthcheck, deploy, healthcheck, notify
            ),
            Task(
                id="medium-canary-rollout",
                prompt="Perform a canary rollout of 'recommendations-engine' v4.1.0 by deploying to partial capacity, scaling up after a healthcheck, and verifying.",
                difficulty="medium",
                required_slots=[
                    "deployment_execution",
                    "deployment_verification",
                    "scaling_execution",
                    "scaling_verification"
                ],
                baseline_call_count=4 # deploy, healthcheck, scale, healthcheck
            ),
            Task(
                id="medium-blue-green-prep",
                prompt="Deploy 'api-gateway' v3.0.0 in a separate configuration, verify it works, then restart the legacy proxy to redirect flow, and notify success.",
                difficulty="medium",
                required_slots=[
                    "deployment_execution",
                    "deployment_verification",
                    "restart_execution",
                    "restart_verification",
                    "deployment_notification"
                ],
                baseline_call_count=5 # deploy, healthcheck, restart, healthcheck, notify
            ),
        ]
    },
    {
        "list_id": "medium-troubleshooting-recovery",
        "description": "Diagnosis and recovery from failures with multiple steps",
        "tasks": [
            Task(
                id="medium-partial-failure-recovery",
                prompt="Three replicas of 'worker-pool' are malfunctioning. Restart them, scale up if needed to maintain capacity, and validate the service is fully recovered then notify operators.",
                difficulty="medium",
                required_slots=[
                    "restart_execution",
                    "restart_verification",
                    "scaling_execution",
                    "scaling_verification",
                    "restart_notification"
                ],
                baseline_call_count=5 # restart, healthcheck, scale, healthcheck, notify
            ),
            Task(
                id="medium-cascade-failure",
                prompt="An error in 'service-a' is causing failures. Rollback 'service-a', check health, restart its dependent 'service-b' and confirm health.",
                difficulty="medium",
                required_slots=[
                    "rollback_execution",
                    "rollback_verification",
                    "restart_execution",
                    "restart_verification"
                ],
                baseline_call_count=4 # rollback, healthcheck, restart, healthcheck
            ),
            Task(
                id="medium-resource-constraint",
                prompt="The 'ml-trainer' service is out of memory. Restart it, and if it fails to stabilise with a healthcheck, scale it down to prevent cluster overload and verify.",
                difficulty="medium",
                required_slots=[
                    "restart_execution",
                    "restart_verification",
                    "scaling_execution",
                    "scaling_verification"
                ],
                baseline_call_count=4 # restart, healthcheck, scale, healthcheck
            ),
        ]
    },
    {
        "list_id": "medium-maintenance-windows",
        "description": "Coordinated maintenance with minimal downtime",
        "tasks": [
            Task(
                id="medium-drain-and-upgrade",
                prompt="Scale 'connection-pool' down to 1 instance temporarily, deploy v2.5.1, restart the service to apply new config, scale back up, and notify teams.",
                difficulty="medium",
                required_slots=[
                    "scaling_execution",
                    "deployment_execution",
                    "restart_execution",
                    "scaling_verification",
                    "deployment_notification"
                ],
                baseline_call_count=6 # scale, deploy, restart, scale, healthcheck, notify
            ),
            Task(
                id="medium-rolling-update",
                prompt="Perform a rolling application update by doing a deployment, immediately performing a healthcheck, and notifying the ops center about the updated nodes.",
                difficulty="medium",
                required_slots=[
                    "deployment_execution",
                    "deployment_verification",
                    "deployment_notification"
                ],
                baseline_call_count=5  # deploy, healthcheck, deploy, healthcheck, notify
            ),
        ]
    },
]

# ======================== HARD TASKS ========================

HARD_TASKS = [
    {
        "list_id": "hard-data-migration",
        "description": "Complex data migrations with validation and rollback planning",
        "tasks": [
            Task(
                id="hard-schema-migration-zero-downtime",
                prompt="Migrate 'user-database' dynamically: Deploy the schema-compatible service backend, restart all dependencies, do a health check, and if anything fails, be ready to rollback and notify.",
                difficulty="hard",
                required_slots=[
                    "deployment_execution",
                    "restart_execution",
                    "deployment_verification",
                    "rollback_execution",
                    "rollback_notification"
                ],
                baseline_call_count=5 # deploy, restart, healthcheck, rollback, notify
            ),
            Task(
                id="hard-cross-region-sync",
                prompt="Sync process failed during 'payment-ledger' update. Rollback the primary cluster, check health, scale up replicas for retry, deploy the patch, and verify health returning.",
                difficulty="hard",
                required_slots=[
                    "rollback_execution",
                    "rollback_verification",
                    "scaling_execution",
                    "deployment_execution",
                    "deployment_verification"
                ],
                baseline_call_count=5 # rollback, healthcheck, scale, deploy, healthcheck
            ),
        ]
    },
    {
        "list_id": "hard-disaster-recovery",
        "description": "Large-scale disaster recovery and resilience scenarios",
        "tasks": [
            Task(
                id="hard-full-region-failover",
                prompt="Primary database is unresponsive creating massive latency. scale down the problematic region edge nodes, restart auth servers to clear bad connections, deploy temporary fallback static pages, verify system stabilization and notify incident response.",
                difficulty="hard",
                required_slots=[
                    "scaling_execution",
                    "restart_execution",
                    "deployment_execution",
                    "deployment_verification",
                    "deployment_notification"
                ],
                baseline_call_count=5 # scale, restart, deploy, healthcheck, notify
            ),
            Task(
                id="hard-cascade-recovery",
                prompt="Cascading API failure. Restart the web proxy, rollback the cache layer to stable, deploy new connection timeout configs, perform thorough healthchecks on all components, and message stakeholders.",
                difficulty="hard",
                required_slots=[
                    "restart_execution",
                    "rollback_execution",
                    "deployment_execution",
                    "deployment_verification",
                    "deployment_notification"
                ],
                baseline_call_count=6 # restart, rollback, deploy, healthcheck, healthcheck, notify
            ),
        ]
    },
    {
        "list_id": "hard-optimization-refactor",
        "description": "Large-scale refactoring and optimization with complex validation",
        "tasks": [
            Task(
                id="hard-service-decomposition",
                prompt="Break 'admin-service' into smaller chunks. Deploy auth component, healthcheck, deploy datastore proxy, healthcheck, deploy UI, healthcheck, then scale down the original legacy monolith and notify completion.",
                difficulty="hard",
                required_slots=[
                    "deployment_execution",
                    "deployment_verification",
                    "scaling_execution",
                    "deployment_notification"
                ],
                baseline_call_count=8 # deploy, healthcheck, deploy, healthcheck, deploy, healthcheck, scale, notify
            ),
            Task(
                id="hard-performance-optimization",
                prompt="Optimize 'search-engine'. First, deploy the new binary. Restart the search workers to pick it up. Scale the indexers way up to populate cache quickly. Healthcheck them all, and notify the search team.",
                difficulty="hard",
                required_slots=[
                    "deployment_execution",
                    "restart_execution",
                    "scaling_execution",
                    "scaling_verification",
                    "deployment_notification"
                ],
                baseline_call_count=5 # deploy, restart, scale, healthcheck, notify
            ),
        ]
    },
    {
        "list_id": "hard-compliance-audit",
        "description": "Complex operational tasks with compliance and audit requirements",
        "tasks": [
            Task(
                id="hard-pci-compliance-audit",
                prompt="A PCI-DSS patch is mandated immediately across all nodes. Deploy the new secure proxy image, scale the old nodes to 0, restart the datastores to refresh keys, run healthchecks, and notify compliance.",
                difficulty="hard",
                required_slots=[
                    "deployment_execution",
                    "scaling_execution",
                    "restart_execution",
                    "restart_verification",
                    "deployment_notification"
                ],
                baseline_call_count=5 # deploy, scale, restart, healthcheck, notify
            ),
            Task(
                id="hard-security-incident-response",
                prompt="Active intrusion detected. Isolate immediately: scale external gateways to zero, rollback the compromised control plane, restart internal validators, check internal health, deploy clean patch, and send urgent notification.",
                difficulty="hard",
                required_slots=[
                    "scaling_execution",
                    "rollback_execution",
                    "restart_execution",
                    "restart_verification",
                    "deployment_execution",
                    "deployment_notification"
                ],
                baseline_call_count=6 # scale, rollback, restart, healthcheck, deploy, notify
            ),
        ]
    },
]

# ======================== ALL TASKS BY DIFFICULTY ========================

TASKS_BY_DIFFICULTY = {
    "easy": EASY_TASKS,
    "medium": MEDIUM_TASKS,
    "hard": HARD_TASKS,
}
