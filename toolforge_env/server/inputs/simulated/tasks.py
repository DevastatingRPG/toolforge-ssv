from models import Task

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
                required_steps=["deploy", "healthcheck", "notify"],
                core_steps=["deploy"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="easy-deploy-restart",
                prompt="Deploy 'backend-api' version 'v1.4.2', check its health. It usually needs a restart after deployment, so restart it, check health again, and then notify '#backend-ops' that the deploy is complete.",
                difficulty="easy",
                required_steps=["deploy", "healthcheck", "restart", "healthcheck", "notify"],
                core_steps=["deploy", "restart"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="easy-deploy-scale",
                prompt="Deploy 'analytics-engine' version 'v3.0.0', then quickly scale it to 10 replicas for the incoming load test. Check health to verify and notify '#data-team'.",
                difficulty="easy",
                required_steps=["deploy", "scale", "healthcheck", "notify"],
                core_steps=["deploy", "scale"],
                required_slots=[],
                baseline_token_cost=10
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
                required_steps=["scale", "healthcheck", "notify"],
                core_steps=["scale"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="easy-scale-down",
                prompt="Scale down the 'batch-processor' service from 8 to 3 replicas to save costs during off-peak hours. Verify health and notify '#cost-ops' when complete.",
                difficulty="easy",
                required_steps=["scale", "healthcheck", "notify"],
                core_steps=["scale"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="easy-scale-verify",
                prompt="Scale 'user-service' to 4 replicas and perform a thorough health check on all replicas to ensure they are fully ready.",
                difficulty="easy",
                required_steps=["scale", "healthcheck"],
                core_steps=["scale"],
                required_slots=[],
                baseline_token_cost=10
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
                required_steps=["restart", "healthcheck", "notify"],
                core_steps=["restart"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="easy-rollback-notify",
                prompt="The recent deployment of 'auth-service' caused a spike in errors. Rollback the deployment, check the health, and notify '#on-call' that the service has been rolled back.",
                difficulty="easy",
                required_steps=["rollback", "healthcheck", "notify"],
                core_steps=["rollback"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="easy-restart-verify",
                prompt="Restart 'payment-service' and verify it comes back healthy with all its dependent connections restored.",
                difficulty="easy",
                required_steps=["restart", "healthcheck"],
                core_steps=["restart"],
                required_slots=[],
                baseline_token_cost=10
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
                prompt="Deploy 'data-pipeline' v2.0.0, then deploy 'analytics-ui' v1.8.0 which depends on it. Verify each with health checks, synchronize schemas between them, and notify '#data-platform' when both are stable.",
                difficulty="medium",
                required_steps=["deploy", "healthcheck", "deploy", "healthcheck", "sync", "notify"],
                core_steps=["deploy", "deploy", "sync"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="medium-canary-rollout",
                prompt="Perform a canary deployment of 'recommendations-engine' v4.1.0 by deploying to 2 replicas first, monitoring metrics for 2 minutes, then gradually scaling to 8 replicas if metrics look good.",
                difficulty="medium",
                required_steps=["deploy", "scale", "monitor", "scale", "healthcheck"],
                core_steps=["deploy", "scale", "monitor"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="medium-blue-green-prep",
                prompt="Prepare 'api-gateway' v3.0.0 in a blue-green setup: deploy to green environment, run integration tests, switch traffic, then verify old blue environment can be cleaned.",
                difficulty="medium",
                required_steps=["deploy", "test", "switch-traffic", "healthcheck", "cleanup"],
                core_steps=["deploy", "test", "switch-traffic"],
                required_slots=[],
                baseline_token_cost=10
            ),
        ]
    },
    {
        "list_id": "medium-troubleshooting-recovery",
        "description": "Diagnosis and recovery from failures with multiple steps",
        "tasks": [
            Task(
                id="medium-partial-failure-recovery",
                prompt="Three replicas of 'worker-pool' are down. Investigate which ones are unhealthy, restart them selectively, scale up if needed to maintain capacity, then validate the service is fully recovered.",
                difficulty="medium",
                required_steps=["healthcheck", "restart", "scale", "healthcheck", "validate"],
                core_steps=["restart", "validate"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="medium-cascade-failure",
                prompt="An error in 'service-a' is causing failures in dependent 'service-b' and 'service-c'. Identify the root cause, rollback 'service-a', verify its health and its dependents, then notify all affected teams.",
                difficulty="medium",
                required_steps=["diagnose", "rollback", "healthcheck", "verify-deps", "notify"],
                core_steps=["rollback", "verify-deps"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="medium-resource-constraint",
                prompt="The 'ml-trainer' service is consuming too many resources. Monitor its usage, first try restarting it with resource limits, if that fails scale down replicas and optimize configuration.",
                difficulty="medium",
                required_steps=["monitor", "restart", "configure", "healthcheck"],
                core_steps=["monitor", "configure"],
                required_slots=[],
                baseline_token_cost=10
            ),
        ]
    },
    {
        "list_id": "medium-maintenance-windows",
        "description": "Coordinated maintenance with minimal downtime",
        "tasks": [
            Task(
                id="medium-drain-and-upgrade",
                prompt="Gracefully drain connections from 'connection-pool' service, upgrade it to v2.5.1, restart with new config, verify all connections reconnect properly, and notify teams when complete.",
                difficulty="medium",
                required_steps=["drain", "deploy", "restart", "verify-connections", "notify"],
                core_steps=["drain", "deploy"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="medium-rolling-update",
                prompt="Perform a rolling update of 'web-frontend' from 6 to 8 replicas with version v5.2.0: update 2 at a time, verify each batch, ensure load balancer keeps routing correctly.",
                difficulty="medium",
                required_steps=["drain", "deploy", "verify", "scale"],
                core_steps=["deploy", "scale"],
                required_slots=[],
                baseline_token_cost=10
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
                prompt="Migrate 'user-database' schema to v2.0 with zero downtime: deploy new code that handles both old and new schemas, migrate data incrementally in background jobs, verify consistency across nodes, switch to new schema, then cleanup old schema.",
                difficulty="hard",
                required_steps=["deploy", "migrate", "validate", "healthcheck", "verify-consistency", "switch-schema", "cleanup"],
                core_steps=["migrate", "verify-consistency", "switch-schema"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="hard-cross-region-sync",
                prompt="Synchronize 'payment-ledger' across 3 regions maintaining referential integrity: verify current state in each region, identify conflicts, merge using version vectors, test in staging, then deploy to production with monitoring.",
                difficulty="hard",
                required_steps=["verify", "diagnose", "merge", "test", "deploy", "monitor"],
                core_steps=["merge", "deploy", "monitor"],
                required_slots=[],
                baseline_token_cost=10
            ),
        ]
    },
    {
        "list_id": "hard-disaster-recovery",
        "description": "Large-scale disaster recovery and resilience scenarios",
        "tasks": [
            Task(
                id="hard-full-region-failover",
                prompt="Entire 'us-west' region is down. Failover all services to 'us-east': verify data consistency, promote read-replicas, redirect DNS traffic, validate all service health across 15 services, monitor for anomalies, and coordinate communication with 5 different teams.",
                difficulty="hard",
                required_steps=["verify-data", "promote", "dns-switch", "healthcheck-all", "monitor", "notify-teams", "validate-traffic"],
                core_steps=["promote", "dns-switch", "validate-traffic"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="hard-cascade-recovery",
                prompt="Cascading failure: load balancer down → API servers failing → database connections exhausted → cache eviction. Recover in correct order: restore load balancer, restore API health, drain connection pool, restore cache, verify end-to-end transactions.",
                difficulty="hard",
                required_steps=["restore-lb", "healthcheck-api", "drain-connections", "restore-cache", "e2e-test", "validate"],
                core_steps=["restore-lb", "drain-connections", "e2e-test"],
                required_slots=[],
                baseline_token_cost=10
            ),
        ]
    },
    {
        "list_id": "hard-optimization-refactor",
        "description": "Large-scale refactoring and optimization with complex validation",
        "tasks": [
            Task(
                id="hard-service-decomposition",
                prompt="Decompose monolithic 'admin-service' into 3 microservices while running both old and new in parallel: deploy new services, implement adapter layer in monolith, gradually shift traffic using feature flags, validate functionality, remove legacy code once fully transitioned.",
                difficulty="hard",
                required_steps=["deploy-services", "adapter-layer", "enable-flags", "shift-traffic", "validate-all", "cleanup"],
                core_steps=["deploy-services", "shift-traffic", "cleanup"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="hard-performance-optimization",
                prompt="Optimize 'search-engine' for 10x throughput: profile current bottlenecks, implement caching layer, refactor database queries, deploy incrementally to canary, compare metrics, scale if metrics good, verify no regressions in search quality.",
                difficulty="hard",
                required_steps=["profile", "cache-layer", "optimize-queries", "deploy-canary", "compare-metrics", "scale", "validate-quality"],
                core_steps=["cache-layer", "optimize-queries", "scale"],
                required_slots=[],
                baseline_token_cost=10
            ),
        ]
    },
    {
        "list_id": "hard-compliance-audit",
        "description": "Complex operational tasks with compliance and audit requirements",
        "tasks": [
            Task(
                id="hard-pci-compliance-audit",
                prompt="Audit all systems for PCI-DSS compliance: scan 25 services for secrets, rotate credentials, enable encryption at rest and in transit, validate audit logs, generate compliance report, remediate any findings, and schedule follow-up validation.",
                difficulty="hard",
                required_steps=["scan-secrets", "rotate-creds", "enable-encryption", "validate-logs", "generate-report", "remediate", "schedule-followup"],
                core_steps=["rotate-creds", "enable-encryption", "remediate"],
                required_slots=[],
                baseline_token_cost=10
            ),
            Task(
                id="hard-security-incident-response",
                prompt="Respond to security incident: detect compromised 'user-service', isolate it from network, preserve forensic logs, identify blast radius (affected users/data), contact security team, patch vulnerability, reimage server, restore from clean backup, validate and communicate with users.",
                difficulty="hard",
                required_steps=["detect", "isolate", "preserve-logs", "identify-blast-radius", "contact-security", "patch", "reimage", "restore", "validate", "communicate"],
                core_steps=["isolate", "patch", "restore"],
                required_slots=[],
                baseline_token_cost=10
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
