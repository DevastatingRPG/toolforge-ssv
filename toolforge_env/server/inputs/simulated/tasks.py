from models import Task

TASKS = [
    {
        "task_id": "easy-deployment-sprints",
        "description": "Repetitive service rollout and validation cycles",
        "tasks": [
            Task(id="e-dep-1", prompt="Deploy 'web-app' v1.1, check health, and notify #ops-chat.", difficulty="easy", required_slots=["deployment_execution", "deployment_verification", "deployment_notification"], baseline_call_count=3),
            Task(id="e-dep-2", prompt="Deploy 'api-v2' v2.0.4, verify health, and notify #dev-updates.", difficulty="easy", required_slots=["deployment_execution", "deployment_verification", "deployment_notification"], baseline_call_count=3),
            Task(id="e-dep-3", prompt="Deploy 'auth-svc' v3.1, run a healthcheck, and message #security.", difficulty="easy", required_slots=["deployment_execution", "deployment_verification", "deployment_notification"], baseline_call_count=3),
            Task(id="e-dep-4", prompt="Deploy 'search-ui' v1.5, check health status, and notify #product.", difficulty="easy", required_slots=["deployment_execution", "deployment_verification", "deployment_notification"], baseline_call_count=3),
            Task(id="e-dep-5", prompt="Deploy 'cart-logic' v2.2, confirm health, and alert #on-call.", difficulty="easy", required_slots=["deployment_execution", "deployment_verification", "deployment_notification"], baseline_call_count=3),
            Task(id="e-dep-6", prompt="Deploy 'notification-service' v4.0, run validation checks, and post to #releases.", difficulty="easy", required_slots=["deployment_execution", "deployment_verification", "deployment_notification"], baseline_call_count=3),
            Task(id="e-dep-7", prompt="Deploy 'image-processor' v1.0.1, check its health, and notify #media-team.", difficulty="easy", required_slots=["deployment_execution", "deployment_verification", "deployment_notification"], baseline_call_count=3),
        ]
    },
    {
        "task_id": "easy-resource-management",
        "description": "Standard scaling and restart procedures",
        "tasks": [
            Task(id="e-res-1", prompt="Scale 'worker-pool' to 5, check health, and notify #infra.", difficulty="easy", required_slots=["scaling_execution", "scaling_verification", "scaling_notification"], baseline_call_count=3),
            Task(id="e-res-2", prompt="Restart 'redis-cache', verify it is up, and notify #db-team.", difficulty="easy", required_slots=["restart_execution", "restart_verification", "restart_notification"], baseline_call_count=3),
            Task(id="e-res-3", prompt="Scale 'ingress-node' to 10, check health, and notify #traffic-ops.", difficulty="easy", required_slots=["scaling_execution", "scaling_verification", "scaling_notification"], baseline_call_count=3),
            Task(id="e-res-4", prompt="Restart 'logging-pod', check health, and notify #monitoring.", difficulty="easy", required_slots=["restart_execution", "restart_verification", "restart_notification"], baseline_call_count=3),
            Task(id="e-res-5", prompt="Scale 'pdf-generator' to 2, check health, and notify #backend-devs.", difficulty="easy", required_slots=["scaling_execution", "scaling_verification", "scaling_notification"], baseline_call_count=3),
            Task(id="e-res-6", prompt="Restart 'websocket-server', run a healthcheck, and alert #realtime-ops.", difficulty="easy", required_slots=["restart_execution", "restart_verification", "restart_notification"], baseline_call_count=3),
            Task(id="e-res-7", prompt="Scale 'metrics-aggregator' to 4, verify health, and post to #platform.", difficulty="easy", required_slots=["scaling_execution", "scaling_verification", "scaling_notification"], baseline_call_count=3),
        ]
    },
    {
        "task_id": "easy-rollback-drills",
        "description": "Repetitive rollback patterns for quick reversion practice",
        "tasks": [
            Task(id="e-roll-1", prompt="Rollback 'user-profile' service to previous version, verify health, and notify #ops-alerts.", difficulty="easy", required_slots=["rollback_execution", "rollback_verification", "rollback_notification"], baseline_call_count=3),
            Task(id="e-roll-2", prompt="Undo the latest deployment for 'checkout-api', check health, and inform #sales-tech.", difficulty="easy", required_slots=["rollback_execution", "rollback_verification", "rollback_notification"], baseline_call_count=3),
            Task(id="e-roll-3", prompt="Rollback 'inventory-db-proxy', run verification, and ping #database-admins.", difficulty="easy", required_slots=["rollback_execution", "rollback_verification", "rollback_notification"], baseline_call_count=3),
            Task(id="e-roll-4", prompt="Revert 'shipping-calculator' to stable, check system health, and notify #logistics.", difficulty="easy", required_slots=["rollback_execution", "rollback_verification", "rollback_notification"], baseline_call_count=3),
            Task(id="e-roll-5", prompt="Rollback 'oauth-provider', verify it is stable, and alert #security-logs.", difficulty="easy", required_slots=["rollback_execution", "rollback_verification", "rollback_notification"], baseline_call_count=3),
            Task(id="e-roll-6", prompt="Revert 'marketing-landing-page', check health, and notify #growth-team.", difficulty="easy", required_slots=["rollback_execution", "rollback_verification", "rollback_notification"], baseline_call_count=3),
        ]
    },
    {
        "task_id": "medium-traffic-readiness",
        "description": "Preparing services for specific traffic goals",
        "tasks": [
            Task(id="m-tr-1", prompt="Prepare for the 9AM traffic spike: Scale 'gateway-api' to 15 replicas and confirm stability.", difficulty="medium", required_slots=["scaling_execution", "scaling_verification"], baseline_call_count=2),
            Task(id="m-tr-2", prompt="The 'search-engine' is lagging. Deploy the latest performance patch v4.2 and restart all search-workers.", difficulty="medium", required_slots=["deployment_execution", "restart_execution", "deployment_verification"], baseline_call_count=3),
            Task(id="m-tr-3", prompt="Optimize cloud spend: Scale down 'dev-environment' clusters to 1 replica each and notify #finance.", difficulty="medium", required_slots=["scaling_execution", "scaling_verification", "scaling_notification"], baseline_call_count=3),
            Task(id="m-tr-4", prompt="Rollout the 'dark-mode' feature to the 'frontend-canary'. Check health; if it fails, rollback immediately.", difficulty="medium", required_slots=["deployment_execution", "deployment_verification", "rollback_execution"], baseline_call_count=3),
            Task(id="m-tr-5", prompt="A memory leak was found in 'payment-v3'. Deploy 'v3.1-fix', verify it, and notify #on-call-emergency.", difficulty="medium", required_slots=["deployment_execution", "deployment_verification", "deployment_notification"], baseline_call_count=3),
            Task(id="m-tr-6", prompt="Marketing just launched a surprise campaign. Rapidly scale 'landing-page-ui' to 20, check health, and alert #marketing-ops.", difficulty="medium", required_slots=["scaling_execution", "scaling_verification", "scaling_notification"], baseline_call_count=3),
            Task(id="m-tr-7", prompt="Prepare for nightly batch processing by scaling 'data-cruncher' to 10. Once verified, notify #data-engineering.", difficulty="medium", required_slots=["scaling_execution", "scaling_verification", "scaling_notification"], baseline_call_count=3),
        ]
    },
    {
        "task_id": "medium-incident-response",
        "description": "Goal-driven mitigation of active system anomalies",
        "tasks": [
            Task(id="m-inc-1", prompt="Users are getting 502s from 'api-gateway'. Restart the gateway nodes, check if health improves, and notify #incident-room.", difficulty="medium", required_slots=["restart_execution", "restart_verification", "restart_notification"], baseline_call_count=3),
            Task(id="m-inc-2", prompt="The newly deployed 'recommendation-engine' is crashing. Roll it back, restart the dependent 'ui-dashboard', and verify.", difficulty="medium", required_slots=["rollback_execution", "restart_execution", "restart_verification"], baseline_call_count=3),
            Task(id="m-inc-3", prompt="'worker-queue' is deadlocked. Scale it to 0 to flush, scale back to 5, verify health, and notify #backend.", difficulty="medium", required_slots=["scaling_execution", "scaling_verification", "scaling_notification"], baseline_call_count=4),
            Task(id="m-inc-4", prompt="DDoS mitigation: Deploy the 'strict-rate-limit' config to 'edge-proxy', verify the deployment, and alert #security.", difficulty="medium", required_slots=["deployment_execution", "deployment_verification", "deployment_notification"], baseline_call_count=3),
            Task(id="m-inc-5", prompt="Database connections are maxed out. Restart 'connection-pooler', check health. If it fails, scale down 'background-jobs' to reduce load.", difficulty="medium", required_slots=["restart_execution", "restart_verification", "scaling_execution"], baseline_call_count=3),
            Task(id="m-inc-6", prompt="'cache-layer' is serving stale data. Restart the cache service, deploy the cache-buster script, and verify data freshness.", difficulty="medium", required_slots=["restart_execution", "deployment_execution", "deployment_verification"], baseline_call_count=3),
            Task(id="m-inc-7", prompt="Payment timeout alerts triggered. Rollback the last update to 'stripe-connector', check health, and update #finance-alerts.", difficulty="medium", required_slots=["rollback_execution", "rollback_verification", "rollback_notification"], baseline_call_count=3),
        ]
    },
    {
        "task_id": "hard-project-legacy-migration",
        "description": "Project: Decommissioning Legacy Data Center to Cloud",
        "tasks": [
            Task(id="h-mig-1", prompt="Phase 1: Deploy 'cloud-connector-v1' and scale it to 20 replicas to handle initial data sync. Verify the connection health and notify #migration-hq.", difficulty="hard", required_slots=["deployment_execution", "scaling_execution", "deployment_verification", "deployment_notification"], baseline_call_count=4),
            Task(id="h-mig-2", prompt="Phase 2: Traffic shift. Scale down 'legacy-monolith' by 50%. Restart the 'global-load-balancer' to route traffic to the cloud nodes. Verify health.", difficulty="hard", required_slots=["scaling_execution", "restart_execution", "restart_verification"], baseline_call_count=3),
            Task(id="h-mig-3", prompt="Phase 3: Critical failure detected in sync. Rollback 'cloud-connector' to v0.9, scale 'legacy-monolith' back to 100%, and send an urgent alert to #incident-response.", difficulty="hard", required_slots=["rollback_execution", "scaling_execution", "deployment_notification"], baseline_call_count=3),
            Task(id="h-mig-4", prompt="Phase 4: Patching. Deploy the 'hotfix-v1.1' for cloud sync. Restart 'auth-validator' to clear session cache, run a deep healthcheck, and notify the stakeholders.", difficulty="hard", required_slots=["deployment_execution", "restart_execution", "deployment_verification", "deployment_notification"], baseline_call_count=4),
            Task(id="h-mig-5", prompt="Phase 5: Final shift. Scale 'legacy-monolith' to 10%. Scale 'cloud-services' to maximum capacity. Verify stability across all regions.", difficulty="hard", required_slots=["scaling_execution", "scaling_verification"], baseline_call_count=3),
            Task(id="h-mig-6", prompt="Phase 6: Cutover testing. Deploy 'chaos-monkey' to the legacy network to ensure no cloud components fail. Verify health, then rollback chaos monkey.", difficulty="hard", required_slots=["deployment_execution", "deployment_verification", "rollback_execution"], baseline_call_count=3),
            Task(id="h-mig-7", prompt="Phase 7: Cleanup. Decommission legacy by scaling 'legacy-monolith' to 0. Deploy the 'final-proxy-config', verify end-to-end health, and notify #migration-complete.", difficulty="hard", required_slots=["scaling_execution", "deployment_execution", "deployment_verification", "deployment_notification"], baseline_call_count=4),
        ]
    },
    {
        "task_id": "hard-project-zero-trust",
        "description": "Project: Zero-Trust Security Mesh Implementation",
        "tasks": [
            Task(id="h-sec-1", prompt="Phase 1: Deploy 'envoy-sidecar-injector' to the cluster. Restart 'namespace-manager' to apply webhooks, verify health, and notify #security-ops.", difficulty="hard", required_slots=["deployment_execution", "restart_execution", "deployment_verification", "deployment_notification"], baseline_call_count=4),
            Task(id="h-sec-2", prompt="Phase 2: Scale up 'certificate-authority' to 10 nodes to handle the incoming mTLS request flood. Verify nodes are healthy before proceeding.", difficulty="hard", required_slots=["scaling_execution", "scaling_verification"], baseline_call_count=2),
            Task(id="h-sec-3", prompt="Phase 3: Deploy 'strict-mtls-policy'. Immediately run healthchecks on 'billing-api'. If billing fails, rollback the policy and alert #sec-incidents.", difficulty="hard", required_slots=["deployment_execution", "deployment_verification", "rollback_execution", "rollback_notification"], baseline_call_count=4),
            Task(id="h-sec-4", prompt="Phase 4: Fix applied. Redeploy 'strict-mtls-policy-v2'. Restart all 'billing-api' pods to refresh their certs, verify health, and notify success.", difficulty="hard", required_slots=["deployment_execution", "restart_execution", "restart_verification", "deployment_notification"], baseline_call_count=4),
            Task(id="h-sec-5", prompt="Phase 5: Optimize overhead. Scale down 'legacy-vpn-gateway' from 15 to 2. Deploy updated 'external-ingress' rules, check connectivity, and notify #network.", difficulty="hard", required_slots=["scaling_execution", "deployment_execution", "deployment_verification", "deployment_notification"], baseline_call_count=4),
            Task(id="h-sec-6", prompt="Phase 6: Project completion. Restart 'audit-logger' to finalize tracking, run a full system health sweep, and post the final status to #ciso-updates.", difficulty="hard", required_slots=["restart_execution", "restart_verification", "restart_notification"], baseline_call_count=3),
        ]
    }
]
