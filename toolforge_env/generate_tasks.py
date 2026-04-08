import random

# Seed for reproducibility
random.seed(42)

def generate_easy_deployment_sprints(count=15):
    services = ['web-app', 'api-v2', 'auth-svc', 'search-ui', 'cart-logic', 'notification-service', 'image-processor', 'payment-gateway', 'user-profile', 'recommendation-engine', 'inventory-db-proxy', 'shipping-calculator', 'oauth-provider', 'billing-api', 'frontend-canary']
    channels = ['#ops-chat', '#dev-updates', '#security', '#product', '#on-call', '#releases', '#media-team', '#finance', '#support']
    verbs = ['Deploy', 'Push out', 'Release', 'Roll out']
    verify = ['check health', 'verify health', 'run a healthcheck', 'check health status', 'confirm health', 'run validation checks', 'ping the endpoint', 'run tests']
    
    tasks = []
    for i in range(count):
        service = random.choice(services)
        version = f"v{random.randint(1,4)}.{random.randint(0,9)}.{random.randint(0,5)}"
        cmd = f"{random.choice(verbs)} '{service}' {version}, {random.choice(verify)}, and notify {random.choice(channels)}."
        
        tasks.append({
            "id": f"e-dep-{i+1}",
            "prompt": cmd,
            "difficulty": "easy",
            "required_slots": ["deployment_execution", "deployment_verification", "deployment_notification"],
            "baseline_call_count": 3
        })
    return tasks

def generate_easy_resource_management(count=15):
    services = ['worker-pool', 'redis-cache', 'ingress-node', 'logging-pod', 'pdf-generator', 'websocket-server', 'metrics-aggregator', 'data-cruncher', 'cache-layer', 'ui-dashboard', 'background-jobs', 'message-queue', 'session-store']
    channels = ['#infra', '#db-team', '#traffic-ops', '#monitoring', '#backend-devs', '#realtime-ops', '#platform']
    verify = ['check health', 'verify it is up', 'run a healthcheck', 'verify health', 'ping it to ensure connectivity']
    
    tasks = []
    for i in range(count):
        service = random.choice(services)
        if random.random() < 0.5:
            # Scale
            replicas = random.randint(2, 20)
            cmd = f"Scale '{service}' to {replicas}, {random.choice(verify)}, and notify {random.choice(channels)}."
            slots = ["scaling_execution", "scaling_verification", "scaling_notification"]
        else:
            # Restart
            cmd = f"Restart '{service}', {random.choice(verify)}, and notify {random.choice(channels)}."
            slots = ["restart_execution", "restart_verification", "restart_notification"]
            
        tasks.append({
            "id": f"e-res-{i+1}",
            "prompt": cmd,
            "difficulty": "easy",
            "required_slots": slots,
            "baseline_call_count": 3
        })
    return tasks

def generate_easy_rollback_drills(count=15):
    services = ['user-profile', 'checkout-api', 'inventory-db-proxy', 'shipping-calculator', 'oauth-provider', 'marketing-landing-page', 'search-engine', 'payment-v3', 'stripe-connector', 'legacy-monolith', 'auth-validator']
    channels = ['#ops-alerts', '#sales-tech', '#database-admins', '#logistics', '#security-logs', '#growth-team', '#reverting']
    verbs = ['Rollback', 'Undo the latest deployment for', 'Revert']
    verify = ['verify health', 'check health', 'run verification', 'check system health', 'verify it is stable', 'run core tests']
    
    tasks = []
    for i in range(count):
        service = random.choice(services)
        cmd = f"{random.choice(verbs)} '{service}', {random.choice(verify)}, and notify {random.choice(channels)}."
        tasks.append({
            "id": f"e-roll-{i+1}",
            "prompt": cmd,
            "difficulty": "easy",
            "required_slots": ["rollback_execution", "rollback_verification", "rollback_notification"],
            "baseline_call_count": 3
        })
    return tasks

def generate_medium_traffic_readiness(count=20):
    services = ['gateway-api', 'search-engine', 'dev-environment', 'frontend-canary', 'payment-v3', 'landing-page-ui', 'data-cruncher', 'cache-layer', 'edge-proxy', 'video-transcoder']
    
    templates = [
        ("Prepare for the {event}: Scale '{svc}' to {rep} replicas and confirm stability.", ["scaling_execution", "scaling_verification"], 2),
        ("The '{svc}' is lagging. Deploy the latest performance patch {ver} and restart all workers.", ["deployment_execution", "restart_execution"], 2),
        ("Optimize cloud spend: Scale down '{svc}' to 1 replica and notify #finance.", ["scaling_execution", "scaling_notification"], 2),
        ("Rollout feature-flag to '{svc}'. Check health; if it fails, rollback immediately.", ["deployment_execution", "deployment_verification", "rollback_execution"], 3),
        ("A memory leak was found in '{svc}'. Deploy '{ver}-fix', verify it, and notify #on-call-emergency.", ["deployment_execution", "deployment_verification", "deployment_notification"], 3),
        ("Marketing just launched a surprise campaign. Rapidly scale '{svc}' to {rep}, check health, and alert #marketing-ops.", ["scaling_execution", "scaling_verification", "scaling_notification"], 3),
        ("Nightly batch processing started: scale '{svc}' to {rep}. Once verified, notify #data-engineering.", ["scaling_execution", "scaling_verification", "scaling_notification"], 3),
        ("Traffic is dropping, downscale '{svc}' back to {rep} instances. Send a pagerduty alert to #billing just in case.", ["scaling_execution", "scaling_notification"], 2),
        ("We are migrating traffic. Apply patch {ver} to '{svc}', scale it to {rep}, and run extensive tests.", ["deployment_execution", "scaling_execution", "deployment_verification"], 3),
        ("Readiness check: Restart '{svc}', and trigger a pagerduty alert to #ops-readiness about the bounce.", ["restart_execution", "restart_notification"], 2)
    ]
    
    events = ['9AM traffic spike', 'Black Friday', 'New Year Launch', 'Flash Sale']
    
    tasks = []
    for i in range(count):
        tmpl, slots, bl = random.choice(templates)
        svc = random.choice(services)
        rep = random.randint(10, 50)
        ver = f"v{random.randint(1,5)}.{random.randint(0,4)}"
        event = random.choice(events)
        
        prompt = tmpl.format(svc=svc, rep=rep, ver=ver, event=event)
        
        tasks.append({
            "id": f"m-tr-{i+1}",
            "prompt": prompt,
            "difficulty": "medium",
            "required_slots": slots,
            "baseline_call_count": bl
        })
    return tasks

def generate_medium_incident_response(count=20):
    services = ['api-gateway', 'recommendation-engine', 'worker-queue', 'edge-proxy', 'connection-pooler', 'cache-layer', 'stripe-connector', 'auth-validator', 'session-store', 'checkout-ui']
    
    templates = [
        ("Users are getting 502s from '{svc}'. Restart the nodes, check if health improves, and notify #incident-room.", ["restart_execution", "restart_verification", "restart_notification"], 3),
        ("The newly deployed '{svc}' is crashing. Roll it back, restart the dependent 'ui-dashboard', and verify.", ["rollback_execution", "restart_execution", "restart_verification"], 3),
        ("'{svc}' is deadlocked. Scale it to 0 to flush, scale back to 5, verify health, and notify #backend.", ["scaling_execution", "scaling_verification", "scaling_notification"], 4),
        ("DDoS mitigation: Deploy the 'strict-rate-limit' config to '{svc}', verify the deployment, and alert #security.", ["deployment_execution", "deployment_verification", "deployment_notification"], 3),
        ("Database connections are maxed out. Restart '{svc}', check health. Then scale down 'background-jobs' to reduce load.", ["restart_execution", "restart_verification", "scaling_execution"], 3),
        ("'{svc}' is serving stale data. Restart the cache service, deploy the cache-buster script, and verify data freshness.", ["restart_execution", "deployment_execution", "deployment_verification"], 3),
        ("Timeout alerts triggered. Rollback the last update to '{svc}', check health, and update #finance-alerts.", ["rollback_execution", "rollback_verification", "rollback_notification"], 3),
        ("Critical vulnerability explicitly found in '{svc}'. Apply hotfix patch {ver}, restart the service, and trigger pagerduty alert.", ["deployment_execution", "restart_execution", "deployment_notification"], 3),
        ("Disk space full on '{svc}'. Scale it down to 1 node, rollback the latest offending deploy, and run tests.", ["scaling_execution", "rollback_execution", "rollback_verification"], 3),
        ("'{svc}' is completely unresponsive. Ping the endpoint to double check, then restart it aggressively and page #on-call.", ["restart_verification", "restart_execution", "restart_notification"], 3)
    ]
    
    tasks = []
    for i in range(count):
        tmpl, slots, bl = random.choice(templates)
        svc = random.choice(services)
        ver = f"v{random.randint(6,9)}.0"
        
        prompt = tmpl.format(svc=svc, ver=ver)
        tasks.append({
            "id": f"m-inc-{i+1}",
            "prompt": prompt,
            "difficulty": "medium",
            "required_slots": slots,
            "baseline_call_count": bl
        })
    return tasks

def generate_medium_maintenance_operations(count=20):
    services = ['legacy-monolith', 'log-aggregator', 'metrics-exporter', 'sso-router', 'vpn-gateway', 'internal-admin-tool', 'report-generator']
    
    templates = [
        ("Routine maintenance: Patch the '{svc}' to {ver}. Verify it's healthy, then restart the 'proxy-layer'.", ["deployment_execution", "deployment_verification", "restart_execution"], 3),
        ("Off-peak cleanup: Rollback the temporary '{svc}' deployment, scale down the background workers, and notify #ops-chat.", ["rollback_execution", "scaling_execution", "rollback_notification"], 3),
        ("Cert rotation: Deploy new certs to '{svc}', restart the service, run tests.", ["deployment_execution", "restart_execution", "deployment_verification"], 3),
        ("Cost optimization sweep: Scale '{svc}' down to 2 replicas. Ping it to make sure it handles the baseline load, and send an alert.", ["scaling_execution", "scaling_verification", "scaling_notification"], 3),
        ("Dependency update: Apply patch for log4j in '{svc}', run full test suite, and notify #security.", ["deployment_execution", "deployment_verification", "deployment_notification"], 3),
        ("Scheduled node bounce: Restart '{svc}', and pagerduty alert the infra folks about the active maintenance window.", ["restart_execution", "restart_notification"], 2),
        ("Audit compliance: Rollback '{svc}' to standard configuration, check health, and scale 'audit-logger' to 5.", ["rollback_execution", "rollback_verification", "scaling_execution"], 3),
        ("Data balancing: Scale up '{svc}' to {rep} while data is shuffling, deploy the monitoring agent {ver}, and verify.", ["scaling_execution", "deployment_execution", "deployment_verification"], 3)
    ]
    
    tasks = []
    for i in range(count):
        tmpl, slots, bl = random.choice(templates)
        svc = random.choice(services)
        ver = f"v{random.randint(1,2)}.{random.randint(0,4)}"
        rep = random.randint(10, 20)
        
        prompt = tmpl.format(svc=svc, ver=ver, rep=rep)
        tasks.append({
            "id": f"m-maint-{i+1}",
            "prompt": prompt,
            "difficulty": "medium",
            "required_slots": slots,
            "baseline_call_count": bl
        })
    return tasks


# --- Legacy Hard Tasks ---
hard_tasks = [
    {
        "task_id": "hard-project-legacy-migration",
        "description": "Project: Decommissioning Legacy Data Center to Cloud",
        "tasks": [
            {
                "id": "h-mig-1", 
                "prompt": "Phase 1: Deploy 'cloud-connector-v1' and scale it to 20 replicas to handle initial data sync. Verify the connection health and notify #migration-hq.", 
                "difficulty": "hard", 
                "required_slots": ["deployment_execution", "scaling_execution", "deployment_verification", "deployment_notification"], 
                "baseline_call_count": 4
            },
            {
                "id": "h-mig-2", 
                "prompt": "Phase 2: Traffic shift. Scale down 'legacy-monolith' by 50%. Restart the 'global-load-balancer' to route traffic to the cloud nodes. Verify health.", 
                "difficulty": "hard", 
                "required_slots": ["scaling_execution", "restart_execution", "restart_verification"], 
                "baseline_call_count": 3
            },
            {
                "id": "h-mig-3", 
                "prompt": "Phase 3: Critical failure detected in sync. Rollback 'cloud-connector' to v0.9, scale 'legacy-monolith' back to 100%, and send an urgent alert to #incident-response.", 
                "difficulty": "hard", 
                "required_slots": ["rollback_execution", "scaling_execution", "deployment_notification"], 
                "baseline_call_count": 3
            },
            {
                "id": "h-mig-4", 
                "prompt": "Phase 4: Patching. Deploy the 'hotfix-v1.1' for cloud sync. Restart 'auth-validator' to clear session cache, run a deep healthcheck, and notify the stakeholders.", 
                "difficulty": "hard", 
                "required_slots": ["deployment_execution", "restart_execution", "deployment_verification", "deployment_notification"], 
                "baseline_call_count": 4
            },
            {
                "id": "h-mig-5", 
                "prompt": "Phase 5: Final shift. Scale 'legacy-monolith' to 10%. Scale 'cloud-services' to maximum capacity. Verify stability across all regions.", 
                "difficulty": "hard", 
                "required_slots": ["scaling_execution", "scaling_verification"], 
                "baseline_call_count": 3
            },
            {
                "id": "h-mig-6", 
                "prompt": "Phase 6: Cutover testing. Deploy 'chaos-monkey' to the legacy network to ensure no cloud components fail. Verify health, then rollback chaos monkey.", 
                "difficulty": "hard", 
                "required_slots": ["deployment_execution", "deployment_verification", "rollback_execution"], 
                "baseline_call_count": 3
            },
            {
                "id": "h-mig-7", 
                "prompt": "Phase 7: Cleanup. Decommission legacy by scaling 'legacy-monolith' to 0. Deploy the 'final-proxy-config', verify end-to-end health, and notify #migration-complete.", 
                "difficulty": "hard", 
                "required_slots": ["scaling_execution", "deployment_execution", "deployment_verification", "deployment_notification"], 
                "baseline_call_count": 4
            },
        ]
    },
    {
        "task_id": "hard-project-zero-trust",
        "description": "Project: Zero-Trust Security Mesh Implementation",
        "tasks": [
            {
                "id": "h-sec-1", 
                "prompt": "Phase 1: Deploy 'envoy-sidecar-injector' to the cluster. Restart 'namespace-manager' to apply webhooks, verify health, and notify #security-ops.", 
                "difficulty": "hard", 
                "required_slots": ["deployment_execution", "restart_execution", "deployment_verification", "deployment_notification"], 
                "baseline_call_count": 4
            },
            {
                "id": "h-sec-2", 
                "prompt": "Phase 2: Scale up 'certificate-authority' to 10 nodes to handle the incoming mTLS request flood. Verify nodes are healthy before proceeding.", 
                "difficulty": "hard", 
                "required_slots": ["scaling_execution", "scaling_verification"], 
                "baseline_call_count": 2
            },
            {
                "id": "h-sec-3", 
                "prompt": "Phase 3: Deploy 'strict-mtls-policy'. Immediately run healthchecks on 'billing-api'. If billing fails, rollback the policy and alert #sec-incidents.", 
                "difficulty": "hard", 
                "required_slots": ["deployment_execution", "deployment_verification", "rollback_execution", "rollback_notification"], 
                "baseline_call_count": 4
            },
            {
                "id": "h-sec-4", 
                "prompt": "Phase 4: Fix applied. Redeploy 'strict-mtls-policy-v2'. Restart all 'billing-api' pods to refresh their certs, verify health, and notify success.", 
                "difficulty": "hard", 
                "required_slots": ["deployment_execution", "restart_execution", "restart_verification", "deployment_notification"], 
                "baseline_call_count": 4
            },
            {
                "id": "h-sec-5", 
                "prompt": "Phase 5: Optimize overhead. Scale down 'legacy-vpn-gateway' from 15 to 2. Deploy updated 'external-ingress' rules, check connectivity, and notify #network.", 
                "difficulty": "hard", 
                "required_slots": ["scaling_execution", "deployment_execution", "deployment_verification", "deployment_notification"], 
                "baseline_call_count": 4
            },
            {
                "id": "h-sec-6", 
                "prompt": "Phase 6: Project completion. Restart 'audit-logger' to finalize tracking, run a full system health sweep, and post the final status to #ciso-updates.", 
                "difficulty": "hard", 
                "required_slots": ["restart_execution", "restart_verification", "restart_notification"], 
                "baseline_call_count": 3
            },
        ]
    }
]

import json

def build_tasks_module():
    all_groups = [
        {"task_id": "easy-deployment-sprints", "description": "Repetitive service rollout and validation cycles", "tasks": generate_easy_deployment_sprints()},
        {"task_id": "easy-resource-management", "description": "Standard scaling and restart procedures", "tasks": generate_easy_resource_management()},
        {"task_id": "easy-rollback-drills", "description": "Repetitive rollback patterns for quick reversion practice", "tasks": generate_easy_rollback_drills()},
        {"task_id": "medium-traffic-readiness", "description": "Preparing services for specific traffic goals", "tasks": generate_medium_traffic_readiness()},
        {"task_id": "medium-incident-response", "description": "Goal-driven mitigation of active system anomalies", "tasks": generate_medium_incident_response()},
        {"task_id": "medium-maintenance-operations", "description": "Routine and scheduled maintenance operations", "tasks": generate_medium_maintenance_operations()},
    ] + hard_tasks

    lines = ["from models import Task\n\nTASKS = ["]
    for g in all_groups:
        lines.append('    {')
        lines.append(f'        "task_id": "{g["task_id"]}",')
        lines.append(f'        "description": "{g["description"]}",')
        lines.append('        "tasks": [')
        for t in g["tasks"]:
            lines.append(f'            Task(id="{t["id"]}", prompt="{t["prompt"]}", difficulty="{t["difficulty"]}", required_slots={json.dumps(t["required_slots"])}, baseline_call_count={t["baseline_call_count"]}),')
        lines.append('        ]')
        lines.append('    },')
    lines.append("]\n")

    with open("tasks.py", "w") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    build_tasks_module()
