import logging
from toolforge_env.models import Task

logger = logging.getLogger(__name__)

# Define Easy Tasks
_EASY_TASKS = [
    Task(
        id="easy-deploy-notify",
        prompt="Deploy the 'frontend-web' service version 'v2.1.0', check its health, and notify the '#deployments' channel that it is done.",
        difficulty="easy",
        required_steps=["deploy", "healthcheck", "notify"],
        core_steps=["deploy"]
    ),
    Task(
        id="easy-deploy-restart",
        prompt="Deploy 'backend-api' version 'v1.4.2', check its health. It usually needs a restart after deployment, so restart it, check health again, and then notify '#backend-ops' that the deploy is complete.",
        difficulty="easy",
        required_steps=["deploy", "healthcheck", "restart", "healthcheck", "notify"],
        core_steps=["deploy", "restart"]
    ),
    Task(
        id="easy-scale-notify",
        prompt="Scale the 'queue-worker' service to 5 replicas, check its health to ensure all are up, and notify '#ops-alerts' with the updated scaling status.",
        difficulty="easy",
        required_steps=["scale", "healthcheck", "notify"],
        core_steps=["scale"]
    ),
    Task(
        id="easy-restart-notify",
        prompt="Restart the 'cache-redis' service because it has been acting up. Check its health afterwards and notify '#db-admins' that the bounce is complete.",
        difficulty="easy",
        required_steps=["restart", "healthcheck", "notify"],
        core_steps=["restart"]
    ),
    Task(
        id="easy-rollback-notify",
        prompt="The recent deployment of 'auth-service' caused a spike in errors. Rollback the deployment, check the health, and notify '#on-call' that the service has been rolled back.",
        difficulty="easy",
        required_steps=["rollback", "healthcheck", "notify"],
        core_steps=["rollback"]
    ),
    Task(
        id="easy-deploy-scale",
        prompt="Deploy 'analytics-engine' version 'v3.0.0', then quickly scale it to 10 replicas for the incoming load test. Check health to verify and notify '#data-team'.",
        difficulty="easy",
        required_steps=["deploy", "scale", "healthcheck", "notify"],
        core_steps=["deploy", "scale"]
    ),
]

EASY_TASKS_BY_ID = {t.id: t for t in _EASY_TASKS}
DEFAULT_EASY_TASK_QUEUE = list(_EASY_TASKS)

def build_easy_task_queue(task_id: str | None = None) -> list[Task]:
    """
    Returns a deterministic queue of easy tasks.
    If task_id is provided, starts the queue with that task, followed by
    the remaining easy tasks in their default order.
    """
    if task_id is None:
        return list(DEFAULT_EASY_TASK_QUEUE)
    
    if task_id not in EASY_TASKS_BY_ID:
        raise ValueError(f"Task ID '{task_id}' is not a valid easy task. Available: {list(EASY_TASKS_BY_ID.keys())}")
        
    # Build queue starting with the requested task, then append the rest in order
    first_task = EASY_TASKS_BY_ID[task_id]
    queue = [first_task]
    for task in DEFAULT_EASY_TASK_QUEUE:
        if task.id != task_id:
            queue.append(task)
            
    return queue
