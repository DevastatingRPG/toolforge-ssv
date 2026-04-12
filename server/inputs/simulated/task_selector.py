from typing import Dict, Any

from server.inputs.simulated.tasks import TASKS


TASK_ID_MAP = {
    "easy": "easy-deployment-sprints",
    "medium": "medium-traffic-readiness",
    "hard": "hard-project-legacy-migration",
}


class TaskSelector:
    def __init__(self):
        pass

    def next_task_list(self, task_id: str) -> Dict:
        # Prefer direct internal IDs; if not provided, resolve deterministic aliases.
        resolved_task_id = task_id
        direct_match = next((item for item in TASKS if item.get("task_id") == resolved_task_id), None)
        if direct_match is not None:
            return direct_match

        resolved_task_id = TASK_ID_MAP.get(task_id, task_id)

        match = next((item for item in TASKS if item.get("task_id") == resolved_task_id), None)
        if match is None:
            raise ValueError(f"Unknown task_id: {task_id} (resolved: {resolved_task_id})")
        return match