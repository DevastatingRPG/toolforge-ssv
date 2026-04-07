from server.inputs.simulated.tasks import TASKS

class TaskSelector:
    def __init__(self):
        pass

    def next_task_list(self, task_id: str):
        match = next((item for item in TASKS if item.get("task_id") == task_id), None)
        if match is None:
            raise ValueError(f"Unknown task_id: {task_id}")
        return match["tasks"]
