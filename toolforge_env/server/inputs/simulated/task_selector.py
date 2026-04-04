class TaskSelector:
    def __init__(self, mode: str):
        self.mode = mode

        self.indices = {
            "easy": 0,
            "medium": 0,
            "hard": 0,
        }

        self.eval_map = {
            "easy": 0,
            "medium": 1,
            "hard": 0,
        }

    def next_task_list(self, difficulty: str):
        from server.inputs.simulated.tasks import TASKS_BY_DIFFICULTY

        task_lists = TASKS_BY_DIFFICULTY[difficulty]

        if self.mode == "train":
            idx = self.indices[difficulty]
            task_list = task_lists[idx]
            self.indices[difficulty] = (idx + 1) % len(task_lists)
            return task_list

        elif self.mode == "eval":
            return task_lists[self.eval_map[difficulty]]

        else:
            raise ValueError(f"Unknown mode {self.mode}")