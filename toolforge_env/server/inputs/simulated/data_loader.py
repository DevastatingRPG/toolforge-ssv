from typing import Dict, Any, List
from server.inputs.base import InputProvider
from models import Task


class SimulatedDataLoader(InputProvider):
    def __init__(self, task_list: Dict):
        self.tasks: List[Task] = task_list["tasks"]
        self.idx = 0

    def get_input(self) -> Task:
        if self.is_done():
            raise StopIteration

        task = self.tasks[self.idx]
        self.idx += 1

        return task

    def is_done(self) -> bool:
        return self.idx >= len(self.tasks)

    def reset(self):
        self.idx = 0