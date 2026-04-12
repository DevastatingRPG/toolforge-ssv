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

    def task_count(self) -> int:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # task_count
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return the total number of tasks in this episode.
        #
        # The environment passes this value to the UI via observation
        # metadata so the UI knows the episode length without relying on
        # env.done, which would break when the HTTP server creates a fresh
        # env instance per request (stateless REST mode).
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return len(self.tasks)

    def reset(self):
        self.idx = 0