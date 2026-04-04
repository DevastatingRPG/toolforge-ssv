from typing import List, Dict, Any
from server.inputs.base import InputProvider
from models import Task


class SimulatedDataLoader(InputProvider):
    """Simulates user input from a predefined dataset."""

    def __init__(self, data: List[Task]):
        self.data = data
        self.idx = 0

    def get_input(self) -> Task:
        if self.is_done():
            raise StopIteration("No more inputs")

        item = self.data[self.idx]
        self.idx += 1
        return item

    def is_done(self) -> bool:
        return self.idx >= len(self.data)

    def reset(self) -> None:
        self.idx = 0