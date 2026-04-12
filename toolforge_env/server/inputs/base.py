from abc import ABC, abstractmethod
from models import Task


class InputProvider(ABC):
    """Abstract interface for all input sources."""

    @abstractmethod
    def get_input(self) -> Task:
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def task_count(self) -> int:
        """Return the total number of tasks in this episode.

        Used by the environment to expose episode length metadata to callers
        (e.g. the Gradio UI) so they can track progress without relying on
        the env's done flag.
        """
        pass
