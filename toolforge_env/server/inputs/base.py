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
