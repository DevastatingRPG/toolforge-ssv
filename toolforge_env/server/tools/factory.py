from server.tools.seeded.tools import build_atomic_tools
from server.tools.base import AbstractToolStore
from server.tools.seeded.seeded_store import SeededInMemoryToolStore

def create_tool_store() -> AbstractToolStore:
    """Create the default seeded in-memory tool store for the environment."""
    return SeededInMemoryToolStore(build_atomic_tools())
