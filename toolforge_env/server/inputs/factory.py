from server.inputs.simulated.data_loader import SimulatedDataLoader
from tasks import _EASY_TASKS

def create_input_provider():
    """Factory for input providers (currently only simulated)."""

    data = _EASY_TASKS

    return SimulatedDataLoader(data=data)