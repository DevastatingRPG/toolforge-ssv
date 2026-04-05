from server.toolforge_env_environment import ToolforgeEnvironment
from server.inputs.simulated.task_selector import TaskSelector
from server.inputs.factory import create_input_provider

def create_env() -> ToolforgeEnvironment:
    # mode unknown yet → placeholder
    selector = TaskSelector(mode="eval")

    return ToolforgeEnvironment(
        task_selector=selector,
        input_provider_factory=create_input_provider
    )