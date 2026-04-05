from server.inputs.simulated.data_loader import SimulatedDataLoader


def create_input_provider(task_list):
    return SimulatedDataLoader(task_list=task_list)