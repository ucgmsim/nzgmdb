from enum import Enum
from pathlib import Path

import yaml


class MachineName(str, Enum):
    MACHINE1 = "machine1"
    MACHINE2 = "machine2"


class WorkflowStep(str, Enum):
    GEONET = "geonet"
    PHASE_TABLE = "phase_table"
    SNR = "snr"
    GMC = "gmc"
    IM = "IM"
    DISTANCES = "distances"
    DEFAULT = "default"


def read_config(config_path: Path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_n_procs(config, machine_name: MachineName, step: WorkflowStep):
    machine_config = config.get("machines", {}).get(machine_name.value, {})
    return machine_config.get(
        step.value, machine_config.get(WorkflowStep.DEFAULT.value)
    )


# Example usage
if __name__ == "__main__":
    config_path = Path("machine_config.yaml")
    config = read_config(config_path)

    machine_name = MachineName.MACHINE1
    step = WorkflowStep.GEONET

    n_procs = get_n_procs(config, machine_name, step)
    print(f"Number of processes for {machine_name} and {step}: {n_procs}")
