"""
Module to manage the configuration file for constants and configuration settings for an NZGMDB run.
"""

from enum import Enum

import yaml

from nzgmdb.management import file_structure


class MachineName(str, Enum):
    """
    Enum for the machine names.
    """

    LOCAL = "local"
    MANTLE = "mantle"
    HYPOCENTRE = "hypocentre"


class WorkflowStep(str, Enum):
    """
    Enum for the workflow steps.
    """

    GEONET = "geonet"
    TEC_DOMAIN = "tec_domain"
    PHASE_TABLE = "phase_table"
    SNR = "snr"
    FMAX = "fmax"
    GMC = "gmc"
    PROCESS = "process"
    IM = "im"
    DISTANCES = "distances"
    UPLOAD = "upload"
    DEFAULT = "default"


class Config:
    """
    Class to manage the config file for constants and configuration settings for an NZGMDB run.

    This class follows a singleton pattern, ensuring only one instance exists.
    It loads configuration values from a YAML file.
    """

    _instance = None
    config_path = file_structure.get_data_dir() / "config.yaml"
    machine_config_path = file_structure.get_data_dir() / "machine_config.yaml"

    def __new__(cls, *args, **kwargs):
        """
        Ensure only one instance of the class is created (Singleton Pattern).

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        Config
            The single instance of the `Config` class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._config_data = cls._instance._load_config()
            cls._instance._machine_config_data = cls._instance._load_machine_config()
        return cls._instance

    def _load_config(self) -> dict:
        """
        Load the configuration file.

        Returns
        -------
        dict
            The loaded configuration as a dictionary. Returns an empty dictionary if the file is not found.
        """
        try:
            with open(self.config_path, "r") as file:
                config_data = yaml.safe_load(file)
            return config_data or {}  # Ensure it always returns a dictionary
        except FileNotFoundError:
            print(f"Config file not found at {self.config_path}")
            return {}

    def _load_machine_config(self):
        """
        Load the machine config file.
        """
        try:
            with open(self.machine_config_path, "r") as file:
                machine_config_data = yaml.safe_load(file)
            return machine_config_data
        except FileNotFoundError:
            print("Machine config file not found.")

    def get_value(self, key: str):
        """
        Retrieve the value associated with a key in the configuration.

        Parameters
        ----------
        key : str
            The key to search for in the configuration file.

        Returns
        -------
        Any
            The value associated with the key if found.

        Raises
        ------
        KeyError
            If the key is not found in the configuration.
        """
        if key in self._config_data:
            return self._config_data[key]
        raise KeyError(f"Error: Key '{key}' not found in {self.config_path}")

    def get_n_procs(self, machine_name: MachineName, step: WorkflowStep):
        """
        Get the number of processes for a given machine and workflow step.

        Parameters
        ----------
        machine_name : MachineName
            The name of the machine.
        step : WorkflowStep
            The workflow step.

        Returns
        -------
        int
            The number of processes.

        Raises
        ------
        KeyError
            If the machine or workflow step is not found in the configuration.
        """
        machine_config = self._machine_config_data.get(machine_name.value)
        if not machine_config:
            raise KeyError(
                f"Machine '{machine_name.value}' not found in the configuration."
            )

        n_procs = machine_config.get(step.value)
        if n_procs is None:
            raise KeyError(
                f"Workflow step '{step.value}' not found for machine '{machine_name.value}' in the configuration."
            )

        return n_procs
