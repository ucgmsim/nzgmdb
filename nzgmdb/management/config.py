import yaml
from nzgmdb.management import file_structure


class Config:
    """
    Class to manage the config file for constants and configuration settings for an NZGMDB run.

    This class follows a singleton pattern, ensuring only one instance exists.
    It loads configuration values from a YAML file.
    """

    _instance = None
    config_path = file_structure.get_data_dir() / "config.yaml"

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
