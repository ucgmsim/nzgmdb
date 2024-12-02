import yaml

from nzgmdb.management import file_structure


class Config:
    """
    Class to manage the config file for constants and configuration settings for a NZGMDB run.
    """

    _instance = None
    config_path = file_structure.get_data_dir() / "config.yaml"

    def __new__(cls, *args, **kwargs):
        """
        Singleton pattern to ensure only one instance of the class is created.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._config_data = cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """
        Load the config file.
        """
        try:
            with open(self.config_path, "r") as file:
                config_data = yaml.safe_load(file)
            return config_data
        except FileNotFoundError:
            print("Config file not found.")
            return {}

    def get_value(self, key: str):
        """
        Get the value of a key from the config file.
        If the key is not found, return an error message.

        Parameters
        ----------
        key : str
            The key to search for in the config file.
        """
        if key in self._config_data:
            return self._config_data[key]
        else:
            return KeyError(f"Error: Key '{key}' not found in {self.config_path}")
