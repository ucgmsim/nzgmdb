"""
Contains functions to run shell commands with Conda environment activation.
"""

import os
import subprocess
from pathlib import Path


def run_command(
    command: str, env_sh: Path, env_activate_command: str, log_file_path: Path
):
    """
    Run a shell command with Conda environment activation.

    Parameters
    ----------
    command : str
        The command to run.
    env_sh : Path
        The path to the conda.sh script.
    env_activate_command : str
        The command to activate the conda environment needed.
    log_file_path : Path
        The path to the log file.

    Raises
    ------
    Exception
        If the command fails.
    """
    with open(log_file_path, "w") as log_file:
        # Create the command to source conda.sh, activate the environment, and execute the full command
        command = f"source {env_sh} && {env_activate_command} && {command}"
        env = os.environ.copy()
        try:
            subprocess.check_call(
                command,
                stdout=log_file,
                stderr=log_file,
                shell=True,
                executable="/bin/bash",
                env=env,
            )
        except subprocess.CalledProcessError:
            raise Exception(f"Command failed please check logs in {log_file_path}")
