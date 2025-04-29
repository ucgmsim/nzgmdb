"""
This script is used to convert mseed data to gmprocess format and file structure.
"""

from pathlib import Path
from typing import Annotated

import typer

from nzgmdb.mseed_management.mseed_to_gmprocess import convert_mseed_to_gmprocess
from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


@cli.from_docstring(app)
def mseed_to_gmprocess(
    main_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    n_procs: Annotated[int, typer.Option()] = 1,
):
    """
    Convert MiniSEED data to gmprocess format and file structure.

    This function processes MiniSEED data and converts it into a format
    compatible with gmprocess, saving the results in the specified output directory.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (highest level directory).
    output_dir : Path
        The directory where the gmprocessed data will be saved.
    n_procs : int, optional
        The number of processes to use for processing (default is 1).
    """
    convert_mseed_to_gmprocess(main_dir, output_dir, n_procs)


if __name__ == "__main__":
    app()
