from pathlib import Path
from typing import Annotated

import typer

from nzgmdb.mseed_management.mseed_to_gmprocess import (
    convert_mseed_to_gmprocess,
)

app = typer.Typer()


@app.command(help="Converts mseed data to gmprocess format and file structure")
def mseed_to_gmprocess(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            exists=True,
            file_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory to save the gmprocessed data",
            exists=True,
            file_okay=False,
        ),
    ],
    n_procs: Annotated[
        int, typer.Option(help="The number of processes to use for processing")
    ] = 1,
):
    convert_mseed_to_gmprocess(main_dir, output_dir, n_procs)


if __name__ == "__main__":
    app()
