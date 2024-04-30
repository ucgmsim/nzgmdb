import typer
from pathlib import Path

from nzgmdb.mseed_management.mseed_to_gmprocess import convert_mseed_to_gmprocess

app = typer.Typer()


@app.command()
def mseed_to_gmprocess(
    main_dir: Path, output_dir: Path, old_style: bool = False, n_procs: int = 1
):
    """
    Converts mseed data to gmprocess format and file structure.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    output_dir : Path
        The directory to save the gmprocessed data
    old_style : bool (optional)
        Whether the data is stored in the old style
    n_procs : int (optional)
        The number of processes to use for processing
    """
    convert_mseed_to_gmprocess(main_dir, output_dir, old_style, n_procs)


if __name__ == "__main__":
    app()
