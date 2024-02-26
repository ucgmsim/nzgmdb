"""
File that contains the function scripts that can be called to run the NZGMDB pipeline.
"""
import typer
from pathlib import Path

app = typer.Typer()


@app.command()
def gen_phase_arrival_table(data_dir: Path, output_dir: Path, n_procs: int):
    """
    Generate the phase arrival table, taking mseed data and finding the phase arrivals using a p_wave picker.

    Parameters
    ----------
    data_dir : Path
        The top directory containing the mseed files
        (glob is used to find all mseed files recursively)
    output_dir : Path
        The directory to save the phase arrival table
    n_procs : int
        The number of processes to use to generate the phase arrival table
    """
    from phase_arrival.gen_phase_arrival_table import generate_phase_arrival_table

    generate_phase_arrival_table(data_dir, output_dir, n_procs)


@app.command()
def test():
    """
    Simple Typer Test Case
    """
    print("Testing Typer")


if __name__ == "__main__":
    app()
