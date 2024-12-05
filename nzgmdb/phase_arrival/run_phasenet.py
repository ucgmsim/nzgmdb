import os

# os.environ["LD_LIBRARY_PATH"] = (
#     os.environ.get("LD_LIBRARY_PATH", "") + ":/home/joel/anaconda3/lib"
# )
print(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
import argparse
import json
from pathlib import Path

import mseedlib
import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import FDSNNoDataException


def run_phase_net(
    input_data: np.ndarray,
    dt: float,
    t: np.ndarray = None,
    return_prob_series: bool = False,
):
    """Uses PhaseNet to get the p- & s-wave pick"""
    import phase_net as ph

    # Only supports a single record
    assert input_data.shape[0] == 1

    t = t if t is not None else np.arange(input_data.shape[1]) * dt

    # Have to re-sample
    if not np.isclose(dt, 1 / 100):
        dt_new = 1 / 100
        t_new = np.arange(t.max() / dt_new) * dt_new
        input_resampled = np.full((1, t_new.shape[0], 3), np.nan)
        input_resampled[0, :, 0] = np.interp(t_new, t, input_data[0, :, 0])
        input_resampled[0, :, 1] = np.interp(t_new, t, input_data[0, :, 1])
        input_resampled[0, :, 2] = np.interp(t_new, t, input_data[0, :, 2])

        assert np.all(~np.isnan(input_resampled))

        probs = ph.predict(input_resampled)
        p_wave_ix, s_wave_ix = np.argmax(probs[0, :, 1]), np.argmax(probs[0, :, 2])

        # Adjust for original dt
        p_wave_ix = int(np.round((dt_new / dt) * p_wave_ix))
        s_wave_ix = int(np.round((dt_new / dt) * s_wave_ix))
    else:
        probs = ph.predict(input_data)
        p_wave_ix, s_wave_ix = np.argmax(probs[0, :, 1]), np.argmax(probs[0, :, 2])

    if return_prob_series:
        return p_wave_ix, s_wave_ix, probs[0, :, 1], probs[0, :, 2]

    return p_wave_ix, s_wave_ix


def process_mseed(mseed_file: Path):
    """
    Process an mseed file and return the phase arrival data.

    Parameters
    ----------
    mseed_file : Path
        Path to the mseed file.

    Returns
    -------
    pd.DataFrame | None
        The phase arrival data.
    pd.DataFrame | None
        The skipped record data.
    """
    mseed = Stream()
    nptype = {"i": np.int32, "f": np.float32, "d": np.float64, "t": np.char}
    mstl = mseedlib.MSTraceList()
    mstl.read_file(str(mseed_file), unpack_data=False, record_list=True)

    for traceid in mstl.traceids():
        for segment in traceid.segments():
            # Determine data type and allocate array
            (sample_size, sample_type) = segment.sample_size_type
            dtype = nptype[sample_type]
            data_samples = np.zeros(segment.samplecnt, dtype=dtype)

            # Unpack data samples
            segment.unpack_recordlist(
                buffer_pointer=np.ctypeslib.as_ctypes(data_samples),
                buffer_bytes=data_samples.nbytes,
            )

            # Get metadata
            sourceid = traceid.sourceid.split("FDSN:")[1]
            parts = sourceid.split("_")
            if len(parts) > 4:
                network, station, location, *channel = parts
                channel = "".join(channel)
            else:
                network, station, location, channel = parts
            start_time = UTCDateTime(segment.starttime_seconds)
            sampling_rate = segment.samprate

            # Create ObsPy Trace and add to Stream
            trace = Trace(data=data_samples)
            trace.stats.network = network
            trace.stats.station = station
            trace.stats.location = location
            trace.stats.channel = channel
            trace.stats.starttime = start_time
            trace.stats.sampling_rate = sampling_rate
            mseed.append(trace)

    # Small Processing
    mseed.detrend("demean")
    mseed.detrend("linear")

    # Get the inventory information
    station = mseed[0].stats.station
    location = mseed[0].stats.location

    # Get Station Information from geonet clients
    # Fetching here instead of passing the inventory object as searching for the station, network, and channel
    # information takes a long time as it's implemented in a for loop
    try:
        client_NZ = FDSN_Client("GEONET")
        inv = client_NZ.get_stations(
            level="response", network="NZ", station=station, location=location
        )
    except FDSNNoDataException:
        skipped_record = pd.DataFrame(
            {
                "record_id": [mseed_file.stem],
                "reason": ["Failed to find Inventory information"],
            }
        )
        return None, skipped_record

    # Add the response (Same for all channels)
    # this is done so that the sensitivity can be removed otherwise it tries to find the exact same channel
    # which can fail when including the inventory information
    response = next(cha.response for sta in inv.networks[0] for cha in sta.channels)
    for tr in mseed:
        tr.stats.response = response

    try:
        mseed = mseed.remove_sensitivity()
    except ValueError:
        skipped_record = pd.DataFrame(
            {
                "record_id": [mseed_file.stem],
                "reason": ["Failed to remove sensitivity"],
            }
        )
        return None, skipped_record

    p_wave_ix, s_wave_ix, p_prob_series, s_prob_series = run_phase_net(
        np.stack([trace.data for trace in mseed], axis=1)[np.newaxis, ...],
        mseed[0].stats["delta"],
        return_prob_series=True,
    )

    # Convert the probability series to JSON strings
    p_prob_series_json = json.dumps(p_prob_series.tolist())
    s_prob_series_json = json.dumps(s_prob_series.tolist())

    return (
        pd.DataFrame(
            {
                "record_id": [mseed_file.stem],
                "p_wave_ix": [p_wave_ix],
                "s_wave_ix": [s_wave_ix],
                "p_prob_series": [p_prob_series_json],
                "s_prob_series": [s_prob_series_json],
            }
        ),
        None,
    )


def run_phasenet(mseed_files_ffp: Path, output_dir: Path):
    """
    Run PhaseNet on the mseed files.

    Parameters
    ----------
    mseed_files_ffp : Path
        Full File path to a list of mseed full file paths to process.
    output_dir : Path
        Output directory for skipped records and phase arrival information.
    """
    # Read the .txt for the mseed files to process
    with open(mseed_files_ffp, "r") as f:
        mseed_files = f.readlines()

    skipped_records = []
    phase_arrival_table = []

    # Process each mseed file
    for mseed_file in mseed_files:
        mseed_file = mseed_file.strip()
        mseed_file = Path(mseed_file)
        phase_arrival, skipped_record = process_mseed(mseed_file)
        if phase_arrival is not None:
            phase_arrival_table.append(phase_arrival)
        if skipped_record is not None:
            skipped_records.append(skipped_record)

    # Combine the phase arrival data
    if len(phase_arrival_table) > 0:
        phase_arrival_table = pd.concat(phase_arrival_table)
    else:
        phase_arrival_table = pd.DataFrame(
            columns=[
                "record_id",
                "p_wave_ix",
                "s_wave_ix",
                "p_prob_series",
                "s_prob_series",
            ]
        )
    phase_arrival_table.to_csv(output_dir / "phase_arrival_table.csv", index=False)

    # Combine the skipped records
    if len(skipped_records) > 0:
        skipped_records = pd.concat(skipped_records)
    else:
        skipped_records = pd.DataFrame(columns=["record_id", "reason"])
    skipped_records.to_csv(output_dir / "skipped_records.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PhaseNet on mseed files.")
    parser.add_argument(
        "mseed_files_ffp",
        type=Path,
        help="File path to a list of mseed files to process.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for skipped records and phase arrival information.",
    )
    args = parser.parse_args()
    # Set the environment variable
    run_phasenet(args.mseed_files_ffp, args.output_dir)
