"""
Script to run PhaseNet on mseed files, predict p and s waves as well as save the probability series to an HDF5 file.
"""

import argparse
from pathlib import Path

import h5py
import mseedlib
import numpy as np
import pandas as pd
from obspy import Inventory, Stream, Trace, UTCDateTime, read_inventory
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.inventory.response import Response


def check_sensitivity(
    resp: Response,
    threshold: float = 10.0,
) -> tuple[bool, float]:
    """
    Returns True if full response removal is safe,
    False if sensitivity mismatch suggests using remove_sensitivity().

    Parameters
    ----------
    resp: Response
        The response object to check the sensitivity of
    threshold : float, optional
        The percentage difference threshold to determine if the sensitivity mismatch is acceptable, by default 10.0

    Returns
    -------
    bool
        True if the percentage difference is less than or equal to the threshold, False otherwise
    float
        The percentage difference between the total sensitivity and the stage sensitivity

    Reference
    ----------
        USGS gmprocess instrument-response helper:
        https://ghsc.code-pages.usgs.gov/esi/groundmotion-processing/_modules/gmprocess/waveform_processing/instrument_response.html
    """
    stages = resp.response_stages

    # Total sensitivity
    total = resp.instrument_sensitivity.value

    # Stage sensitivity (product of stage gains)
    stage = 1.0
    for s in stages:
        stage *= s.stage_gain

    # Percent difference
    pct_diff = 200.0 * abs(total - stage) / (total + stage)

    return pct_diff <= threshold, pct_diff


def run_phase_net(
    input_data: np.ndarray,
    dt: float,
    t: np.ndarray = None,
    return_prob_series: bool = False,
):
    """
    Uses PhaseNet to get the p- & s-wave pick

    Parameters
    ----------
    input_data : np.ndarray
        The input data to run PhaseNet on. Shape (1, n_samples, 3)
    dt : float
        The sampling rate of the input data
    t : np.ndarray, optional
        The time vector of the input data, by default None
    return_prob_series : bool, optional
        Whether to return the probability series, by default False

    Returns
    -------
    int: p_wave_ix
        The p-wave index
    int: s_wave_ix
        The s-wave index
    """
    import phase_net as ph  # noqa: DEP001

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

        assert not np.any(np.isnan(input_resampled))

        probs = ph.predict(input_resampled)
        p_wave_ix, s_wave_ix = np.argmax(probs[0, :, 1]), np.argmax(probs[0, :, 2])

        # Get the probability of the p and s wave
        p_prob, s_prob = probs[0, p_wave_ix, 1], probs[0, s_wave_ix, 2]

        # Adjust for original dt
        p_wave_ix = int(np.round((dt_new / dt) * p_wave_ix))
        s_wave_ix = int(np.round((dt_new / dt) * s_wave_ix))
    else:
        probs = ph.predict(input_data)
        p_wave_ix, s_wave_ix = np.argmax(probs[0, :, 1]), np.argmax(probs[0, :, 2])
        p_prob, s_prob = probs[0, p_wave_ix, 1], probs[0, s_wave_ix, 2]

    if return_prob_series:
        return p_wave_ix, s_wave_ix, probs[0, :, 1], probs[0, :, 2], p_prob, s_prob

    return p_wave_ix, s_wave_ix


def process_mseed(
    mseed_file: Path,
    h5_ffp: Path,
    bypass_row: pd.Series = None,
    inventory: Inventory = None,
):
    """
    Process an mseed file and return the phase arrival data.

    Parameters
    ----------
    mseed_file : Path
        Path to the mseed file.
    h5_ffp : Path
        Path to the HDF5 file to save the probability series.
    bypass_row : pd.Series, optional
        A row from the bypass file with known p and s wave datetimes, by default None
    inventory : Inventory, optional
        The inventory object to use for sensitivity removal, by default None (Will try extract from FDSN if not provided)

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
    try:
        mstl.read_file(str(mseed_file), unpack_data=False, record_list=True)
    except mseedlib.exceptions.MseedLibError:
        skipped_record = pd.DataFrame(
            {
                "record_id": [mseed_file.stem],
                "reason": ["Failed to read mseed file with mseedlib"],
            }
        )
        return None, skipped_record

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

    if len(mseed) != 3:
        skipped_record = pd.DataFrame(
            {
                "record_id": [mseed_file.stem],
                "reason": ["File did not contain 3 components"],
            }
        )
        return None, skipped_record

    # Small Processing
    mseed.detrend("demean")
    mseed.detrend("linear")

    # Get the inventory information
    station = mseed[0].stats.station
    location = mseed[0].stats.location
    channel = mseed[0].stats.channel[:2]

    if inventory is None:
        # try:
        #     client_NZ = FDSN_Client("GEONET")
        #     inv = client_NZ.get_stations(
        #         level="response", network="NZ", station=station, location=location
        #     )
        # except (FDSNNoDataException, TypeError):
        skipped_record = pd.DataFrame(
            {
                "record_id": [mseed_file.stem],
                "reason": ["Failed to find Inventory information"],
            }
        )
        return None, skipped_record
    else:
        inv = inventory

    try:
        # Apply the correct sensitivity removal based on the check_sensitivity function
        t = UTCDateTime(mseed[0].stats.starttime)
        resp = inv.get_response(mseed[0].id, t)
        ok, diff = check_sensitivity(resp)
        paz = resp.get_paz()
        has_paz = not (len(paz.poles) == 0 and len(paz.zeros) == 0)

        # Checks that the response has poles and zeros and that the sensitivity mismatch is acceptable before applying the full remove_response method.
        if has_paz and ok:
            if channel[:2] in ["HN", "BN"]:
                mseed = mseed.remove_response(inventory=inv, output="ACC")
            else:
                # We have a broadband record so need to apply some pre-filters
                f_nyq = 0.5 / mseed[0].stats.delta
                f3 = 0.9 * f_nyq
                pre_filt = (0.01, 0.05, f3, f_nyq)
                mseed = mseed.remove_response(
                    inventory=inv,
                    output="VEL",
                    pre_filt=pre_filt,
                    zero_mean=True,
                    taper=True,
                )
        else:
            # Now we must use remove sensitivity instead
            mseed = mseed.remove_sensitivity(inventory=inv)
    except Exception:  # noqa: BLE001
        skipped_record = pd.DataFrame(
            {
                "record_id": [mseed_file.stem],
                "reason": ["Failed to remove sensitivity"],
            }
        )
        return None, skipped_record

    if channel[:2] not in ["HN", "BN"]:
        try:
            # differentiate data i.e., m/s to m/s^2
            mseed.differentiate()
        except ValueError:
            skipped_record = pd.DataFrame(
                {
                    "record_id": [mseed_file.stem],
                    "reason": [
                        "Failed to differentiate data after sensitivity removal"
                    ],
                }
            )
            return None, skipped_record

    try:
        p_wave_ix, s_wave_ix, p_prob_series, s_prob_series, p_prob, s_prob = (
            run_phase_net(
                np.stack([trace.data for trace in mseed], axis=1)[np.newaxis, ...],
                mseed[0].stats["delta"],
                return_prob_series=True,
            )
        )
    except ValueError:
        skipped_record = pd.DataFrame(
            {
                "record_id": [mseed_file.stem],
                "reason": ["Zero size array after re-sample"],
            }
        )
        return None, skipped_record

    # Save the prob_series
    with h5py.File(h5_ffp, "a") as f:
        group = f.create_group(mseed_file.stem)
        group.create_dataset(
            "p_prob_series",
            data=p_prob_series.astype(np.float32),
            dtype="float32",
            compression="lzf",
        )
        group.create_dataset(
            "s_prob_series",
            data=s_prob_series.astype(np.float32),
            dtype="float32",
            compression="lzf",
        )

    # Get the extra datetime columns of p and s wave
    tr1 = mseed[0]
    start_time = tr1.stats.starttime
    end_time = tr1.stats.endtime
    times = np.linspace(start_time.timestamp, end_time.timestamp, tr1.stats.npts)

    if bypass_row is not None:
        p_wave_val = bypass_row["p_wave_time"]
        s_wave_val = bypass_row["s_wave_time"]

        if p_wave_val is not None and not pd.isna(p_wave_val):
            p_wave_datetime = UTCDateTime(p_wave_val)
            p_wave_ix = int((p_wave_datetime - start_time) * tr1.stats.sampling_rate)
            p_prob = 1.0
        else:
            p_wave_datetime = UTCDateTime(times[p_wave_ix])

        if s_wave_val is not None and not pd.isna(s_wave_val):
            s_wave_datetime = UTCDateTime(s_wave_val)
            s_wave_ix = int((s_wave_datetime - start_time) * tr1.stats.sampling_rate)
            s_prob = 1.0
        else:
            s_wave_datetime = UTCDateTime(times[s_wave_ix])
    else:
        p_wave_datetime = UTCDateTime(times[p_wave_ix])
        s_wave_datetime = UTCDateTime(times[s_wave_ix])

    return (
        pd.DataFrame(
            {
                "record_id": [mseed_file.stem],
                "p_wave_ix": [p_wave_ix],
                "p_wave_datetime": [p_wave_datetime],
                "p_wave_prob": [p_prob],
                "s_wave_ix": [s_wave_ix],
                "s_wave_datetime": [s_wave_datetime],
                "s_wave_prob": [s_prob],
            }
        ),
        None,
    )


def run_phasenet(
    mseed_files_ffp: Path,
    output_dir: Path,
    bypass_ffp: Path = None,
    xml_dir: Path = None,
):
    """
    Run PhaseNet on the mseed files.

    Parameters
    ----------
    mseed_files_ffp : Path
        Full File path to a list of mseed full file paths to process.
    output_dir : Path
        Output directory for skipped records and phase arrival information.
    bypass_ffp : Path, optional
        Optional bypass file path with known p and s wave datetimes, by default None
    xml_dir : Path, optional
        Optional directory containing station xml files to use for sensitivity removal, by default None (Will try extract from FDSN if not provided)
    """
    # Read the .txt for the mseed files to process
    mseed_files = mseed_files_ffp.read_text().splitlines()

    skipped_records = []
    phase_arrival_table = []
    h5_ffp = output_dir / "prob_series.h5"

    # Ensure output directory and HDF5 container exist before processing.
    output_dir.mkdir(parents=True, exist_ok=True)
    if not h5_ffp.exists():
        # create an empty HDF5 file (no groups)
        with h5py.File(h5_ffp, "w"):
            pass

    if bypass_ffp is not None:
        # Read the bypass file
        bypass_df = pd.read_csv(bypass_ffp)

    # Process each mseed file
    for mseed_file in mseed_files:
        mseed_file = mseed_file.strip()
        mseed_file = Path(mseed_file)
        bypass_row = None
        if bypass_ffp is not None:
            bypass_rows = bypass_df.loc[bypass_df["record_id"] == mseed_file.stem]
            if len(bypass_rows) > 0:
                bypass_row = bypass_rows.iloc[0]

        inventory = None
        if xml_dir is not None:
            station = mseed_file.stem.split("_")[1]
            xml_file = xml_dir / f"{station}.xml"
            if xml_file.exists():
                inventory = read_inventory(xml_file)
        phase_arrival, skipped_record = process_mseed(
            mseed_file, h5_ffp, bypass_row, inventory=inventory
        )
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
    parser.add_argument(
        "--bypass_ffp",
        type=Path,
        help="Optional bypass file path with known p and s wave datetimes.",
        default=None,
    )
    parser.add_argument(
        "--xml_dir",
        type=Path,
        help="Optional directory containing station xml files to use for sensitivity removal.",
        default=None,
    )
    args = parser.parse_args()
    run_phasenet(args.mseed_files_ffp, args.output_dir, args.bypass_ffp, args.xml_dir)
