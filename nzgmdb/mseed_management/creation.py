import warnings
import http.client
from pathlib import Path

import pandas as pd
from obspy import Stream
from obspy.taup import TauPyModel
from obspy.core.event import Origin
from obspy.geodetics import kilometers2degrees
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError

from nzgmdb.management import config as cfg
from empirical.util import classdef, openquake_wrapper_vectorized, z_model_calculations


def get_waveforms(
    preferred_origin: Origin,
    client: FDSN_Client,
    net: str,
    sta: str,
    mag: float,
    rrup: float,
    r_epi: float,
):
    """
    Get the waveforms for a given event and station
    Calculate the start and end times for the waveform based on the P and S arrival times

    Parameters
    ----------
    preferred_origin : Origin
        The preferred origin object from the event catalogue
    client : FDSN_Client
        The FDSN client to use to get the waveforms
    net : str
        The network code
    sta : str
        The station
    mag : float
        The magnitude of the event
    rrup : float
        The closest distance to the event
    r_epi : float
        The epicentral distance to the event

    Returns
    -------
    st : Union[Stream, None]
        The stream object containing the waveform data or None if no data is found
    """
    config = cfg.Config()
    vs30 = config.get_value("vs30")
    rake = 90  # TODO get from the earthquake source table
    z1p0 = z_model_calculations.chiou_young_08_calc_z1p0(vs30)
    # Predict significant duration time from Afshari and Stewart (2016)
    input_df = pd.DataFrame(
        {
            "mag": [mag],
            "rake": [rake],
            "rrup": [rrup],
            "vs30": [vs30],
            "z1pt0": [z1p0],
        }
    )
    result_df = openquake_wrapper_vectorized.oq_run(
        classdef.GMM.AS_16,
        classdef.TectType.ACTIVE_SHALLOW,
        input_df,
        "Ds595",
    )
    ds = result_df["Ds595_mean"].values[0]

    deg = kilometers2degrees(r_epi)

    model = TauPyModel(model="iasp91")

    # Estimate arrival times for P and S phases
    p_arrivals = model.get_travel_times(
        source_depth_in_km=preferred_origin.depth / 1000,
        distance_in_degree=deg,
        phase_list=["ttp"],
    )
    s_arrivals = model.get_travel_times(
        source_depth_in_km=preferred_origin.depth / 1000,
        distance_in_degree=deg,
        phase_list=["tts"],
    )
    ptime_est = (
        preferred_origin.time + p_arrivals[0].time
    )  # Estimated earliest P arrival time from taup
    stime_est = preferred_origin.time + s_arrivals[0].time

    # Find the start and end times for the mseed data
    min_time_difference = config.get_value("min_time_difference")
    ds_multiplier = config.get_value("ds_multiplier")
    start_time = ptime_est - min_time_difference
    end_time = stime_est + (
        min_time_difference
        if stime_est + ds * ds_multiplier - ptime_est < min_time_difference
        else ds * ds_multiplier
    )

    channel_codes = ",".join(config.get_value("channel_codes"))
    # Get the waveforms with multiple retries when IncompleteReadError occurs
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                st = client.get_waveforms(
                    net,
                    sta,
                    "*",
                    channel_codes,
                    start_time,
                    end_time,
                    attach_response=True,
                )
            break
        except FDSNNoDataException:
            print(f"No data for {net}.{sta}")
            return None
        except ObsPyMSEEDFilesizeTooSmallError:
            print(f"File too small for {net}.{sta}")
            return None
        except http.client.IncompleteRead:
            if attempt < max_retries - 1:  # i.e. not the last attempt
                continue  # try again
            else:
                raise  # re-raise exception on the last attempt
    return st


def create_mseed_from_waveforms(st: Stream, event_id: str, sta: str, output_dir: Path):
    """
    Create mseed files from the waveform data

    Parameters
    ----------
    st : object
        The stream object containing the waveform data from each component
    event_id : str
        The event id
    sta : str
        The station name
    output_dir : Path
        The directory to save the mseed files

    Returns
    -------
    chan_locs : list
        A list of tuples containing the channel and location for each mseed file created
        [(channel, location), ...]
    """
    chan_locs = []

    # Get the unique channels (Using first 2 keys) and locations
    unique_channels = set([(tr.stats.channel[:2], tr.stats.location) for tr in st])

    for chan, loc in unique_channels:
        st_new = st.select(location=loc, channel=f"{chan}?")
        if len(st_new) > 3:
            # Check if all the sample rates are the same
            samples = [tr.stats.sampling_rate for tr in st_new]
            if len(set(samples)) > 1:
                # If they are different take the highest and resample with the others using interpolation
                st_new = st_new.select(sampling_rate=max(samples))
                st_new.merge(fill_value="interpolate")

        # Ensure traces all have the same length
        starttime_trim = max([tr.stats.starttime for tr in st_new])
        endtime_trim = min([tr.stats.endtime for tr in st_new])
        # Check that the start time is before the end time
        if starttime_trim > endtime_trim:
            continue
        st_new.trim(starttime_trim, endtime_trim)

        # Write the mseed file
        filename = f"{event_id}_{sta}_{chan}_{loc}.mseed"
        mseed_ffp = output_dir / filename
        st_new.write(mseed_ffp, format="MSEED")

        # Extend the chan and loc to the full channel codes and loc for each trace
        chan_locs.extend([(tr.stats.channel, tr.stats.location) for tr in st_new])
    return chan_locs
