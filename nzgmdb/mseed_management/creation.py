import warnings
import http.client
from pathlib import Path

from obspy import Stream
from obspy.taup import TauPyModel
from obspy.core.event import Origin
from obspy.geodetics import kilometers2degrees
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError

from nzgmdb.management.config import Config
from nzgmdb.mseed_management.AfshariStewart_2016_Ds import Afshari_Stewart_2016_Ds


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
    config = Config()
    vs30 = config.get_value("vs30")
    # Predict significant duration time from Afshari and Stewart (2016)
    ds, _ = Afshari_Stewart_2016_Ds(mag, rrup, vs30, None, "Ds595")

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
    """
    chan_locs = []
    for tr in st:
        # Check that the trace is not all 0's
        if not any(tr.data):
            continue

        loc, chan = tr.stats.location, tr.stats.channel[0:2]
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
        filename = f"{event_id}_{sta}_{chan}.mseed"
        mseed_ffp = output_dir / filename
        st_new.write(mseed_ffp, format="MSEED")

        # Add the channel and location to the list
        chan_locs.append((chan, loc))
    return chan_locs
