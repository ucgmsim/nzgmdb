from obspy.core.stream import Stream
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import FDSNNoDataException

from nzgmdb.management import config as cfg


def basic_manipulation(mseed: Stream):
    """
    Basic manipulation of the waveform data

    Parameters
    ----------
    mseed : Stream
        The waveform data

    Returns
    -------
    mseed : Stream or None
        The processed waveform data, or None if the processing failed
    """
    # Small Processing
    mseed.detrend('demean')
    mseed.detrend()
    # Get the inventory information
    station = mseed[0].stats.station
    location = mseed[0].stats.location

    # Get Station Information from geonet clients
    # Fetching here instead of passing the inventory object as searching for the station, network, and channel
    # information takes a long time as it's implemented in a for loop
    try:
        client_NZ = FDSN_Client("GEONET")
        inv = client_NZ.get_stations(level="response", network="NZ", station=station, location=location)
    except FDSNNoDataException:
        try:
            client_IU = FDSN_Client("IRIS")
            inv = client_IU.get_stations(level="response", network="IU", station=station, location=location)
        except FDSNNoDataException:
            print(f"Failed to find inventory information")
            return None

    # Add the response (Same for all channels)
    # this is done so that the sensitivity can be removed otherwise it tries to find the exact same channel
    # which can fail when including the inventory information
    response = next(cha.response for sta in inv.networks[0] for cha in sta.channels)
    for tr in mseed:
        tr.stats.response = response

    try:
        mseed = mseed.remove_sensitivity()
    except:
        print(f"Failed to remove sensitivity")
        return None

    # Get constant gravity (g)
    config = cfg.Config()
    g = config.get_value("g")

    # Divide each trace data by g
    for tr in mseed:
        tr.data /= g

    return mseed
