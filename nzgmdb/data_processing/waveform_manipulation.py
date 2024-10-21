import numpy as np
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.stream import Stream
from scipy import integrate, signal

from nzgmdb.management import config as cfg
from nzgmdb.management import custom_errors


def initial_preprocessing(
    mseed: Stream, apply_taper: bool = True, apply_zero_padding: bool = True
):
    """
    Basic pre-processing of the waveform data
    This performs the following:
    - Demean and Detrend the data
    - Taper the data by the taper_fraction in the config to each end (5% default)
    - Perform zero padding
    - Rotate the data to NEZ
    - Remove the sensitivity using the inventory response information if possible
    - Divide the data by the constant gravity (g)

    Parameters
    ----------
    mseed : Stream
        The waveform data
    apply_taper : bool, optional
        Whether to apply the tapering, by default True
    apply_zero_padding : bool, optional
        Whether to apply zero padding, by default True

    Returns
    -------
    mseed : Stream
        The processed waveform data

    Raises
    ------
    InventoryNotFoundError
        If no inventory information is found for the station and location pair
    SensitivityRemovalError
        If the sensitivity removal fails
    RotationError
        If the rotation fails
    """
    # Small Processing
    mseed.detrend("demean")
    mseed.detrend("linear")

    # Load config
    config = cfg.Config()
    no_response_stations = config.get_value("no_response_stations")
    no_response_conversion = config.get_value("no_response_conversion")
    taper_fraction = config.get_value("taper_fraction")
    zero_padding_time = config.get_value("zero_padding_time")

    if apply_taper:
        # Taper the data by the taper_fraction
        mseed.taper(taper_fraction, side="both", max_length=5)

    if apply_zero_padding:
        # Perform zero-padding
        for tr in mseed:
            tr_starttime_trim = tr.stats.starttime - zero_padding_time
            tr_endtime_trim = tr.stats.endtime + zero_padding_time
            tr.trim(tr_starttime_trim, tr_endtime_trim, pad=True, fill_value=0)

    # Get the inventory information
    station = mseed[0].stats.station
    location = mseed[0].stats.location

    if station in no_response_stations:
        # Divide trace counts by 10^6 to convert to units g as there is no response for these stations
        for tr in mseed:
            tr.data /= no_response_conversion
    else:
        # Get Station Information from geonet clients
        # Fetching here instead of passing the inventory object as searching for the station, network, and channel
        # information takes a long time as it's implemented in a for loop
        try:
            client_NZ = FDSN_Client("GEONET")
            inv = client_NZ.get_stations(
                level="response", network="NZ", station=station, location=location
            )
        except FDSNNoDataException:
            try:
                client_IU = FDSN_Client("IRIS")
                inv = client_IU.get_stations(
                    level="response", network="IU", station=station, location=location
                )
            except FDSNNoDataException:
                raise custom_errors.InventoryNotFoundError(
                    f"No inventory information found for station {station} with location {location}"
                )

        # Rotate
        try:
            mseed.rotate("->ZNE", inventory=inv)
        except (
            Exception
        ):  # Due to obspy raising an Exception instead of a specific error
            # Error for no matching channel metadata found
            raise custom_errors.RotationError(
                f"Failed to rotate for station {station} with location {location}"
            )

        # Add the response (Same for all channels)
        # this is done so that the sensitivity can be removed otherwise it tries to find the exact same channel
        # which can fail when including the inventory information
        response = next(cha.response for sta in inv.networks[0] for cha in sta.channels)
        for tr in mseed:
            tr.stats.response = response

        try:
            mseed = mseed.remove_sensitivity()
        except ValueError:
            raise custom_errors.SensitivityRemovalError(
                f"Failed to remove sensitivity for station {station} with location {location}"
            )

        # Get constant gravity (g)
        g = config.get_value("g")

        # Divide each trace data by g
        for tr in mseed:
            tr.data /= g

    return mseed


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int):
    """
    Create a butter bandpass filter

    Parameters
    ----------
    lowcut : float
        The lowcut frequency
    highcut : float
        The highcut frequency
    fs : float
        The sampling frequency
    order : int
        The order of the butter bandpass

    Returns
    -------
    sos : np.ndarray
        Array of second-order filter coefficients
    """
    nyquist_frequency = 0.5 * fs
    low = lowcut / nyquist_frequency
    high = highcut / nyquist_frequency
    sos = signal.butter(order, [low, high], btype="band", analog=False, output="sos")
    return sos


def butter_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int
):
    """
    Apply a butter bandpass filter to the data

    Parameters
    ----------
    data : np.ndarray
        The data to filter
    lowcut : float
        The lowcut frequency
    highcut : float
        The highcut frequency
    fs : float
        The sampling frequency
    order : int
        The order of the butter bandpass

    Returns
    -------
    y_sos : np.ndarray
        The digital filtered data ouptut
    """
    sos = butter_bandpass(lowcut, highcut, fs, order)
    y_sos = signal.sosfilt(sos, data)
    return y_sos


def high_and_low_cut_processing(
    mseed: Stream, dt: float, fmin: float = None, fmax: float = None
):
    """
    Process the waveform data by using the highcut and lowcut for the butter bandpass filter
    This processing performs the following:
    - Apply the bandpass filter
    - Remove the zero padding
    - Calculate the velocity and displacement
    - Fit a six-order polynomial to the displacement series
    - Subtract the 2nd derivative of the polynomial from the original acc series

    Parameters
    ----------
    mseed : Stream
        The waveform data
    dt : float
        The time step of the data
    fmin : float (optional)
        The minimum frequency to cut off at
        When not provided will use the default value in the config
    fmax : float (optional)
        The maximum frequency to cut off at
        When not provided will use 1 / (2.5 * dt)

    Returns
    -------
    acc_bb_000 : np.array
        The processed data for the 000 component
    acc_bb_090 : np.array
        The processed data for the 090 component
    acc_bb_ver : np.array
        The processed data for the vertical component

    Raises
    ------
    LowcutHighcutError
        If the lowcut frequency is greater than the highcut frequency
    ComponentSelectionError
        If no N, E, X, or Y components are found in the mseed
    """

    # Load config variables
    config = cfg.Config()
    g = config.get_value("g")
    order = config.get_value("order_default")
    poly_order = config.get_value("poly_order_default")

    # Determine the high and low cut frequencies
    highcut = fmax or 1 / (2.5 * dt)
    lowcut = config.get_value("low_cut_default") if fmin is None else fmin / 1.25

    # Check if the lowcut is greater than the highcut
    if lowcut > highcut:
        raise custom_errors.LowcutHighcutError(
            f"Lowcut frequency {lowcut} is greater than the highcut frequency {highcut}"
        )

    # Set fs
    fs = 1.0 / dt

    # Get the traces for the different components
    try:
        acc_000 = mseed.select(channel="*N")[0]
        acc_090 = mseed.select(channel="*E")[0]
    except IndexError:
        try:
            # If the N and E components are not found, try the X and Y components
            acc_000 = mseed.select(channel="*Y")[0]
            acc_090 = mseed.select(channel="*X")[0]
        except IndexError:
            raise custom_errors.ComponentSelectionError(
                "No N, X or E, Y components found in the mseed"
            )
    acc_ver = mseed.select(channel="*Z")[0]

    # Apply the bandpass filter
    acc_bb_000 = butter_bandpass_filter(acc_000, lowcut, highcut, fs, order)
    acc_bb_090 = butter_bandpass_filter(acc_090, lowcut, highcut, fs, order)
    acc_bb_ver = butter_bandpass_filter(acc_ver, lowcut, highcut, fs, order)

    # Remove Zero padding
    for tr in mseed:
        tr_starttime_trim = tr.stats.starttime
        tr_endtime_trim = tr.stats.endtime
        tr.trim(tr_starttime_trim, tr_endtime_trim)

    # Calculate the velocity
    vel_000 = (
        integrate.cumulative_trapezoid(y=acc_bb_000, dx=dt, initial=0.0) * g / 10.0
    )
    vel_090 = (
        integrate.cumulative_trapezoid(y=acc_bb_090, dx=dt, initial=0.0) * g / 10.0
    )
    vel_ver = (
        integrate.cumulative_trapezoid(y=acc_bb_ver, dx=dt, initial=0.0) * g / 10.0
    )

    # Calculate the displacement
    disp_000 = integrate.cumulative_trapezoid(y=vel_000, dx=dt, initial=0.0)
    disp_090 = integrate.cumulative_trapezoid(y=vel_090, dx=dt, initial=0.0)
    disp_ver = integrate.cumulative_trapezoid(y=vel_ver, dx=dt, initial=0.0)

    # The following steps were added to align the processing with NGA-West
    # Fit six-order polynomial to the displacement series
    coeff_000 = np.polyfit(np.arange(len(disp_000)), disp_000, poly_order)
    coeff_090 = np.polyfit(np.arange(len(disp_090)), disp_090, poly_order)
    coeff_ver = np.polyfit(np.arange(len(disp_ver)), disp_ver, poly_order)

    # Find the second derivative of the coefficients
    coeff_000_2nd = np.polyder(coeff_000, 2)
    coeff_090_2nd = np.polyder(coeff_090, 2)
    coeff_ver_2nd = np.polyder(coeff_ver, 2)

    # Generate polynomial values from the coefficients
    poly_000 = np.polyval(coeff_000_2nd, np.arange(len(acc_bb_000)))
    poly_090 = np.polyval(coeff_090_2nd, np.arange(len(acc_bb_000)))
    poly_v = np.polyval(coeff_ver_2nd, np.arange(len(acc_bb_000)))

    # Subtract the polynomial from the original acc series
    acc_bb_000 -= poly_000
    acc_bb_090 -= poly_090
    acc_bb_ver -= poly_v

    return acc_bb_000, acc_bb_090, acc_bb_ver
