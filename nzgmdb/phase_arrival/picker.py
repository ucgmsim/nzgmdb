""" P-phase Picker """

import numpy as np
import scipy.linalg as alg
import scipy.signal as sig


def p_phase_picker(
    x: np.ndarray,
    dt: int,
    wftype: str,
    Tn: int = 0,
    xi: float = 0.6,
    nbins: int = 0,
    o: str = "to_peak",
) -> int:
    """
    P-phase picker based on the fixed-base viscously damped single-degree-of-freedom (SDF) oscillator model

    Parameters
    ----------
    x : np.ndarray
        The waveform data from a single component
    dt : int
        The sample rate of the data
    wftype : str
        The type of waveform data (strong motion 'sm', weak motion 'wm', or na)
    Tn : int
        The natural period of the oscillator in seconds
    xi : float
        The damping ratio of the oscillator
    nbins : int
        The number of bins to use for the histogram method
    o : str
        The method to use for the phase arrival picker ('to_peak' or 'full')

    Returns
    -------
    p_wave_index : int
        The index of the P-phase arrival
    """
    # Check Arguments!
    if "sm" in wftype.lower():
        wftype = "sm"
    elif "wm" in wftype.lower():
        wftype = "wm"
    elif "na" in wftype.lower():
        wftype = "na"
    else:
        return

    if Tn == 0:
        if dt <= 0.01:
            Tn = 0.01
        else:
            Tn = 0.1

    if nbins == 0:
        if dt <= 0.01:
            nbins = round(2 / dt)
        else:
            nbins = 200

    if "full" in o:
        o = "full"
    elif "to_peak" in o:
        o = "to_peak"
    else:
        return

    if "wm" in wftype.lower():
        filtflag = 1
        flp = 0.1
        fhp = 10.0
    elif "sm" in wftype.lower():
        # Strong-motion low- and high-pass corner frequencies in Hz
        filtflag = 1
        flp = 0.1
        fhp = 20.0
    elif "na" in wftype.lower():
        # No bandpass filter will be applied
        filtflag = 0
        # detrend waveform
        x_d = sig.detrend(x, axis=0)

    # Normalize input to prevent numerical instability from very low amplitudes
    x = x / np.max(np.abs(x))

    # Bandpass filter and detrend waveform
    if filtflag != 0:
        x_f = butter_bandpass_filter(x, flp, fhp, dt, 4)
        x_d = sig.detrend(x_f, axis=0)

    if "to_peak" in o:
        ind_peak = np.nonzero(np.abs(x_d) == np.max(np.abs(x_d)))
        xnew = x_d[0 : ind_peak[0][0]]
    elif "full" in o:
        xnew = x_d

    # Construct a fixed-base viscously damped SDF oscillator
    omegan = 2 * np.pi / Tn  # natural frequency in radian/second
    C = 2 * xi * omegan  # viscous damping term
    K = omegan**2  # stiffness term
    y = np.zeros((2, len(xnew)))  # response vector

    # Solve second-order ordinary differential equation of motion
    A = np.array([[0, 1], [-K, -C]])
    Ae = alg.expm(A * dt)
    AeB = np.dot(alg.lstsq(A, (Ae - np.identity(2)))[0], np.array((0, 1)))

    for k in range(1, len(xnew)):
        y[:, k] = np.dot(Ae, y[:, k - 1]) + AeB * xnew[k]

    veloc = y[1, :]  # relative velocity of mass
    Edi = np.dot(
        2 * xi * omegan, np.power(veloc, 2)
    )  # integrand of viscous damping energy

    # Apply histogram method
    levels, histogram, bins = state_level(Edi, nbins)
    locs = np.nonzero(Edi > levels[0])[0]
    indx = np.nonzero(np.multiply(xnew[0 : locs[0] - 1], xnew[1 : locs[0]]) < 0)[
        0
    ]  # get zero crossings
    TF = indx.size

    # Update first onset
    if TF != 0:
        p_wave_index = (indx[TF - 1] + 1) * dt
    else:
        levels, histogram, bins = state_level(Edi, np.ceil(nbins / 2))  # try nbins/2
        locs = np.nonzero(Edi > levels[0])[0]
        indx = np.nonzero(np.multiply(xnew[0 : locs[0] - 1], xnew[1 : locs[0]]) < 0)[
            0
        ]  # get zero crossings
        TF = indx.size
        if TF != 0:
            p_wave_index = (indx[TF - 1] + 1) * dt
        else:
            p_wave_index = -1
    return int(np.round(p_wave_index))


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5):
    """
    Butterworth Bandpass Filter Design

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency
    highcut : float
        Upper cutoff frequency
    fs : float
        Sampling frequency
    order : int
        Filter order
    """
    nyq = 1 / (2 * fs)  # Nyquist frequency
    Wn = [lowcut / nyq, highcut / nyq]  # Butterworth bandpass non-dimensional frequency
    b, a = sig.butter(order, Wn, btype="bandpass")
    return b, a


def butter_bandpass_filter(
    data: np.array, lowcut: float, highcut: float, fs: float, order: int = 4
):
    """
    Butterworth Acausal Bandpass Filter

    Parameters
    ----------
    data : np.array
        Input data
    lowcut : float
        Lower cutoff frequency
    highcut : float
        Upper cutoff frequency
    fs : float
        Sampling frequency
    order : int
        Filter order

    Returns
    -------
    y : np.array
        Filtered data
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = sig.filtfilt(b, a, data, axis=0)
    return y


def state_level(y: np.array, n: int):
    """
    Compute the state levels for the histogram method

    Parameters
    ----------
    y: array_like
    n: int

    Returns
    -------
    levels: array_like
    histogram: array_like
    bins: array_like
    """
    ymax = np.amax(y)
    ymin = np.min(y) - np.finfo(float).eps

    # Compute Histogram
    idx = np.ceil(n * (y - ymin) / (ymax - ymin))
    condition = np.logical_and(idx >= 1, idx <= n)
    idx = np.extract(condition, idx)
    s = (int(n), 1)
    histogram = np.zeros(s)
    for i in range(1, np.size(idx)):
        histogram[int(idx[i]) - 1] = histogram[int(idx[i]) - 1] + 1

    # Compute Center of Each Bin
    ymin = np.min(y)
    Ry = ymax - ymin
    dy = Ry / n
    bins = ymin + (np.arange(1, n) - 0.5) * dy

    # Compute State Levels
    nz = np.nonzero(histogram)[0]  # indices
    iLowerRegion = nz[0]
    iUpperRegion = nz[np.size(nz) - 1]

    iLow = iLowerRegion
    iHigh = iUpperRegion

    # Define the lower and upper histogram regions halfway
    # between the lowest and highest nonzero bins.
    lLow = iLow
    lHigh = iLow + np.floor((iHigh - iLow) / 2)
    uLow = iLow + np.floor((iHigh - iLow) / 2)
    uHigh = iHigh

    # Upper and lower histograms
    lHist = histogram[int(lLow) : int(lHigh)]
    uHist = histogram[int(uLow) : int(uHigh)]

    levels = np.zeros(2)
    iMax = np.argmax(lHist[1, :])
    iMin = np.argmax(uHist)
    levels[0] = ymin + dy * (lLow + iMax + 0.5)
    levels[1] = ymin + dy * (uLow + iMin + 0.5)

    return levels, histogram, bins
