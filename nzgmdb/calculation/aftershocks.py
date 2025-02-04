import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
from shapely import MultiPoint, Polygon, affinity, geometry
import alphashape
from scipy.spatial import ConvexHull

from abwd_declust.abwd_declust_v2_1 import abwd_crjb, abwd_rclosestp2h, abwd_rclosestp2p


import math


def ftd(time, tbin):
    """
    calculates the cumulative number of events in
    each time bin to identify possible aftershock sequences
    """

    mintime = math.floor(min(time / tbin)) * tbin  # Lowest time bin
    maxtime = math.ceil(max(time / tbin)) * tbin  # Highest time bin
    ti = np.arange(mintime, maxtime + tbin, tbin)  # Sequence of time bins
    nbt = len(ti)  # No. of time bins
    cumnbt = np.zeros(
        nbt
    )  # Pre-allocate array for cumulative no. of events in time bin and higher

    # Get cumulative no. of events in mag bin and higher
    for i in range(nbt):
        cumnbt[i] = np.where(time > ti[i] - tbin / 2)[0].shape[0]

    # Get no. of events in each time bin:
    nbt = abs(np.diff(np.append(cumnbt, 0)))
    cumnbt = np.cumsum(nbt)

    return (
        ti,
        nbt,
        cumnbt,
    )  # Return time bins, no. of events in bin, and cumulative no. of events


def fmd(mag, mbin):
    """
    calculates the cumulative number of events in
    each magnitude bin to identify possible aftershock sequences
    and determine overall poissonian behaviour
    """
    minmag = math.floor(min(mag / mbin)) * mbin  # Lowest magnitude bin
    maxmag = math.ceil(max(mag / mbin)) * mbin  # Highest magnitude bin
    mi = np.arange(minmag, maxmag + mbin, mbin)  # Sequence of magnitude bins
    nbm = len(mi)  # No. of magnitude bins
    cumnbmag = np.zeros(
        nbm
    )  # Pre-allocate array for cumulative no. of events in mag bin and higher

    # Get cumulative no. of events in mag bin and higher
    for i in range(nbm):
        cumnbmag[i] = np.where(mag > mi[i] - mbin / 2)[0].shape[0]

    # Get no. of events in each mag bin:
    nbmag = abs(np.diff(np.append(cumnbmag, 0)))

    return (
        mi,
        nbmag,
        cumnbmag,
    )  # Return magnitude bins, no. of events in bin, and cumulative no. of events


def decimalyear(catalog_org):
    from datetime import datetime as dt
    import time

    def toYearFraction(date):
        def sinceEpoch(date):  # returns seconds since epoch
            return time.mktime(date.timetuple())

        s = sinceEpoch

        year = date.year
        startOfThisYear = dt(year=year, month=1, day=1)
        startOfNextYear = dt(year=year + 1, month=1, day=1)

        yearElapsed = s(date) - s(startOfThisYear)
        yearDuration = s(startOfNextYear) - s(startOfThisYear)
        fraction = yearElapsed / yearDuration

        return date.year + fraction

    time_pd = pd.to_datetime(catalog_org.datetime, format="ISO8601")
    time_pd.reset_index(drop=True, inplace=True)
    dtime_orig = []

    for i in range(len(time_pd)):
        temp_orig = toYearFraction(time_pd[i])
        dtime_orig.append(temp_orig)

    return dtime_orig


# run the closest model
abwd_declust_v2_1.abwd_rclosestp2p(
    source_table,
)
