"""
Generate a HTML report for the NZGMDB database comparing to a previous version / giving a summary of the current state.
"""

import base64
from io import BytesIO
from pathlib import Path
from collections.abc import Sequence
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import StrEnum, Enum
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

import oq_wrapper as oqw
from nzgmdb.management import file_structure, data_registry


# constants.py
class ObsDataSource(StrEnum):
    NZGMDB = "NZGMDB"
    NGAWest2 = "NGAWest2"
    NGASubduction = "NGASubduction"


class NZGMDBVersion(StrEnum):
    v3p0 = "v3.0"
    v3p4 = "v3.4"
    v4p0 = "v4.0"
    v4p1 = "v4.1"
    v4p2 = "v4.2"
    v4p3 = "v4.3"
    v4p3_final = "v4.3_final"


class TectonicType(StrEnum):
    CRUSTAL = "crustal"
    SUBDUCTION_INTERFACE = "subduction_interface"
    SUBDUCTION_SLAB = "subduction_slab"
    OUTER_RISE = "outer_rise"
    MANTLE = "mantle"
    UNKNOWN = "unknown"


class IMSet(StrEnum):
    pSA = "pSA"
    all = "all"


class RankingMethod(Enum):
    emp_cMVN = 1
    sim_cMVN = 2

    # Same as sim_cMVN but uses correlation coefficients
    # from the empirical model
    sim_cMVN_emp_corr = 3

    ml_prob = 4
    ml_prob_per_im = 5


OQ_INPUT_COLUMNS = [
    "vs30",
    "rrup",
    "rjb",
    "z1pt0",
    "mag",
    "rake",
    "dip",
    "vs30measured",
    "ztor",
    "rx",
    "hypo_depth",
]


PERIODS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.12,
    0.15,
    0.17,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
    1.2,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.5,
    10.0,
]
PSA_KEYS = [f"pSA_{x}" for x in PERIODS]

NON_PSA_IMs = ["PGV", "PGA", "AI", "CAV", "Ds575", "Ds595"]
IMs = NON_PSA_IMs + PSA_KEYS


# Define a function to display both percentage and absolute value
def func(pct, allvals):
    absolute = round(pct / 100.0 * np.sum(allvals))
    normal = "{:.1f}%\n({:d})".format(pct, absolute)
    return normal if pct > 5 else ""


import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_pie_chart_1p(full_labels: list[str], full_sizes: list[int], title: str):
    non_zero_indices = [i for i, size in enumerate(full_sizes) if size != 0]
    sizes = [full_sizes[i] for i in non_zero_indices]
    total_records = sum(sizes)
    pie_labels = [
        full_labels[i] if sizes[i] / total_records * 100 > 5 else ""
        for i in non_zero_indices
    ]
    labels = [full_labels[i] for i in non_zero_indices]
    colors = [plt.cm.tab20.colors[i] for i in non_zero_indices]

    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=pie_labels,
        colors=colors,
        autopct=lambda pct: func(pct, sizes),
        startangle=270,
    )

    ax.set_title(f"{title} ({total_records} total records)")
    ax.axis("equal")

    small_values_indices = [
        i for i, size in enumerate(sizes) if size / total_records * 100 < 5
    ]
    if small_values_indices:
        small_labels = [
            labels[i] + f" <5% ({sizes[i]})" if sizes[i] != 0 else " (0%)"
            for i in small_values_indices
        ]
        small_colors = [colors[i] for i in small_values_indices]
        legend_elements = [
            Patch(facecolor=color, label=label)
            for color, label in zip(small_colors, small_labels)
        ]
        ax.legend(
            handles=legend_elements,
            title="Categories < 5%",
            bbox_to_anchor=(1, 0.8),
            loc="center left",
        )

    return fig


# MLT
def numpy_str_join(sep: str, *arrays: str | Sequence[str]) -> np.ndarray[str]:
    """
    Joins multiple string arrays together using the specified separator.
    Also support joining of string values, or a combination of both.

    Parameters
    ----------
    sep: str
        The separator to use
    arrays: string value or string arrays
        The arrays (or string values) to join together

    Returns
    -------
    np.ndarray
        The joined string array

    Examples
    --------
    Example 1: Joining two arrays
    >>> numpy_str_join("_", ["a", "b"], ["c", "d"])
    array(['a_c', 'b_d'], dtype='<U3')

    Example 2: Combination of string value and arrays
    >>> arr1 = np.array(["a", "b"])
    >>> numpy_str_join("_", arr1, "c", ["d", "e"])
    array(['a_c_d', 'b_c_e'], dtype='<U5')
    """
    result = arrays[0]
    for cur_array in arrays[1:]:
        result = np.char.add(result, sep)
        result = np.char.add(result, cur_array)

    return result


def get_fig_axes(
    n_subplots: int,
    n_cols: int,
    n_rows: int,
    ind_figsize: tuple[int, int],
    dpi: int | None = None,
):
    """
    Given the number of desired subplots, and either the desired
    number of columns or rows, will return the figure and
    appropriate number of axes objects.

    Note I:One of n_cols or n_rows must be specified,
    the other has to be set to -1.

    Note II: The returned number of axes can be
    larger than the specified number of subplots.

    Parameters
    ----------
    n_subplots: int
        The number of subplots.
    n_cols:
        The number of columns.
        Set to -1 if n_rows is to be computed.
    n_rows:
        The number of rows.
        Set to -1 if n_cols is to be computed.
    ind_figsize:
        The individual figure size for each subplot

    Returns
    -------
    fig: plt.Figure
        The figure object
    axs: List[plt.Axes]
        The axes objects

    Raises
    ------
    ValueError
        If both n_cols and n_rows are specified

    Examples
    --------
    >>> fig, axs = get_fig_axes(4, 2, -1, (5, 5))
    >>> len(axs)
    4
    >>> fig, axs = get_fig_axes(5, -1, 2, (5, 5))
    >>> len(axs)
    6
    >>> fig, axs = get_fig_axes(5, -1, -1, (5, 5))
    Traceback (most recent call last):
        ...
    ValueError: One of n_cols/n_rows must be specified
    >>> fig, axs = get_fig_axes(6, 2, 3, (5, 5))
    >>>len(axs)
    6
    """

    if n_cols == -1 or n_rows == -1:
        if n_cols > 0:
            n_rows = int(np.ceil(n_subplots / n_cols))
        elif n_rows > 0:
            n_cols = int(np.ceil(n_subplots / n_rows))
        else:
            raise ValueError("One of n_cols/n_rows must be specified")

    figsize = (n_cols * ind_figsize[0], n_rows * ind_figsize[1])
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)

    if n_subplots == 1:
        axs = (axs,)
    else:
        axs = list(axs.flatten())

    return fig, axs


# data_classes.py
class ObservedData:
    class EventColEnums(StrEnum):
        EVENT_ID = "event_id"
        MAG = "mag"
        DEPTH = "depth"
        RAKE = "rake"
        STRIKE = "strike"
        DIP = "dip"
        ZTOR = "ztor"
        TECT_TYPE = "tect_type"
        EVENT_LAT = "event_lat"
        EVENT_LON = "event_lon"

    class SiteColEnums(StrEnum):
        SITE_ID = "site_id"
        VS30 = "vs30"
        TSITE = "tsite"
        Z1P0 = "z1p0"
        Z2P5 = "z2p5"
        SITE_LAT = "site_lat"
        SITE_LON = "site_lon"

    class EventSiteColEnums(StrEnum):
        RJB = "rjb"
        RRUP = "rrup"
        RX = "rx"

    class OtherColEnums(StrEnum):
        FHP_HORIZONTAL = "HPF_h"
        """High-pass filter frequency for horizontal components"""
        FHP_VERTICAL = "HPF_v"
        """High-pass filter frequency for vertical components"""
        FHP = "fhp"
        """High-pass filter frequency"""
        TMAX = "tmax"
        """Maximum usable period"""
        FMIN = "fmin"
        """Minimum usable frequency"""
        FMIN_H1 = "fmin_h1"
        """Minimum usable frequency for horizontal component 1"""
        FMIN_H2 = "fmin_h2"
        """Minimum usable frequency for horizontal component 2"""
        FMIN_V = "fmin_v"
        """Minimum usable frequency for vertical component"""
        QUALITY_SCORE_H1 = "score_h1"
        QUALITY_SCORE_H2 = "score_h2"
        QUALITY_SCORE_V = "score_v"
        IS_GROUND_LEVEL = "is_ground_level"
        CHANNEL = "channel"

    SITE_COLS = list(SiteColEnums)
    EVENT_COLS = list(EventColEnums)
    EVENT_SITE_COLS = list(EventSiteColEnums)
    IM_COLUMNS = ["PGA", "PGV"]
    OTHER_COLUMNS = list(OtherColEnums)
    COLUMNS = SITE_COLS + EVENT_COLS + EVENT_SITE_COLS + IM_COLUMNS + OTHER_COLUMNS

    def __init__(
        self,
        record_df: pd.DataFrame,
        data_ffp: Path,
        data_source: ObsDataSource,
        nzgmdb_version: NZGMDBVersion = None,
    ):
        """
        Class for storing and handling observed data.

        Note: Instantiation of this class should be done through the class methods!

        Parameters
        ----------
        record_df: pd.DataFrame
            DataFrame containing the observed data, event and site information.
        data_ffp: Path
            File path to the data file.
        data_source: constants.ObsDataSource
            Source of the observed data.
        nzgmdb_version: constants.NZGMDBVersion
            Version of the NZGMDB data.
            Only applicable if the data source is NZGMDB, otherwise None.
        """
        self.record_df = record_df
        """DataFrame containing the observed data, event and site information"""
        self.data_ffp = data_ffp
        """Path to the data file"""

        self.data_source = data_source
        """Source of the observed data"""
        self.nzgmdb_version = nzgmdb_version
        """Version of the NZGMDB data. 
        Only applicable if the data source is NZGMDB, otherwise None."""

        # Cache variables
        self._sites = None
        self._events = None
        self._site_df = None
        self._event_df = None
        self._event_sites = None
        self._ims = None

    def __hash__(self):
        return hash(self.data_ffp)

    def __reset_cache(self):
        self._sites = None
        self._events = None
        self._site_df = None
        self._event_df = None
        self._event_sites = None
        self._ims = None

    def get_event_data(
        self, event_id: str, sites: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Gets data for the specified event and sites."""
        result_df = self.record_df[self.record_df["event_id"] == event_id]

        if sites is not None:
            result_df = result_df[result_df["site_id"].isin(sites)]

        return result_df.set_index("site_id")

    def __setitem__(self, column: str, value: np.ndarray | pd.Series):
        """Support adding of columns"""
        self.record_df[column] = value

    @property
    def ims(self):
        if self._ims is None:
            pSA_vals = [
                cur_col
                for cur_col in self.record_df.columns
                if cur_col.startswith("pSA")
            ]
            other_ims = [
                cur_im for cur_im in self.IM_COLUMNS if cur_im in self.record_df.columns
            ]
            self._ims = np.asarray(pSA_vals + other_ims)

        return self._ims

    @property
    def n_records(self):
        return self.record_df.shape[0]

    @property
    def sites(self):
        """All sites in the observed data."""
        if self._sites is None:
            self._sites = self.record_df.site_id.unique().astype(str)
        return self._sites

    @property
    def events(self):
        """All events in the observed data."""
        if self._events is None:
            self._events = self.record_df.event_id.unique().astype(str)
        return self._events

    @property
    def site_df(self):
        if self._site_df is None:
            self._site_df = (
                self.record_df[self.SITE_COLS]
                .drop_duplicates("site_id")
                .set_index("site_id")
                .rename(columns={"site_lat": "lat", "site_lon": "lon"})
            )
        return self._site_df

    @property
    def event_df(self):
        if self._event_df is None:
            self._event_df = (
                self.record_df[self.EVENT_COLS]
                .drop_duplicates("event_id")
                .set_index("event_id")
            ).rename(columns={"event_lat": "lat", "event_lon": "lon"})
        return self._event_df

    @property
    def event_sites(self):
        if self._event_sites is None:
            self._event_sites = {}
            for cur_event, cur_group in self.record_df.groupby("event_id"):
                self._event_sites[cur_event] = cur_group.site_id.unique().astype(str)
        return self._event_sites

    def drop_nan(self, subset: Sequence[str] = None, verbose: bool = False):
        """Drops any rows with NaN values."""
        cols = subset if subset else self.record_df.columns
        nan_mask = self.record_df.loc[:, cols].isna().any(axis=1)
        if np.count_nonzero(nan_mask) > 0:
            if verbose:
                print(f"Dropping {np.count_nonzero(nan_mask)} records with NaN values")
            self.record_df = self.record_df[~nan_mask]

            self.__reset_cache()
        return self

    def drop_events(self, events: Sequence[str]):
        """Drops all records associated with the specified events."""
        self.record_df = self.record_df[
            ~self.record_df[self.EventColEnums.EVENT_ID].isin(events)
        ]

        self.__reset_cache()
        return self

    def drop_duplicates(
        self, subset: Sequence[str] = None, sort_key: str = None, ascending: bool = True
    ):
        """
        Drops any duplicate rows.
        Allows for sorting of the dataframe first.
        """
        if sort_key is not None:
            self.record_df = self.record_df.sort_values(sort_key, ascending=ascending)
        self.record_df = self.record_df.drop_duplicates(subset=subset)

        self.__reset_cache()
        return self

    def metadata_filter(
        self,
        filter_dict: dict[str, tuple[float, float]],
    ):
        """
        Performs filtering on the record metadata.
        Does not return anything, but modifies the observed data instance in place.

        Parameters
        ----------
        filter_dict: dict
            Dictionary of key (column name) and
            value (tuple, bool) to filter on.
            E.g. {"mag": (5.0, 6.0), "rrup": (0.0, 10.0),
            "is_ground_level": True}
        """
        for cur_key, cur_filter in filter_dict.items():
            if isinstance(cur_filter, tuple):
                self.record_df = self.record_df[
                    (self.record_df[cur_key] >= cur_filter[0])
                    & (self.record_df[cur_key] <= cur_filter[1])
                ]
            elif isinstance(cur_filter, bool):
                self.record_df = self.record_df.loc[
                    self.record_df[cur_key] == cur_filter
                ]
            else:
                raise ValueError(f"Unknown filter type: {type(cur_filter)}")

        self.__reset_cache()
        return self

    def filter_record_ids(
        self,
        record_ids: np.ndarray[str],
    ):
        """
        Filters the observed data based on the provided record IDs.

        Parameters
        ----------
        record_ids: array of strings
            Record IDs to keep.
        """
        self.record_df = self.record_df[self.record_df.index.isin(record_ids)]
        self.__reset_cache()
        return self

    def apply_fmin_filter(self, fmin_col: str):
        """Applies fmin filtering to pSA"""
        max_usable_period = 1 / self.record_df[fmin_col]
        pSA_cols = [
            cur_col for cur_col in self.record_df.columns if cur_col.startswith("pSA")
        ]

        for cur_pSA_col in pSA_cols:
            cur_period = float(cur_pSA_col.split("_")[1])
            self.record_df[cur_pSA_col] = np.where(
                cur_period > max_usable_period, np.nan, self.record_df[cur_pSA_col]
            )

        return self

    def to_event_site_index(self):
        """
        Updates the index to be {event_id}_{site_id}

        Note: This will cause if there are duplicate event_id & site_id pairs.
        """
        index = numpy_str_join(
            "_",
            self.record_df["event_id"].values.astype(str),
            self.record_df["site_id"].values.astype(str),
        )
        self.record_df.index = index
        self.record_df = self.record_df.sort_index()

        return self

    @classmethod
    def from_nzgmdb_flat(
        cls,
        nzgmdb_flat_ffp: Path,
        version: NZGMDBVersion = None,
        event_site_id_index: bool = True,
    ):
        site_cols_map = {
            "sta": cls.SiteColEnums.SITE_ID,
            "Vs30": cls.SiteColEnums.VS30,
            "Tsite": cls.SiteColEnums.TSITE,
            "T0": cls.SiteColEnums.TSITE,
            "Z1.0": cls.SiteColEnums.Z1P0,
            "Z2.5": cls.SiteColEnums.Z2P5,
            "sta_lat": cls.SiteColEnums.SITE_LAT,
            "sta_lon": cls.SiteColEnums.SITE_LON,
        }
        event_map = {
            "evid": cls.EventColEnums.EVENT_ID,
            "mag": cls.EventColEnums.MAG,
            "rake": cls.EventColEnums.RAKE,
            "strike": cls.EventColEnums.STRIKE,
            "dip": cls.EventColEnums.DIP,
            "tect_class": cls.EventColEnums.TECT_TYPE,
            "ev_depth": cls.EventColEnums.DEPTH,
            "z_tor": cls.EventColEnums.ZTOR,
            "ev_lat": cls.EventColEnums.EVENT_LAT,
            "ev_lon": cls.EventColEnums.EVENT_LON,
        }
        event_site_map = {
            "r_jb": cls.EventSiteColEnums.RJB,
            "r_rup": cls.EventSiteColEnums.RRUP,
            "r_x": cls.EventSiteColEnums.RX,
        }
        other_map = {
            "HPF_h": cls.OtherColEnums.FHP_HORIZONTAL,
            "HPF_v": cls.OtherColEnums.FHP_VERTICAL,
            "fmin_X": cls.OtherColEnums.FMIN_H1,
            "fmin_Y": cls.OtherColEnums.FMIN_H2,
            "fmin_Z": cls.OtherColEnums.FMIN_V,
            "score_X": cls.OtherColEnums.QUALITY_SCORE_H1,
            "score_Y": cls.OtherColEnums.QUALITY_SCORE_H2,
            "score_Z": cls.OtherColEnums.QUALITY_SCORE_V,
            "fmin_mean_X": cls.OtherColEnums.FMIN_H1,
            "fmin_mean_Y": cls.OtherColEnums.FMIN_H2,
            "fmin_mean_Z": cls.OtherColEnums.FMIN_V,
            "score_mean_X": cls.OtherColEnums.QUALITY_SCORE_H1,
            "score_mean_Y": cls.OtherColEnums.QUALITY_SCORE_H2,
            "score_mean_Z": cls.OtherColEnums.QUALITY_SCORE_V,
            "is_ground_level": cls.OtherColEnums.IS_GROUND_LEVEL,
            "chan": cls.OtherColEnums.CHANNEL,
            "fmin_max": cls.OtherColEnums.FMIN,
        }
        mapping_dict = site_cols_map | event_map | event_site_map | other_map

        tect_type_mapping = {
            "Crustal": TectonicType.CRUSTAL,
            "Interface": TectonicType.SUBDUCTION_INTERFACE,
            "Slab": TectonicType.SUBDUCTION_SLAB,
            "Outer-rise": TectonicType.OUTER_RISE,
            "Undetermined": TectonicType.UNKNOWN,
            np.nan: TectonicType.UNKNOWN,
        }

        # Determine the version if not provided
        if version is None:
            try:
                version = NZGMDBVersion(nzgmdb_flat_ffp.parent.parent.name)
            except ValueError as e:
                raise ValueError(
                    f"Could not determine version of NZGMDB "
                    f"from {nzgmdb_flat_ffp.parent.parent.name}"
                ) from e

        # Load
        if version in [NZGMDBVersion.v3p4, NZGMDBVersion.v3p0]:
            record_df = pd.read_csv(
                nzgmdb_flat_ffp, dtype={"evid": str}, index_col="gmid", engine="c"
            ).sort_index()

            # Renaming
            record_df = record_df.rename(columns=mapping_dict)
            record_df.index.name = "record_id"

            # Convert index
            if event_site_id_index:
                index = numpy_str_join(
                    "_",
                    record_df["event_id"].values.astype(str),
                    record_df["site_id"].values.astype(str),
                )
                record_df.index = index
                record_df = record_df.sort_index()

            # The GMC fmin in version 3.4 is used to select fHP, hence
            # the actual fmin is not the GMC fmin values
            if (
                cls.OtherColEnums.FMIN_H1 in record_df.columns
                and cls.OtherColEnums.FMIN_H2 in record_df.columns
            ):
                # Rotd50 does not contain vertical GMC fmin, hack this in...
                fmin_v = pd.read_csv(
                    nzgmdb_flat_ffp.parent
                    / nzgmdb_flat_ffp.name.replace("rotd50", "ver"),
                    dtype={"evid": str},
                    index_col="gmid",
                    engine="c",
                ).sort_index()

                # Convert index, as gmid is incorrect in 3.4
                index = numpy_str_join(
                    "_",
                    fmin_v["evid"].values.astype(str),
                    fmin_v["sta"].values.astype(str),
                )
                fmin_v.index = index
                fmin_v = fmin_v.sort_index()
                assert fmin_v.index.equals(record_df.index)

                record_df[cls.OtherColEnums.FHP] = np.max(
                    np.stack(
                        (
                            record_df[cls.OtherColEnums.FMIN_H1].values,
                            record_df[cls.OtherColEnums.FMIN_H2].values,
                            fmin_v["fmin_mean_Z"].values,
                        ),
                        axis=1,
                    ),
                    axis=1,
                )

                record_df[cls.OtherColEnums.FMIN] = (
                    record_df[cls.OtherColEnums.FHP] / 1.25
                )
                record_df = record_df.drop(
                    columns=[
                        cls.OtherColEnums.FMIN_H1,
                        cls.OtherColEnums.FMIN_H2,
                        cls.OtherColEnums.FMIN_V,
                    ],
                    errors="ignore",
                )
        else:
            record_df = pd.read_csv(
                nzgmdb_flat_ffp,
                dtype={"evid": str, "loc": str},
                engine="c",
                index_col="record_id",
            ).sort_index()

            if version is NZGMDBVersion.v4p3_final:
                # Add fmin column
                record_df[cls.OtherColEnums.FMIN] = (
                    record_df[cls.OtherColEnums.FHP_HORIZONTAL] * 1.25
                )

            # Renaming
            record_df = record_df.rename(columns=mapping_dict)

        # Tectonic type
        record_df[cls.EventColEnums.TECT_TYPE] = record_df[
            cls.EventColEnums.TECT_TYPE
        ].map(tect_type_mapping)

        # Drop any columns not of interest
        im_cols = IMs
        cols = record_df.columns[record_df.columns.isin(cls.COLUMNS + im_cols)]
        record_df = record_df[cols]

        return cls(record_df, nzgmdb_flat_ffp, ObsDataSource.NZGMDB, version)


# data.py
OBS_DATA_OQ_COLS_MAPPING = {
    ObservedData.SiteColEnums.VS30: "vs30",
    ObservedData.EventSiteColEnums.RRUP: "rrup",
    ObservedData.EventSiteColEnums.RJB: "rjb",
    ObservedData.SiteColEnums.Z1P0: "z1pt0",
    ObservedData.EventColEnums.MAG: "mag",
    ObservedData.EventColEnums.RAKE: "rake",
    ObservedData.EventColEnums.DIP: "dip",
    ObservedData.EventColEnums.ZTOR: "ztor",
    ObservedData.EventSiteColEnums.RX: "rx",
    ObservedData.EventColEnums.DEPTH: "hypo_depth",
}


def load_obs_nzgmdb(nzgmdb_ffp: Path, version: NZGMDBVersion = None):
    """
    Load the observed data from NZGMDB and performs the
    necessary preparation steps, depending
    on the version.

    Parameters
    ----------
    nzgmdb_ffp: Path
        Path to the NZGMDB flat file
    version: NZGMDBVersion, optional
        Version of the NZGMDB data.

    Returns
    -------
    obs_data: ObservedData
        Observed data object
    """
    obs_data = ObservedData.from_nzgmdb_flat(nzgmdb_ffp, version)
    assert obs_data.data_source is ObsDataSource.NZGMDB

    # Filter out nan values
    obs_data = obs_data.drop_nan()

    # Perform filtering in line with Lee et al. (2024) Table 4
    obs_data = obs_data.metadata_filter(dict(rrup=(0, 500), is_ground_level=True))

    # Crustal, Mag >= 3.5 and rrup <= 300
    records_to_keep = list(
        obs_data.record_df.loc[
            (obs_data.record_df.tect_type == TectonicType.CRUSTAL)
            & (obs_data.record_df.mag >= 3.5)
            & (obs_data.record_df.rrup <= 300)
        ].index.values.astype(str)
    )

    # Subduction Interface, Mag >= 4.5 and rrup <= 500
    records_to_keep += list(
        obs_data.record_df.loc[
            (obs_data.record_df.tect_type == TectonicType.SUBDUCTION_INTERFACE)
            & (obs_data.record_df.mag >= 4.5)
        ].index.values.astype(str)
    )

    # Subduction Slab, Mag >= 4.5 and rrup <= 500
    records_to_keep += list(
        obs_data.record_df.loc[
            (obs_data.record_df.tect_type == TectonicType.SUBDUCTION_SLAB)
            & (obs_data.record_df.mag >= 4.5)
        ].index.values.astype(str)
    )

    obs_data.filter_record_ids(records_to_keep)

    # Drop duplicates
    # Use the one with the smaller fmin
    obs_data = obs_data.drop_duplicates(
        ["event_id", "site_id"],
        sort_key=ObservedData.OtherColEnums.FMIN,
        ascending=True,
    )

    # Convert to event_site index
    obs_data = obs_data.to_event_site_index()

    # Apply fmin
    obs_data = obs_data.apply_fmin_filter(ObservedData.OtherColEnums.FMIN)

    return obs_data


def compute_nzgmdb_emp_gm_params(
    obs_data: ObservedData,
    events: Sequence[str] = None,
    periods: Sequence[float] = PERIODS,
):
    """
    Computes the empirical GMM parameters for all
    specified sites and sources, based on inputs
    from NZGMDB

    Parameters
    ----------
    nzgmdb_flat_ffp: Path
        Path to the NZGMDB flat file

    Returns
    -------
    result_df: DataFrame
        The empirical GMM parameters for PGA
        and the default set of pSA periods
    """
    # Create rupture dataframe
    columns = [
        ObservedData.EventColEnums.EVENT_ID,
        ObservedData.SiteColEnums.SITE_ID,
        ObservedData.SiteColEnums.SITE_LON,
        ObservedData.SiteColEnums.SITE_LAT,
        ObservedData.EventColEnums.TECT_TYPE,
    ] + list(OBS_DATA_OQ_COLS_MAPPING.keys())
    # rupture_df = pd.read_csv(nzgmdb_flat_ffp, index_col=0)[columns]
    rupture_df = obs_data.record_df[columns].copy(True)

    # Filter events
    if events is not None:
        rupture_df = rupture_df.loc[rupture_df.event.isin(events)]

    # Convert Z1.0 to kilometres
    rupture_df[ObservedData.SiteColEnums.Z1P0] /= 1000

    # Rename columns for OQ
    rupture_df = rupture_df.rename(columns=OBS_DATA_OQ_COLS_MAPPING)
    rupture_df["vs30measured"] = False

    result_df = _compute_emp_gm_params(rupture_df, periods)
    return result_df


def _compute_emp_gm_params(rupture_df: pd.DataFrame, periods: Sequence[float]):
    """
    Compute empirical GM parameters for the given rupture data using OQ.

    Parameters
    ----------
    rupture_df : pd.DataFrame
        DataFrame containing rupture information with required columns.
        Columns z1pt0 and z2pt5 have to be in kilometres.
    periods : Sequence[float]
        List of periods for which pSA is to be computed.
    output_ffp : Path
        Output file path.
    """
    ### Constants
    GMM_MAPPING = {
        oqw.constants.TectType.ACTIVE_SHALLOW: oqw.constants.GMM.Br_13,
        oqw.constants.TectType.SUBDUCTION_SLAB: oqw.constants.GMM.K_20,
        oqw.constants.TectType.SUBDUCTION_INTERFACE: oqw.constants.GMM.K_20,
    }

    TECT_CLASS_MAPPING = {
        TectonicType.CRUSTAL: oqw.constants.TectType.ACTIVE_SHALLOW,
        TectonicType.SUBDUCTION_SLAB: oqw.constants.TectType.SUBDUCTION_SLAB,
        TectonicType.SUBDUCTION_INTERFACE: oqw.constants.TectType.SUBDUCTION_INTERFACE,
        TectonicType.UNKNOWN: oqw.constants.TectType.ACTIVE_SHALLOW,
        TectonicType.OUTER_RISE: oqw.constants.TectType.SUBDUCTION_SLAB,
    }

    ### GM prediction
    print("Running predictions")
    dfs = []
    sites = np.unique(rupture_df[ObservedData.SiteColEnums.SITE_ID])
    for cur_site in sites:
        cur_site_mask = rupture_df[ObservedData.SiteColEnums.SITE_ID].values == cur_site

        for cur_tect_class in np.unique(
            rupture_df.loc[cur_site_mask, ObservedData.EventColEnums.TECT_TYPE]
        ):
            cur_tect_mask = cur_site_mask & (
                rupture_df[ObservedData.EventColEnums.TECT_TYPE].values
                == cur_tect_class
            )

            if cur_tect_class not in TECT_CLASS_MAPPING:
                continue

            cur_tect_type = TECT_CLASS_MAPPING[cur_tect_class]
            pga_result = oqw.run_gmm(
                GMM_MAPPING[cur_tect_type],
                cur_tect_type,
                rupture_df.loc[cur_tect_mask, OQ_INPUT_COLUMNS],
                "PGA",
            )

            psa_result = oqw.run_gmm(
                GMM_MAPPING[cur_tect_type],
                cur_tect_type,
                rupture_df.loc[cur_tect_mask, OQ_INPUT_COLUMNS],
                "pSA",
                periods,
            )

            cur_df = pd.concat((pga_result, psa_result), axis=1)
            cur_df.index = rupture_df.loc[cur_tect_mask].index
            cur_df[
                [ObservedData.EventColEnums.EVENT_ID, ObservedData.SiteColEnums.SITE_ID]
            ] = rupture_df[
                [ObservedData.EventColEnums.EVENT_ID, ObservedData.SiteColEnums.SITE_ID]
            ]

            dfs.append(cur_df)

    result_df = pd.concat(dfs, axis=0)
    return result_df


def get_residuals(
    results: pd.DataFrame,
    ims: Sequence[str] = PSA_KEYS,
    pred_suffix: str = "pred",
    site_col: str = "site_int",
):
    """Computes the residual between the observed and predicted IMs for each scenario"""
    pred_im_keys = numpy_str_join("_", ims, pred_suffix)
    res_df = pd.DataFrame(
        data=results.loc[:, ims].values - results.loc[:, pred_im_keys].values,
        columns=ims,
    )

    res_df.index = results.index
    res_df["event_id"] = results["event_id"]
    res_df[site_col] = results[site_col]
    if "n_obs_sites" in results.columns:
        res_df["n_obs_sites"] = results["n_obs_sites"]
    return res_df


def get_bias_residual_fig(
    figsize: tuple[float, float] = (16, 6),
    fig_dpi: int = 300,
    left: float = 0.05,
    right: float = 0.98,
    top: float = 0.98,
    bottom: float = 0.1,
    main_wspace: float = 0.1,
    sub_wspace: float = 0.03,
    bias_y_axis_limits: tuple[float, float] = (-1.0, 1.0),
    std_y_axis_limits: tuple[float, float] = (0.0, 1.0),
):
    """
    Create a figure a bias and residual plots for
    pSA and non-pSA IMs

    Parameters
    ----------
    figsize : tuple of float, optional
        Size of the total figure.
    left : float, optional
        Left margin of the figure.
    right : float, optional
        Right margin of the figure.
    top : float, optional
        Top margin of the figure.
    bottom : float, optional
        Bottom margin of the figure.
    main_wspace : float, optional
        Space between the main plots.
        I.e. space between ax2 and ax3
    sub_wspace : float, optional
        Space between the subplots.
        I.e. space between ax1 and ax2
        and between ax3 and ax4

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax1 : matplotlib.axes.Axes
        Axis for the pSA bias plot.
    ax2 : matplotlib.axes.Axes
        Axis for the non-pSA bias plot.
    ax3 : matplotlib.axes.Axes
        Axis for the pSA residual standard deviation plot.
    ax4 : matplotlib.axes.Axes
        Axis for the non-pSA residual standard deviation plot.
    """
    fig = plt.figure(figsize=figsize, dpi=fig_dpi)

    main_grid = gridspec.GridSpec(1, 2, figure=fig, wspace=main_wspace)

    grid_bias = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=main_grid[0], wspace=sub_wspace, width_ratios=[5, 1]
    )

    ax1 = fig.add_subplot(grid_bias[0])
    ax1.set_xlabel("Vibration Period, T(s)")
    ax1.set_ylabel("Model bias")
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xscale("log")
    ax1.axhline(0, color="black", zorder=0)
    ax1.set_ylim(*bias_y_axis_limits)
    ax1.set_xlim(0.01, 10.0)

    ax2 = fig.add_subplot(grid_bias[1])
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_yticklabels([])
    ax2.set_ylim(*bias_y_axis_limits)
    ax2.axhline(0, color="black", zorder=0)
    ax2.tick_params(axis="y", which="both", length=0)

    grid_residual = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=main_grid[1], wspace=sub_wspace, width_ratios=[5, 1]
    )

    ax3 = fig.add_subplot(grid_residual[0])
    ax3.set_xlabel("Vibration Period, T(s)")
    ax3.set_ylabel("Residual standard deviation")
    ax3.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax3.set_xscale("log")
    ax3.set_ylim(*std_y_axis_limits)
    ax3.set_xlim(0.01, 10.0)

    ax4 = fig.add_subplot(grid_residual[1])
    ax4.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax4.set_yticklabels([])
    ax4.set_ylim(*std_y_axis_limits)
    ax4.tick_params(axis="y", which="both", length=0)

    # Remove general figure padding
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    return fig, ax1, ax2, ax3, ax4


def important_set_figures(
    quality_df: pd.DataFrame,
    flatfiles_dir: Path,
    important_set_names: list[str],
    import_set_ffps: list[str],
    label: str,
):
    """
    Creates figures comparing the number of station and event pairs
    and also the reasons for skipped records in the important sets from pie charts.

    Parameters
    ----------
    quality_df: pd.DataFrame
        DataFrame containing the quality information of the NZGMDB.
    flatfiles_dir: Path
        Directory containing the flat files for the skipped records.
    important_set_names: list[str]
        List of names for the important sets to be compared.
    import_set_ffps: list[str]
        List of file paths to the important sets to be compared.
    label: str
        Label for the figures, used in the title of the plots / legend.

    Returns
    -------
    bar_img_base64: str
        Base64 encoded string of the bar plot image.
    pie_imgs_base64: list[str]
        List of Base64 encoded strings of the pie chart images for each important set.
    """

    # Bar plot
    in_nzgmdb = []
    total_subset = []
    missing_event_stations = []
    for compare_set_ffp in import_set_ffps:
        compare_set = pd.read_csv(
            compare_set_ffp, dtype={"event_id": str, "stat_id": str}
        )
        quality_df["station_evid"] = quality_df["sta"] + "_" + quality_df["evid"]
        compare_set["station_evid"] = (
            compare_set["stat_id"] + "_" + compare_set["event_id"]
        )
        nzgmdb_set_station_evid = set(quality_df["station_evid"].unique())
        compare_set_station_evid = set(compare_set["station_evid"].unique())
        common_station_evid = list(
            nzgmdb_set_station_evid.intersection(compare_set_station_evid)
        )
        in_nzgmdb.append(len(common_station_evid))
        total_subset.append(len(compare_set_station_evid))
        missing_event_stations.append(
            compare_set_station_evid - nzgmdb_set_station_evid
        )

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    X_axis = np.arange(len(important_set_names))
    width = 0.75
    in_nzgmdb_arr = np.array(in_nzgmdb)
    missing_arr = np.array(total_subset) - in_nzgmdb_arr
    ax.bar(X_axis, in_nzgmdb_arr, width, label=f"In {label}", color="blue")
    ax.bar(
        X_axis, missing_arr, width, label="Missing", bottom=in_nzgmdb_arr, color="red"
    )
    ax.set_xticks(X_axis)
    ax.set_xticklabels(important_set_names)
    ax.set_xlabel("Datasets")
    ax.set_ylabel("Number of Station and Event Pairs")
    ax.set_title(f"{label} Number of Station and Event Pairs in Each Dataset")
    ax.legend()
    for i, (x, y, m) in enumerate(zip(X_axis, in_nzgmdb_arr, missing_arr)):
        ax.text(
            x,
            y + m + max(total_subset) * 0.01,
            f"{int(m)} Missing Records",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
            fontweight="bold",
        )
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    bar_img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Pie charts for skipped reasons
    skipped_files = [
        flatfiles_dir / "quality_skipped_records.csv",
        flatfiles_dir / "processing_skipped_records.csv",
        flatfiles_dir / "geonet_skipped_records.csv",
    ]
    rename_col_dict = {"mseed_file": "record_id", "skipped_records": "record_id"}
    pie_imgs_base64 = []
    for i, missing_event_stations_set in enumerate(missing_event_stations):
        cur_df = pd.DataFrame(columns=["station_evid", "reason"])
        for skipped_file in skipped_files:
            skipped_records = pd.read_csv(skipped_file)
            skipped_records = skipped_records.rename(columns=rename_col_dict)
            skipped_records["station_evid"] = skipped_records["record_id"].apply(
                lambda x: x.split("_")[1] + "_" + x.split("_")[0]
            )
            for event_station in missing_event_stations_set:
                sta_event_list = skipped_records["station_evid"].to_list()
                if event_station in sta_event_list:
                    cur_df = pd.concat(
                        [
                            cur_df,
                            pd.DataFrame(
                                {
                                    "station_evid": [event_station],
                                    "reason": [
                                        skipped_records[
                                            skipped_records["station_evid"]
                                            == event_station
                                        ]["reason"].values[0]
                                    ],
                                }
                            ),
                        ]
                    )
        errors = list(cur_df["reason"].unique())
        values = [len(cur_df[cur_df["reason"] == error]) for error in errors]
        total_errors = sum(values)
        unknown_errors = len(cur_df) - total_errors
        errors.append("missing")
        values.append(unknown_errors)
        fig = plot_pie_chart_1p(
            errors, values, f"{label} {important_set_names[i]} Skipped Reasons"
        )
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pie_imgs_base64.append(base64.b64encode(buf.read()).decode("utf-8"))

    return bar_img_base64, pie_imgs_base64


def compare_column_barplot(
    full_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    column: str,
    x_label: str,
    title: str,
    figsize=(16, 6),
    dpi=300,
    y_label="Number of Records",
):
    # Get all unique categories
    categories = sorted(
        set(full_df[column].unique()) | set(quality_df[column].unique())
    )
    x = np.arange(len(categories))
    width = 0.35

    # Count values for each category
    full_counts = full_df[column].value_counts().reindex(categories, fill_value=0)
    quality_counts = quality_df[column].value_counts().reindex(categories, fill_value=0)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bars1 = ax.bar(x - width / 2, full_counts.values, width, label="Full Database")
    bars2 = ax.bar(
        x + width / 2, quality_counts.values, width, label="Quality Database"
    )

    # Annotate bars with counts in bold
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(full_counts.max(), quality_counts.max()) * 0.01,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(full_counts.max(), quality_counts.max()) * 0.01,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64


def get_pSA_bias_residual_fig(
    figsize: tuple[float, float] = (16, 6),
    fig_dpi: int = 300,
    left: float = 0.05,
    right: float = 0.98,
    top: float = 0.98,
    bottom: float = 0.1,
    main_wspace: float = 0.1,
    bias_y_axis_limits: tuple[float, float] = (-1.0, 1.0),
    std_y_axis_limits: tuple[float, float] = (0.0, 1.0),
):
    """
    Create a figure for pSA bias and residual standard deviation plots.

    Parameters
    ----------
    figsize : tuple of float, optional
        Size of the figure.
    bias_y_axis_limits : tuple of float, optional
        Y-axis limits for the bias plot.
    std_y_axis_limits : tuple of float, optional
        Y-axis limits for the residual standard deviation plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax1 : matplotlib.axes.Axes
        Axis for the bias plot.
    ax2 : matplotlib.axes.Axes
        Axis for the residual standard deviation plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=fig_dpi)

    ax1.set_xlabel("Vibration Period, T(s)")
    ax1.set_ylabel("Model bias")
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xscale("log")
    ax1.axhline(0, color="black", zorder=0)
    ax1.set_ylim(*bias_y_axis_limits)
    ax1.set_xlim(0.01, 10.0)

    ax2.set_xlabel("Vibration Period, T(s)")
    ax2.set_ylabel("Residual standard deviation")
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_xscale("log")
    ax2.set_ylim(*std_y_axis_limits)
    ax2.set_xlim(0.01, 10.0)

    fig.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, wspace=main_wspace
    )
    return fig, ax1, ax2


def skipped_records_pie_chart(skipped_df: pd.DataFrame, title: str) -> str:
    """
    Generate a pie chart showing the distribution of skipped records by reason.

    Parameters
    ----------
    skipped_df : pd.DataFrame
        DataFrame containing skipped records with a 'reason' column.
    title : str
        Title for the pie chart.
    """
    errors = list(skipped_df["reason"].unique())
    values = [len(skipped_df[skipped_df["reason"] == error]) for error in errors]
    total_errors = sum(values)
    unknown_errors = len(skipped_df) - total_errors
    errors.append("missing")
    values.append(unknown_errors)
    fig = plot_pie_chart_1p(errors, values, title)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64


def generate_report(
    new_version_directory: Path,
    output_file: Path,
    compare_version_directory: Path | None = None,
):
    """
    Generate a HTML report comparing the new version of the database to a previous version.

    Parameters
    ----------
    new_version_directory : Path
        The directory containing the new version of the database. Top Level directory, must contain the 'flatfiles' directory as well as the 'quality_db' directory.
        Must have the skipped record files as well as the ground motion flat files.
    output_file : Path
        The file where the HTML report will be saved.
    compare_version_directory : Path | None
        The Top Level directory containing the previous version of the database to compare against.
        If None, a summary of the new version will be generated instead and comparison plots will not be generated.
    """
    html_parts = []
    html_parts.append(
        f"<html><head><title>NZGMDB Comparison Report ({new_version_directory.stem}){f' vs {compare_version_directory.stem}' if compare_version_directory else ''}</title>"
    )
    # Start of HTML
    html_parts.append(
        """
    <style>
        .fig-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .fig-grid img {
            width: 48%;
            height: auto;
            padding: 4px;
        }
        .fig-single {
            display: flex;
            justify-content: center;
            margin: 30px 0;
        }
        .fig-single img {
            width: 98%;
            height: auto;
            padding: 4px;
        }
    </style>
    </head><body>
    <h1>NZGMDB Comparison Report</h1>
    """
    )
    # Define Directories
    new_flatfiles_dir = file_structure.get_flatfile_dir(new_version_directory)
    new_quality_dir = file_structure.get_quality_db_dir(new_version_directory)
    old_flatifles_dir = (
        file_structure.get_flatfile_dir(compare_version_directory)
        if compare_version_directory
        else None
    )
    old_quality_dir = (
        file_structure.get_quality_db_dir(compare_version_directory)
        if compare_version_directory
        else None
    )

    # Load Ground Motion data
    nzgmdb_full_new_ffp = (
        new_flatfiles_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD50_FLAT
    )
    nzgmdb_quality_new_ffp = (
        new_quality_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD50_FLAT
    )
    nzgmdb_full_old_ffp = (
        (old_flatifles_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD50_FLAT)
        if compare_version_directory
        else None
    )
    nzgmdb_quality_old_ffp = (
        (old_quality_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD50_FLAT)
        if compare_version_directory
        else None
    )
    full_new = pd.read_csv(nzgmdb_full_new_ffp, dtype={"evid": str})
    quality_new = pd.read_csv(nzgmdb_quality_new_ffp, dtype={"evid": str})
    full_old = (
        pd.read_csv(nzgmdb_full_old_ffp, dtype={"evid": str})
        if nzgmdb_full_old_ffp
        else None
    )
    quality_old = (
        pd.read_csv(nzgmdb_quality_old_ffp, dtype={"evid": str})
        if nzgmdb_quality_old_ffp
        else None
    )

    # Load the Skipped Records
    new_quality_skipped = pd.read_csv(
        new_flatfiles_dir
        / file_structure.SkippedRecordFilenames.QUALITY_SKIPPED_RECORDS
    )
    old_quality_skipped = (
        pd.read_csv(
            old_flatifles_dir
            / file_structure.SkippedRecordFilenames.QUALITY_SKIPPED_RECORDS
        )
        if compare_version_directory
        else None
    )

    # Load new and compute empirical parameters
    # new_obs_data = load_obs_nzgmdb(nzgmdb_new_ffp, new_version)
    # new_emp_gm_params = compute_nzgmdb_emp_gm_params(new_obs_data)
    #
    # # Compute residuals
    # new_emp_gm_params[PSA_KEYS] = np.log(
    #     new_obs_data.record_df.loc[new_emp_gm_params.index, PSA_KEYS]
    # )
    # new_res = get_residuals(
    #     new_emp_gm_params,
    #     ims=PSA_KEYS,
    #     pred_suffix="mean",
    #     site_col="site_id",
    # )
    #
    # new_bias = new_res[PSA_KEYS].mean(axis=0)
    # new_std = new_res[PSA_KEYS].std(axis=0)
    #
    # # Load old and compute empirical parameters
    # old_obs_data = load_obs_nzgmdb(nzgmdb_old_ffp, old_version)
    # old_emp_gm_params = compute_nzgmdb_emp_gm_params(old_obs_data)
    #
    # # Compute residuals
    # old_emp_gm_params[PSA_KEYS] = np.log(
    #     old_obs_data.record_df.loc[old_emp_gm_params.index, PSA_KEYS]
    # )
    # old_res = get_residuals(
    #     old_emp_gm_params,
    #     ims=PSA_KEYS,
    #     pred_suffix="mean",
    #     site_col="site_id",
    # )
    #
    # old_bias = old_res[PSA_KEYS].mean(axis=0)
    # old_std = old_res[PSA_KEYS].std(axis=0)
    #
    # html_parts.append("<h2>New Version Summary</h2>")
    # html_parts.append("<div class='fig-single'>")
    #
    # # Generate psa bias and residual figure
    # fig, ax1, ax2 = get_pSA_bias_residual_fig(std_y_axis_limits=(0, 1.25))
    #
    # ax1.plot(
    #     PERIODS,
    #     old_bias,
    #     label="Old NZGMDB",
    # )
    # ax1.plot(
    #     PERIODS,
    #     new_bias,
    #     label="New NZGMDB",
    # )
    # ax1.legend()
    #
    # ax2.plot(
    #     PERIODS,
    #     old_std,
    #     label="Old NZGMDB",
    # )
    # ax2.plot(
    #     PERIODS,
    #     new_std,
    #     label="New NZGMDB",
    # )
    #
    # # Convert to base64
    # buf = BytesIO()
    # fig.savefig(buf, format="png", bbox_inches="tight")
    # plt.close(fig)
    # buf.seek(0)
    # img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    # # Embed in HTML
    # html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
    # html_parts.append("</div>")

    # Show Quality DB Skiped reasons and totals for records between full and quality
    html_parts.append("<h2>New NZGMDB Quality vs Full Statistics</h2>")

    total_records_f = len(full_new)
    unique_events_f = full_new["evid"].nunique()
    unique_sites_f = full_new["sta"].nunique()

    html_parts.append(f"<h2>Full Database</h2>")
    html_parts.append("<ul>")
    html_parts.append(f"<li>Total records: {total_records_f}</li>")
    html_parts.append(f"<li>Unique events: {unique_events_f}</li>")
    html_parts.append(f"<li>Unique sites: {unique_sites_f}</li>")
    html_parts.append("</ul>")

    total_records = len(quality_new)
    unique_events = quality_new["evid"].nunique()
    unique_sites = quality_new["sta"].nunique()

    html_parts.append(f"<h2>Quality Database</h2>")
    html_parts.append("<ul>")
    html_parts.append(f"<li>Total records: {total_records}</li>")
    html_parts.append(f"<li>Unique events: {unique_events}</li>")
    html_parts.append(f"<li>Unique sites: {unique_sites}</li>")
    html_parts.append("</ul>")

    if compare_version_directory:
        html_parts.append("<h2>Old NZGMDB Quality vs Full Statistics</h2>")

        total_records_f_old = len(full_old)
        unique_events_f_old = full_old["evid"].nunique()
        unique_sites_f_old = full_old["sta"].nunique()

        html_parts.append(f"<h2>Full Database</h2>")
        html_parts.append("<ul>")
        html_parts.append(f"<li>Total records: {total_records_f_old}</li>")
        html_parts.append(f"<li>Unique events: {unique_events_f_old}</li>")
        html_parts.append(f"<li>Unique sites: {unique_sites_f_old}</li>")
        html_parts.append("</ul>")

        total_records_old = len(quality_old)
        unique_events_old = quality_old["evid"].nunique()
        unique_sites_old = quality_old["sta"].nunique()
        html_parts.append(f"<h2>Quality Database</h2>")
        html_parts.append("<ul>")
        html_parts.append(f"<li>Total records: {total_records_old}</li>")
        html_parts.append(f"<li>Unique events: {unique_events_old}</li>")
        html_parts.append(f"<li>Unique sites: {unique_sites_old}</li>")
        html_parts.append("</ul>")

    # Compare the reasons why records were skipped into the quality database
    html_parts.append("<h2>New NZGMDB Quality DB Skipped Records</h2>")
    img_base64 = skipped_records_pie_chart(
        new_quality_skipped, "New NZGMDB Quality Skipped Reasons"
    )
    html_parts.append("<div class='fig-single'>")
    html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
    html_parts.append("</div>")

    if compare_version_directory:
        html_parts.append("<h2>Old NZGMDB Quality DB Skipped Records</h2>")
        img_base64 = skipped_records_pie_chart(
            old_quality_skipped, "Old NZGMDB Quality Skipped Reasons"
        )
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append("</div>")

    html_parts.append("<h2>Pipeline Skipped Records</h2>")
    # Prepare skipped files and accepted lengths for both new and old
    skipped_files_new = [
        new_flatfiles_dir
        / file_structure.SkippedRecordFilenames.GEONET_SKIPPED_RECORDS,
        new_flatfiles_dir
        / file_structure.SkippedRecordFilenames.PHASE_ARRIVAL_SKIPPED_RECORDS,
        new_flatfiles_dir / file_structure.SkippedRecordFilenames.SNR_SKIPPED_RECORDS,
        new_flatfiles_dir / file_structure.SkippedRecordFilenames.FMAX_SKIPPED_RECORDS,
        new_flatfiles_dir
        / file_structure.SkippedRecordFilenames.PROCESSING_SKIPPED_RECORDS,
    ]
    skipped_files_old = (
        [
            old_flatifles_dir
            / file_structure.SkippedRecordFilenames.GEONET_SKIPPED_RECORDS,
            old_flatifles_dir
            / file_structure.SkippedRecordFilenames.PHASE_ARRIVAL_SKIPPED_RECORDS,
            old_flatifles_dir
            / file_structure.SkippedRecordFilenames.SNR_SKIPPED_RECORDS,
            old_flatifles_dir
            / file_structure.SkippedRecordFilenames.FMAX_SKIPPED_RECORDS,
            old_flatifles_dir
            / file_structure.SkippedRecordFilenames.PROCESSING_SKIPPED_RECORDS,
        ]
        if compare_version_directory
        else [None] * 5
    )
    accepted_lengths_new = [
        len(
            pd.read_csv(
                new_flatfiles_dir
                / file_structure.PreFlatfileNames.STATION_MAGNITUDE_TABLE_GEONET
            )
        )
        / 3,
        len(
            pd.read_csv(
                new_flatfiles_dir / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE
            )
        ),
        len(pd.read_csv(new_flatfiles_dir / file_structure.FlatfileNames.SNR_METADATA)),
        len(pd.read_csv(new_flatfiles_dir / file_structure.FlatfileNames.FMAX)),
        len(full_new),
    ]
    accepted_lengths_old = (
        [
            len(
                pd.read_csv(
                    old_flatifles_dir
                    / file_structure.PreFlatfileNames.STATION_MAGNITUDE_TABLE_GEONET
                )
            )
            / 3,
            len(
                pd.read_csv(
                    old_flatifles_dir
                    / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE
                )
            ),
            len(
                pd.read_csv(
                    old_flatifles_dir / file_structure.FlatfileNames.SNR_METADATA
                )
            ),
            len(pd.read_csv(old_flatifles_dir / file_structure.FlatfileNames.FMAX)),
            len(full_old),
        ]
        if compare_version_directory
        else [None] * 5
    )
    titles = [
        "Geonet Skipped Records",
        "Phase Arrival Skipped Records",
        "SNR Skipped Records",
        "Fmax Skipped Records",
        "Processing Skipped Records",
    ]
    rename_col_dict = {
        "mseed_file": "record_id",
        "skipped_records": "record_id",
    }

    for i in range(len(skipped_files_new)):
        # New version
        skipped_file = skipped_files_new[i]
        skipped_records = pd.read_csv(skipped_file)
        skipped_records = skipped_records.rename(columns=rename_col_dict)
        errors = list(skipped_records["reason"].unique())
        values = [
            len(skipped_records[skipped_records["reason"] == error]) for error in errors
        ]
        total_errors = sum(values)
        unknown_errors = len(skipped_records) - total_errors
        values.append(accepted_lengths_new[i])
        errors.append("Accepted")
        errors.append("missing")
        values.append(unknown_errors)
        fig = plot_pie_chart_1p(errors, values, f"New NZGMDB {titles[i]}")
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append("</div>")

        # Old version (if available)
        if skipped_files_old[i] is not None:
            skipped_file = skipped_files_old[i]
            skipped_records = pd.read_csv(skipped_file)
            skipped_records = skipped_records.rename(columns=rename_col_dict)
            errors = list(skipped_records["reason"].unique())
            values = [
                len(skipped_records[skipped_records["reason"] == error])
                for error in errors
            ]
            total_errors = sum(values)
            unknown_errors = len(skipped_records) - total_errors
            values.append(accepted_lengths_old[i])
            errors.append("Accepted")
            errors.append("missing")
            values.append(unknown_errors)
            fig = plot_pie_chart_1p(errors, values, f"Old NZGMDB {titles[i]}")
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            html_parts.append("<div class='fig-single'>")
            html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
            html_parts.append("</div>")

    # Add Important Dataset Comparisons
    html_parts.append("<h2>Important Dataset Comparisons</h2>")

    important_set_names = ["lee_large", "lee_small", "brendon"]
    import_set_ffps = [
        data_registry.NZGMDB_DATA.fetch("lee_large.csv"),
        data_registry.NZGMDB_DATA.fetch("lee_small.csv"),
        data_registry.NZGMDB_DATA.fetch("brendon_set.csv"),
    ]

    # For new
    bar_img_base64_new, pie_imgs_base64_new = important_set_figures(
        quality_new,
        new_flatfiles_dir,
        important_set_names,
        import_set_ffps,
        "New NZGMDB",
    )

    # For old (if available)
    if compare_version_directory:
        bar_img_base64_old, pie_imgs_base64_old = important_set_figures(
            quality_old,
            old_flatifles_dir,
            important_set_names,
            import_set_ffps,
            "Old NZGMDB",
        )

    if compare_version_directory:
        html_parts.append("<div class='fig-grid'>")
        html_parts.append(f'<img src="data:image/png;base64,{bar_img_base64_new}">')
        html_parts.append(f'<img src="data:image/png;base64,{bar_img_base64_old}">')
        html_parts.append("</div>")
        # Add skipped reasons
        html_parts.append("<h2>Skipped Reasons</h2>")
        for i, img in enumerate(pie_imgs_base64_old):
            html_parts.append("<div class='fig-single'>")
            html_parts.append(
                f'<img src="data:image/png;base64,{pie_imgs_base64_new[i]}">'
            )
            html_parts.append("</div>")
            html_parts.append("<div class='fig-single'>")
            html_parts.append(f'<img src="data:image/png;base64,{img}">')
            html_parts.append("</div>")
    else:
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{bar_img_base64_new}">')
        html_parts.append("</div>")
        # Add skipped reasons
        html_parts.append("<h2>Skipped Reasons</h2>")
        for img in pie_imgs_base64_new:
            html_parts.append("<div class='fig-single'>")
            html_parts.append(f'<img src="data:image/png;base64,{img}">')
            html_parts.append("</div>")

    # Add Category Column Comparisons
    html_parts.append("<h2>Category Column Comparisons</h2>")

    # Make a subset unique to events
    full_new_events = full_new.drop_duplicates(subset=["evid"])
    quality_new_events = quality_new.drop_duplicates(subset=["evid"])
    if compare_version_directory:
        full_old_events = full_old.drop_duplicates(subset=["evid"])
        quality_old_events = quality_old.drop_duplicates(subset=["evid"])

    # Add Tectonic Type Comparison
    img_base64 = compare_column_barplot(
        full_new_events,
        quality_new_events,
        column="tect_class",
        x_label="Tectonic Type",
        title="New NZGMDB Tectonic Type Comparison",
    )
    if compare_version_directory:
        img_base64_old = compare_column_barplot(
            full_old_events,
            quality_old_events,
            column="tect_class",
            x_label="Tectonic Type",
            title="Old NZGMDB Tectonic Type Comparison",
        )
        html_parts.append("<div class='fig-grid'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append(f'<img src="data:image/png;base64,{img_base64_old}">')
        html_parts.append("</div>")
    else:
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append("</div>")

    # Add f_type Comparison
    img_base64 = compare_column_barplot(
        full_new_events,
        quality_new_events,
        column="f_type",
        x_label="Fault Type",
        title="New NZGMDB Fault Type Comparison",
    )
    if compare_version_directory:
        img_base64_old = compare_column_barplot(
            full_old_events,
            quality_old_events,
            column="f_type",
            x_label="Fault Type",
            title="Old NZGMDB Fault Type Comparison",
        )
        html_parts.append("<div class='fig-grid'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append(f'<img src="data:image/png;base64,{img_base64_old}">')
        html_parts.append("</div>")
    else:
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append("</div>")

    # Add reloc Comparison
    img_base64 = compare_column_barplot(
        full_new_events,
        quality_new_events,
        column="reloc",
        x_label="Relocation",
        title="New NZGMDB Relocation Comparison",
    )
    if compare_version_directory:
        img_base64_old = compare_column_barplot(
            full_old_events,
            quality_old_events,
            column="reloc",
            x_label="Relocation",
            title="Old NZGMDB Relocation Comparison",
        )
        html_parts.append("<div class='fig-grid'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append(f'<img src="data:image/png;base64,{img_base64_old}">')
        html_parts.append("</div>")
    else:
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append("</div>")
    #
    # # Add psa count Comparison
    # html_parts.append("<h2>pSA Record Count Comparison</h2>")
    # html_parts.append('<div class="fig-single">')
    #
    # # Generate fmin plots
    # old_record_count = (~old_obs_data.record_df[PSA_KEYS].isna()).sum(axis=0)
    # new_record_count = (~new_obs_data.record_df[PSA_KEYS].isna()).sum(axis=0)
    #
    # fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
    #
    # ax.plot(
    #     PERIODS,
    #     old_record_count.loc[PSA_KEYS],
    #     label="Old NZGMDB",
    # )
    # ax.plot(
    #     PERIODS,
    #     new_record_count.loc[PSA_KEYS],
    #     label="New NZGMDB",
    # )
    #
    # ax.set_xlabel("Period (s)")
    # ax.set_xscale("log")
    # ax.set_ylabel("Count")
    # ax.set_xlim(0.01, 10.0)
    # ax.legend()
    # ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    #
    # fig.tight_layout()
    #
    # # Convert to base64
    # buf = BytesIO()
    # fig.savefig(buf, format="png", bbox_inches="tight")
    # plt.close(fig)
    # buf.seek(0)
    # img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    # # Embed in HTML
    # html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
    # html_parts.append("</div>")
    #
    # # Add IM Compare
    # html_parts.append("<h2>IM Comparison</h2>")
    # html_parts.append('<div class="fig-single">')
    #
    # # IM Compare
    # shared_record_ids = np.intersect1d(
    #     old_obs_data.record_df.index.values.astype(str),
    #     new_obs_data.record_df.index.values.astype(str),
    # )
    #
    # plot_ims = ["pSA_0.01", "pSA_0.1", "pSA_0.5", "pSA_1.0", "pSA_3.0", "pSA_10.0"]
    #
    # fig, axs = get_fig_axes(len(plot_ims), 2, -1, ind_figsize=(8, 6), dpi=300)
    #
    # for i, (cur_im, cur_ax) in enumerate(zip(plot_ims, axs)):
    #     cur_old = old_obs_data.record_df.loc[shared_record_ids, cur_im]
    #     cur_new = new_obs_data.record_df.loc[shared_record_ids, cur_im]
    #
    #     cur_max = max(cur_old.max(), cur_new.max())
    #
    #     cur_ax.scatter(cur_old, cur_new, s=1)
    #     cur_ax.set_xlabel("Old NZGMDB")
    #     cur_ax.set_ylabel("New NZGMDB")
    #     cur_ax.set_title(cur_im)
    #     cur_ax.plot(
    #         [0, cur_max], [0, cur_max], color="black", linestyle="--", linewidth=0.5
    #     )
    #     cur_ax.set_xlim(0, cur_max)
    #     cur_ax.set_ylim(0, cur_max)
    #     cur_ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    #
    # # Convert to base64
    # buf = BytesIO()
    # fig.savefig(buf, format="png", bbox_inches="tight")
    # plt.close(fig)
    # buf.seek(0)
    # img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    # # Embed in HTML
    # html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
    # html_parts.append("</div>")
    #
    # html_parts.append(
    #     """
    # <h2>GMM Input Comparison</h2>
    # <div class="fig-grid">
    # """
    # )
    # # GMM Input comparison
    # input_cols = [
    #     "event_lat",
    #     "event_lon",
    #     "depth",
    #     "mag",
    #     "dip",
    #     "rake",
    #     "ztor",
    #     "rjb",
    #     "rrup",
    #     "rx",
    #     "vs30",
    #     "z1p0",
    #     "z2p5",
    # ]
    # # Generate figures and embed as base64
    # for i, col in enumerate(input_cols):
    #     cur_x_data = old_obs_data.record_df.loc[shared_record_ids, col].values
    #     cur_y_data = new_obs_data.record_df.loc[shared_record_ids, col].values
    #     nan_mask = np.isnan(cur_x_data) | np.isnan(cur_y_data)
    #     cur_x_data = cur_x_data[~nan_mask]
    #     cur_y_data = cur_y_data[~nan_mask]
    #     lims = (
    #         np.quantile(cur_x_data, 0.01),
    #         np.quantile(cur_x_data, 0.99),
    #     )
    #     fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    #     ax.scatter(cur_x_data, cur_y_data, alpha=0.5, s=1)
    #     ax.plot(lims, lims, color="k")
    #     ax.set_xlabel("Old NZGMDB")
    #     ax.set_ylabel("New NZGMDB")
    #     ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    #     ax.set_xlim(lims)
    #     ax.set_ylim(lims)
    #     ax.set_aspect("equal")
    #     ax.set_title(f"{col} - N: {len(cur_x_data)}")
    #     fig.tight_layout()
    #
    #     # Convert to base64
    #     buf = BytesIO()
    #     fig.savefig(buf, format="png", bbox_inches="tight")
    #     plt.close(fig)
    #     buf.seek(0)
    #     img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    #     # Embed in HTML
    #     html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
    # # End of Section
    # html_parts.append("</div>")

    html_parts.append("</body></html>")
    # Save report
    with open(output_file, "w") as f:
        f.write("".join(html_parts))


generate_report(
    Path("/home/joel/local/gmdb/4p3_mantle"),
    Path("/home/joel/local/gmdb/4p3/report.html"),
    Path("/home/joel/local/gmdb/4p3_backup/4p3_backup"),
)
