"""
Generate a HTML report for the NZGMDB database comparing to a previous version / giving a summary of the current state.
"""

import base64
from collections.abc import Sequence
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.patches import Patch

import oq_wrapper as oqw
from nzgmdb.management import data_registry, file_structure
from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


class TectonicType(StrEnum):
    """Enum for tectonic classification."""

    CRUSTAL = "Crustal"
    SUBDUCTION_INTERFACE = "Interface"
    SUBDUCTION_SLAB = "Slab"
    OUTER_RISE = "Outer-rise"
    UNKNOWN = "Undetermined"


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


def apply_fmin_filter_df(df: pd.DataFrame, pre_4p3: bool = False) -> pd.DataFrame:
    """
    Applies fmin filtering to pSA columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing pSA columns and HPF/HPF_h columns.
    pre_4p3 : bool, optional
        If True, uses the HPF column for filtering (for versions before 4.3).

    Returns
    -------
    pd.DataFrame
        DataFrame with pSA values set to NaN where the period exceeds the max usable period.
    """
    if pre_4p3:
        # Use just the HPF
        max_usable_period = 1 / df["HPF"] * 1.25
    else:
        max_usable_period = 1 / df["HPF_h"] * 1.25
    pSA_cols = [col for col in df.columns if col.startswith("pSA")]
    for col in pSA_cols:
        period = float(col.split("_")[1])
        df[col] = np.where(period > max_usable_period, np.nan, df[col])
    return df


def format_percentage(pct: float, allvals: list[float]):
    """
    Function to format pie chart labels with both percentage and absolute values.

    Parameters
    ----------
    pct : float
        The percentage value.
    allvals : list[float]
        The list of all values to compute the absolute value.

    Returns
    -------
    str
        The formatted string for the pie chart label. For example,
        ``"12.5%\\n(25)"``. Returns an empty string if the percentage
        is less than 5%.
    """
    absolute = round(pct / 100.0 * np.sum(allvals))
    normal = f"{pct:.1f}%\n({absolute:d})"
    return normal if pct > 5 else ""


def plot_pie_chart(full_labels: list[str], full_sizes: list[int], title: str):
    """
    Plots a pie chart with the given labels and sizes.

    Parameters
    ----------
    full_labels : list[str]
        The labels for each category.
    full_sizes : list[int]
        The sizes for each category.
    title : str
        The title for the pie chart.

    Returns
    -------
    plt.Figure
        The matplotlib figure object containing the pie chart.
    """
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
        autopct=lambda pct: format_percentage(pct, sizes),
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


def numpy_str_join(sep: str, *arrays: str | Sequence[str]) -> np.ndarray:
    """
    Join multiple string arrays together using the specified separator.

    Supports joining string scalars, string arrays, or a combination of both.

    Parameters
    ----------
    sep : str
        The separator to use.
    *arrays : str or sequence of str
        String scalars or string arrays to join together.

    Returns
    -------
    numpy.ndarray
        The joined array.
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
    n_subplots : int
        The number of subplots.
    n_cols : int
        The number of columns.
        Set to -1 if n_rows is to be computed.
    n_rows : int
        The number of rows.
        Set to -1 if n_cols is to be computed.
    ind_figsize : tuple[int, int]
        The individual figure size for each subplot

    Returns
    -------
    fig : plt.Figure
        The figure object
    axs : List[plt.Axes]
        The axes objects

    Raises
    ------
    ValueError
        If both n_cols and n_rows are specified
    """

    if n_cols == -1 or n_rows == -1:
        if n_cols > 0:
            n_rows = int(np.ceil(n_subplots / n_cols))
        elif n_rows > 0:
            n_cols = int(np.ceil(n_subplots / n_rows))
        else:
            raise ValueError("One of n_cols/n_rows must be specified")

    figsize = (n_cols * ind_figsize[0], n_rows * ind_figsize[1])
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=300)

    if n_subplots == 1:
        axs = (axs,)
    else:
        axs = list(axs.flatten())

    return fig, axs


OBS_DATA_OQ_COLS_MAPPING = {
    "Vs30": "vs30",
    "r_rup": "rrup",
    "r_jb": "rjb",
    "Z1.0": "z1pt0",
    "mag": "mag",
    "rake": "rake",
    "dip": "dip",
    "z_tor": "ztor",
    "r_x": "rx",
    "ev_depth": "hypo_depth",
}


def compute_nzgmdb_emp_gm_params(obs_data: pd.DataFrame):
    """
    Computes the empirical GMM parameters for all
    specified sites and sources, based on inputs
    from NZGMDB

    Parameters
    ----------
    obs_data : pd.DataFrame
        DataFrame containing the observed data, event and site information.

    Returns
    -------
    pd.DataFrame
        The empirical GMM parameters for PGA
        and the default set of pSA periods
    """
    # Create rupture dataframe
    columns = [
        "evid",
        "sta",
        "sta_lon",
        "sta_lat",
        "tect_class",
    ] + list(OBS_DATA_OQ_COLS_MAPPING.keys())
    # rupture_df = pd.read_csv(nzgmdb_flat_ffp, index_col=0)[columns]
    rupture_df = obs_data[columns].copy(True)

    # Convert Z1.0 to kilometres
    rupture_df["Z1.0"] /= 1000

    # Rename columns for OQ
    rupture_df = rupture_df.rename(columns=OBS_DATA_OQ_COLS_MAPPING)
    rupture_df["vs30measured"] = False

    result_df = _compute_emp_gm_params(rupture_df, PERIODS)
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

    Returns
    -------
    pd.DataFrame
        DataFrame containing the computed empirical GM parameters.
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
    dfs = []
    sites = np.unique(rupture_df["sta"])
    for cur_site in sites:
        cur_site_mask = rupture_df["sta"].values == cur_site

        for cur_tect_class in np.unique(rupture_df.loc[cur_site_mask, "tect_class"]):
            cur_tect_mask = cur_site_mask & (
                rupture_df["tect_class"].values == cur_tect_class
            )

            if cur_tect_class not in TECT_CLASS_MAPPING:
                continue

            cur_tect_type = TECT_CLASS_MAPPING[cur_tect_class]

            # Filter all rrup below 500
            cur_tect_mask = cur_tect_mask & (rupture_df["rrup"].values <= 500)

            # Apply mag/rrup filters for each tectonic type
            if cur_tect_class == TectonicType.CRUSTAL:
                cur_tect_mask = (
                    cur_tect_mask
                    & (rupture_df["mag"].values >= 3.5)
                    & (rupture_df["rrup"].values <= 300)
                )
            elif cur_tect_class == TectonicType.SUBDUCTION_INTERFACE:
                cur_tect_mask = cur_tect_mask & (rupture_df["mag"].values >= 4.5)
            elif cur_tect_class == TectonicType.SUBDUCTION_SLAB:
                cur_tect_mask = cur_tect_mask & (rupture_df["mag"].values >= 4.5)

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
            cur_df[["evid", "sta"]] = rupture_df[["evid", "sta"]]

            dfs.append(cur_df)

    result_df = pd.concat(dfs, axis=0)
    return result_df


def get_residuals(
    results: pd.DataFrame,
    ims: Sequence[str] = PSA_KEYS,
    pred_suffix: str = "pred",
):
    """
    Computes the residual between the observed and predicted IMs for each scenario.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing observed and predicted IMs.
    ims : Sequence[str], optional
        List of IMs to compute residuals for, by default PSA_KEYS.
    pred_suffix : str, optional
        Suffix for the predicted IM columns, by default "pred".

    Returns
    -------
    pd.DataFrame
        DataFrame containing the residuals for each IM, along with evid and sta columns.
    """
    pred_im_keys = numpy_str_join("_", ims, pred_suffix)
    res_df = pd.DataFrame(
        data=results.loc[:, ims].values - results.loc[:, pred_im_keys].values,
        columns=ims,
    )

    res_df.index = results.index
    res_df["evid"] = results["evid"]
    res_df["sta"] = results["sta"]
    if "n_obs_sites" in results.columns:
        res_df["n_obs_sites"] = results["n_obs_sites"]
    return res_df


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
    quality_df : pd.DataFrame
        DataFrame containing the quality information of the NZGMDB.
    flatfiles_dir : Path
        Directory containing the flat files for the skipped records.
    important_set_names : list[str]
        List of names for the important sets to be compared.
    import_set_ffps : list[str]
        List of file paths to the important sets to be compared.
    label : str
        Label for the figures, used in the title of the plots / legend.

    Returns
    -------
    bar_img_base64 : str
        Base64 encoded string of the bar plot image.
    pie_imgs_base64 : list[str]
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
        fig = plot_pie_chart(
            errors, values, f"{label} {important_set_names[i]} Skipped Reasons"
        )
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        pie_imgs_base64.append(base64.b64encode(buf.read()).decode("utf-8"))

    return bar_img_base64, pie_imgs_base64


def skipped_reason_overlap_barplot(skipped_df: pd.DataFrame, title: str):
    """
    Generate a bar plot showing the overlap of skipped reasons.

    Parameters
    ----------
    skipped_df : pd.DataFrame
        DataFrame containing skipped records with 'record_id' and 'reason' columns.
    title : str
        Title for the bar plot.

    Returns
    -------
    str
        Base64 encoded string of the bar plot image.
    """
    # Get all unique reasons
    reasons = sorted(skipped_df["reason"].unique())
    total_counts = skipped_df["reason"].value_counts().reindex(reasons, fill_value=0)

    # For each reason, count how many record_ids also appear in another reason
    overlap_counts = []
    for reason in reasons:
        ids_in_bin = set(skipped_df.loc[skipped_df["reason"] == reason, "record_id"])
        ids_in_other_bins = set(
            skipped_df.loc[skipped_df["reason"] != reason, "record_id"]
        )
        overlap = ids_in_bin & ids_in_other_bins
        overlap_counts.append(len(overlap))
    overlap_counts = np.array(overlap_counts)
    non_overlap_counts = total_counts.values - overlap_counts

    x = np.arange(len(reasons))
    width = 0.75

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.bar(
        x,
        overlap_counts,
        width,
        label="Other Reason",
        color="blue",
    )
    ax.bar(
        x,
        non_overlap_counts,
        width,
        label="Unique to Reason",
        color="red",
        bottom=overlap_counts,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(reasons, rotation=45, ha="right")
    ax.set_xlabel("Skipped Reason")
    ax.set_ylabel("Number of Records")
    ax.set_title(title)
    ax.legend()
    for i, (xv, y1, y2) in enumerate(zip(x, non_overlap_counts, overlap_counts)):
        ax.text(
            xv,
            y1 + y2 + max(total_counts) * 0.01,
            f"{int(y1 + y2)}",
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
    return base64.b64encode(buf.read()).decode("utf-8")


def single_column_barplot(
    df: pd.DataFrame,
    column: str,
    x_label: str,
    title: str,
    y_label: str = "Number of Records",
    bar_label: str = "Records",
):
    """
    Generate a single column bar plot for the specified column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    column : str
        The column to plot.
    x_label : str
        Label for the x-axis.
    title : str
        Title for the bar plot.
    y_label : str, optional
        Label for the y-axis, by default "Number of Records".
    bar_label : str, optional
        Label for the bars in the legend, by default "Records".

    Returns
    -------
    str
        Base64 encoded string of the bar plot image.
    """
    categories = sorted(df[column].unique())
    counts = df[column].value_counts().reindex(categories, fill_value=0)
    x = np.arange(len(categories))
    width = 0.6

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    bars = ax.bar(x, counts.values, width, label=bar_label)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + counts.max() * 0.01,
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


def compare_column_barplot(
    full_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    column: str,
    x_label: str,
    title: str,
    y_label: str = "Number of Records",
    bar_1_label: str = "Full Database",
    bar_2_label: str = "Quality Database",
):
    """
    Generate a comparative bar plot for the specified column in two DataFrames.

    Parameters
    ----------
    full_df : pd.DataFrame
        DataFrame containing the full dataset.
    quality_df : pd.DataFrame
        DataFrame containing the quality dataset.
    column : str
        The column to plot.
    x_label : str
        Label for the x-axis.
    title : str
        Title for the bar plot.
    y_label : str, optional
        Label for the y-axis, by default "Number of Records".
    bar_1_label : str, optional
        Label for the first bar in the legend, by default "Full Database".
    bar_2_label : str, optional
        Label for the second bar in the legend, by default "Quality Database".

    Returns
    -------
    str
        Base64 encoded string of the bar plot image.
    """
    # Get all unique categories
    categories = sorted(
        set(full_df[column].unique()) | set(quality_df[column].unique())
    )
    x = np.arange(len(categories))
    width = 0.35

    # Count values for each category
    full_counts = full_df[column].value_counts().reindex(categories, fill_value=0)
    quality_counts = quality_df[column].value_counts().reindex(categories, fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    bars1 = ax.bar(x - width / 2, full_counts.values, width, label=bar_1_label)
    bars2 = ax.bar(x + width / 2, quality_counts.values, width, label=bar_2_label)

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


def get_pSA_bias_residual_fig():
    """
    Create a figure for pSA bias and residual standard deviation plots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax1 : matplotlib.axes.Axes
        Axis for the bias plot.
    ax2 : matplotlib.axes.Axes
        Axis for the residual standard deviation plot.
    """
    std_y_axis_limits = (0, 1.25)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    ax1.set_xlabel("Vibration Period, T(s)")
    ax1.set_ylabel("Model bias")
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xscale("log")
    ax1.axhline(0, color="black", zorder=0)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_xlim(0.01, 10.0)

    ax2.set_xlabel("Vibration Period, T(s)")
    ax2.set_ylabel("Residual standard deviation")
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_xscale("log")
    ax2.set_ylim(*std_y_axis_limits)
    ax2.set_xlim(0.01, 10.0)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.1, wspace=0.1)
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

    Returns
    -------
    str
        Base64 encoded string of the pie chart image.
    """
    errors = list(skipped_df["reason"].unique())
    values = [len(skipped_df[skipped_df["reason"] == error]) for error in errors]
    total_errors = sum(values)
    unknown_errors = len(skipped_df) - total_errors
    errors.append("missing")
    values.append(unknown_errors)
    fig = plot_pie_chart(errors, values, title)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64


def mag_rrup_scatter(
    df1: pd.DataFrame, df2: pd.DataFrame, title1: str, title2: str
) -> str:
    """
    Generate a side-by-side scatter plot comparing magnitude vs. Rrup for two datasets.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataset containing 'mag' and 'r_rup' columns.
    df2 : pd.DataFrame
        Second dataset containing 'mag' and 'r_rup' columns.
    title1 : str
        Title for the first subplot.
    title2 : str
        Title for the second subplot.

    Returns
    -------
    str
        Base64 encoded string of the scatter plot image.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=300, sharex=True)

    # Compute shared y-axis limits
    mag_min = min(df1["mag"].min() - 0.2, df2["mag"].min() - 0.2)
    mag_max = max(df1["mag"].max() + 0.2, df2["mag"].max() + 0.2)

    axs[0].scatter(df1["r_rup"], df1["mag"], alpha=0.5, s=10)
    axs[0].set_xscale("log")
    axs[0].set_ylabel("Magnitude")
    axs[0].set_xlabel("Rrup (km)")
    axs[0].set_title(title1)
    axs[0].set_ylim(mag_min, mag_max)
    axs[0].grid(True, which="both", ls="--", alpha=0.5)

    axs[1].scatter(df2["r_rup"], df2["mag"], alpha=0.5, s=10)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Rrup (km)")
    axs[1].set_ylabel("Magnitude")
    axs[1].set_title(title2)
    axs[1].set_ylim(mag_min, mag_max)
    axs[1].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@cli.from_docstring(app)
def generate_report(
    new_version_directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Argument(),
    ],
    compare_version_directory: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
        ),
    ] = None,
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
    """
    )
    html_parts.append(
        f"""
    <h1>NZGMDB Comparison Report {new_version_directory.stem}{f' vs {compare_version_directory.stem}' if compare_version_directory else ''}</h1>
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

    # Compute empirical parameters
    new_emp_gm_params = compute_nzgmdb_emp_gm_params(quality_new)

    # Compute residuals
    new_emp_gm_params[PSA_KEYS] = np.log(
        quality_new.loc[new_emp_gm_params.index, PSA_KEYS]
    )
    new_res = get_residuals(
        new_emp_gm_params,
        ims=PSA_KEYS,
        pred_suffix="mean",
    )

    new_bias = new_res[PSA_KEYS].mean(axis=0)
    new_std = new_res[PSA_KEYS].std(axis=0)

    if compare_version_directory:
        # Compute empirical parameters
        old_emp_gm_params = compute_nzgmdb_emp_gm_params(quality_old)

        # Compute residuals
        old_emp_gm_params[PSA_KEYS] = np.log(
            quality_old.loc[old_emp_gm_params.index, PSA_KEYS]
        )
        old_res = get_residuals(
            old_emp_gm_params,
            ims=PSA_KEYS,
            pred_suffix="mean",
        )

        old_bias = old_res[PSA_KEYS].mean(axis=0)
        old_std = old_res[PSA_KEYS].std(axis=0)

    html_parts.append("<h2>New Version Summary</h2>")
    html_parts.append("<div class='fig-single'>")

    # Generate psa bias and residual figure
    fig, ax1, ax2 = get_pSA_bias_residual_fig()

    if compare_version_directory:
        ax1.plot(
            PERIODS,
            old_bias,
            label="Old NZGMDB",
        )
        ax2.plot(
            PERIODS,
            old_std,
            label="Old NZGMDB",
        )

    ax1.plot(
        PERIODS,
        new_bias,
        label="New NZGMDB",
    )
    ax1.legend()

    ax2.plot(
        PERIODS,
        new_std,
        label="New NZGMDB",
    )

    # Convert to base64
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    # Embed in HTML
    html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
    html_parts.append("</div>")

    # Show Quality DB Skiped reasons and totals for records between full and quality
    html_parts.append("<h2>New NZGMDB Quality vs Full Statistics</h2>")

    total_records_f = len(full_new)
    unique_events_f = full_new["evid"].nunique()
    unique_sites_f = full_new["sta"].nunique()

    html_parts.append("<h2>Full Database</h2>")
    html_parts.append("<ul>")
    html_parts.append(f"<li>Total records: {total_records_f}</li>")
    html_parts.append(f"<li>Unique events: {unique_events_f}</li>")
    html_parts.append(f"<li>Unique sites: {unique_sites_f}</li>")
    html_parts.append("</ul>")

    total_records = len(quality_new)
    unique_events = quality_new["evid"].nunique()
    unique_sites = quality_new["sta"].nunique()

    html_parts.append("<h2>Quality Database</h2>")
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

        html_parts.append("<h2>Full Database</h2>")
        html_parts.append("<ul>")
        html_parts.append(f"<li>Total records: {total_records_f_old}</li>")
        html_parts.append(f"<li>Unique events: {unique_events_f_old}</li>")
        html_parts.append(f"<li>Unique sites: {unique_sites_f_old}</li>")
        html_parts.append("</ul>")

        total_records_old = len(quality_old)
        unique_events_old = quality_old["evid"].nunique()
        unique_sites_old = quality_old["sta"].nunique()
        html_parts.append("<h2>Quality Database</h2>")
        html_parts.append("<ul>")
        html_parts.append(f"<li>Total records: {total_records_old}</li>")
        html_parts.append(f"<li>Unique events: {unique_events_old}</li>")
        html_parts.append(f"<li>Unique sites: {unique_sites_old}</li>")
        html_parts.append("</ul>")

    # Add in magnitude and distance scatter plots for new vs old with full vs quality
    html_parts.append("<h2>Magnitude vs Distance Plots</h2>")

    img_base64 = mag_rrup_scatter(
        full_new, quality_new, "Mag vs Rrup: Full (New)", "Mag vs Rrup: Quality (New)"
    )
    html_parts.append("<div class='fig-single'>")
    html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
    html_parts.append("</div>")

    # Old version: Full vs Quality (if available)
    if compare_version_directory:
        img_base64 = mag_rrup_scatter(
            full_old,
            quality_old,
            "Mag vs Rrup: Full (Old)",
            "Mag vs Rrup: Quality (Old)",
        )
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append("</div>")

    html_parts.append("<h2>NZGMDB Channel Comparison</h2>")
    # chan (Channel) Comparison
    if compare_version_directory:
        img_base64_full_chan = compare_column_barplot(
            full_old,
            full_new,
            column="chan",
            x_label="Channel",
            title="Full NZGMDB Channel Comparison",
            bar_1_label="Old NZGMDB",
            bar_2_label="New NZGMDB",
        )
        img_base64_quality_chan = compare_column_barplot(
            quality_old,
            quality_new,
            column="chan",
            x_label="Channel",
            title="Quality NZGMDB Channel Comparison",
            bar_1_label="Old NZGMDB",
            bar_2_label="New NZGMDB",
        )
        html_parts.append("<div class='fig-grid'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64_full_chan}">')
        html_parts.append(
            f'<img src="data:image/png;base64,{img_base64_quality_chan}">'
        )
        html_parts.append("</div>")
    else:
        img_base64_chan = compare_column_barplot(
            full_new,
            quality_new,
            column="chan",
            x_label="Channel",
            title="New NZGMDB Channel Comparison",
        )
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64_chan}">')
        html_parts.append("</div>")

    # Compare the reasons why records were skipped into the quality database
    html_parts.append("<h2>New NZGMDB Quality DB Skipped Records</h2>")
    img_base64 = skipped_records_pie_chart(
        new_quality_skipped.drop_duplicates(subset="record_id"),
        "New NZGMDB Quality Skipped Reasons",
    )
    html_parts.append("<div class='fig-single'>")
    html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
    html_parts.append("</div>")

    if compare_version_directory:
        html_parts.append("<h2>Old NZGMDB Quality DB Skipped Records</h2>")
        img_base64 = skipped_records_pie_chart(
            old_quality_skipped.drop_duplicates(subset="record_id"),
            "Old NZGMDB Quality Skipped Reasons",
        )
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append("</div>")

    html_parts.append("<h2>All Quality metrics compared to Full Database</h2>")
    if compare_version_directory:
        img_base64_quality = compare_column_barplot(
            old_quality_skipped,
            new_quality_skipped,
            column="reason",
            x_label="Skipped Reason",
            title="Quality NZGMDB Skipped Reason Comparison",
            bar_1_label="Old NZGMDB",
            bar_2_label="New NZGMDB",
        )
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64_quality}">')
        html_parts.append("</div>")
    else:
        img_base64 = single_column_barplot(
            new_quality_skipped,
            column="reason",
            x_label="Skipped Reason",
            title="New NZGMDB Skipped Reason Comparison",
        )
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append("</div>")
    # Remove any reason that is "Duplicate channels"
    new_quality_skipped_adjusted = new_quality_skipped[
        new_quality_skipped["reason"] != "Duplicate channels"
    ]
    img_base64 = skipped_reason_overlap_barplot(
        new_quality_skipped_adjusted,
        title="New NZGMDB Skipped Reason Comparison",
    )
    html_parts.append("<div class='fig-single'>")
    html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
    html_parts.append("</div>")

    # Remove any reason that is "Duplicate channels" for old quality skipped
    if compare_version_directory:
        old_quality_skipped_adjusted = old_quality_skipped[
            old_quality_skipped["reason"] != "Duplicate channels"
        ]
        img_base64 = skipped_reason_overlap_barplot(
            old_quality_skipped_adjusted,
            title="Old NZGMDB Skipped Reason Comparison",
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
        fig = plot_pie_chart(errors, values, f"New NZGMDB {titles[i]}")
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
            fig = plot_pie_chart(errors, values, f"Old NZGMDB {titles[i]}")
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
    html_parts.append("<h2>Event Column Comparisons</h2>")

    # Make a subset unique to events
    full_new_events = full_new.drop_duplicates(subset=["evid"])
    quality_new_events = quality_new.drop_duplicates(subset=["evid"])
    if compare_version_directory:
        full_old_events = full_old.drop_duplicates(subset=["evid"])
        quality_old_events = quality_old.drop_duplicates(subset=["evid"])

    if compare_version_directory:
        img_base64_full = compare_column_barplot(
            full_old_events,
            full_new_events,
            column="tect_class",
            x_label="Tectonic Type",
            title="Full NZGMDB Tectonic Type Comparison",
            bar_1_label="Old NZGMDB",
            bar_2_label="New NZGMDB",
        )
        img_base64_quality = compare_column_barplot(
            quality_old_events,
            quality_new_events,
            column="tect_class",
            x_label="Tectonic Type",
            title="Quality NZGMDB Tectonic Type Comparison",
            bar_1_label="Old NZGMDB",
            bar_2_label="New NZGMDB",
        )
        html_parts.append("<div class='fig-grid'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64_full}">')
        html_parts.append(f'<img src="data:image/png;base64,{img_base64_quality}">')
        html_parts.append("</div>")
    else:
        img_base64 = compare_column_barplot(
            full_new_events,
            quality_new_events,
            column="tect_class",
            x_label="Tectonic Type",
            title="New NZGMDB Tectonic Type Comparison",
        )
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append("</div>")
    # f_type Comparison
    if compare_version_directory:
        img_base64_full = compare_column_barplot(
            full_old_events,
            full_new_events,
            column="f_type",
            x_label="Fault Type",
            title="Full NZGMDB Fault Type Comparison",
            bar_1_label="Old NZGMDB",
            bar_2_label="New NZGMDB",
        )
        img_base64_quality = compare_column_barplot(
            quality_old_events,
            quality_new_events,
            column="f_type",
            x_label="Fault Type",
            title="Quality NZGMDB Fault Type Comparison",
            bar_1_label="Old NZGMDB",
            bar_2_label="New NZGMDB",
        )
        html_parts.append("<div class='fig-grid'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64_full}">')
        html_parts.append(f'<img src="data:image/png;base64,{img_base64_quality}">')
        html_parts.append("</div>")
    else:
        img_base64 = compare_column_barplot(
            full_new_events,
            quality_new_events,
            column="f_type",
            x_label="Fault Type",
            title="New NZGMDB Fault Type Comparison",
        )
        html_parts.append("<div class='fig-single'>")
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
        html_parts.append("</div>")
    # Reloc Comparison by Fault Type
    if compare_version_directory:
        all_f_types = sorted(
            set(full_old_events["f_type"].unique())
            | set(full_new_events["f_type"].unique())
            | set(quality_old_events["f_type"].unique())
            | set(quality_new_events["f_type"].unique())
        )
        for f_type in all_f_types:
            full_old_sub = full_old_events[full_old_events["f_type"] == f_type]
            full_new_sub = full_new_events[full_new_events["f_type"] == f_type]
            quality_old_sub = quality_old_events[quality_old_events["f_type"] == f_type]
            quality_new_sub = quality_new_events[quality_new_events["f_type"] == f_type]

            img_base64_full = compare_column_barplot(
                full_old_sub,
                full_new_sub,
                column="reloc",
                x_label="Relocation",
                title=f"Full NZGMDB Relocation Comparison ({f_type})",
                bar_1_label="Old NZGMDB",
                bar_2_label="New NZGMDB",
            )
            img_base64_quality = compare_column_barplot(
                quality_old_sub,
                quality_new_sub,
                column="reloc",
                x_label="Relocation",
                title=f"Quality NZGMDB Relocation Comparison ({f_type})",
                bar_1_label="Old NZGMDB",
                bar_2_label="New NZGMDB",
            )
            html_parts.append(
                f"<h3>Relocation Comparison for Fault Type: {f_type}</h3>"
            )
            html_parts.append("<div class='fig-grid'>")
            html_parts.append(f'<img src="data:image/png;base64,{img_base64_full}">')
            html_parts.append(f'<img src="data:image/png;base64,{img_base64_quality}">')
            html_parts.append("</div>")
    else:
        all_f_types = sorted(
            set(full_new_events["f_type"].unique())
            | set(quality_new_events["f_type"].unique())
        )
        for f_type in all_f_types:
            full_new_sub = full_new_events[full_new_events["f_type"] == f_type]
            quality_new_sub = quality_new_events[quality_new_events["f_type"] == f_type]

            img_base64 = compare_column_barplot(
                full_new_sub,
                quality_new_sub,
                column="reloc",
                x_label="Relocation",
                title=f"New NZGMDB Relocation Comparison ({f_type})",
            )
            html_parts.append(
                f"<h3>Relocation Comparison for Fault Type: {f_type}</h3>"
            )
            html_parts.append("<div class='fig-single'>")
            html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
            html_parts.append("</div>")

    # Add psa count Comparison
    html_parts.append("<h2>Quality pSA Record Count Comparison</h2>")
    html_parts.append('<div class="fig-single">')

    fig, ax = plt.subplots(figsize=(16, 6), dpi=300)

    # Generate fmin plots
    if compare_version_directory:
        filtered_quality_old = apply_fmin_filter_df(quality_old, pre_4p3=False)
        old_record_count = (~filtered_quality_old[PSA_KEYS].isna()).sum(axis=0)
        ax.plot(
            PERIODS,
            old_record_count.loc[PSA_KEYS],
            label="Old NZGMDB",
        )

    filtered_quality_new = apply_fmin_filter_df(quality_new)
    new_record_count = (~filtered_quality_new[PSA_KEYS].isna()).sum(axis=0)
    ax.plot(
        PERIODS,
        new_record_count.loc[PSA_KEYS],
        label="New NZGMDB",
    )

    ax.set_xlabel("Period (s)")
    ax.set_xscale("log")
    ax.set_ylabel("Count")
    ax.set_xlim(0.01, 10.0)
    ax.legend()
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")

    fig.tight_layout()

    # Convert to base64
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    # Embed in HTML
    html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
    html_parts.append("</div>")

    # Add IM Compare
    html_parts.append("<h2>IM Comparison</h2>")
    html_parts.append('<div class="fig-single">')

    quality_old_record_index = quality_old.set_index("record_id")
    quality_new_record_index = quality_new.set_index("record_id")

    # IM Compare
    shared_record_ids = np.intersect1d(
        quality_old_record_index.index.values.astype(str),
        quality_new_record_index.index.values.astype(str),
    )

    plot_ims = [
        "PGV",
        "PGA",
        "pSA_0.01",
        "pSA_0.1",
        "pSA_0.5",
        "pSA_1.0",
        "pSA_3.0",
        "pSA_10.0",
    ]

    fig, axs = get_fig_axes(len(plot_ims), 2, -1, ind_figsize=(8, 6))

    for i, (cur_im, cur_ax) in enumerate(zip(plot_ims, axs)):
        cur_old = quality_old_record_index.loc[shared_record_ids, cur_im]
        cur_new = quality_new_record_index.loc[shared_record_ids, cur_im]

        cur_max = max(cur_old.max(), cur_new.max())

        cur_ax.scatter(cur_old, cur_new, s=1)
        cur_ax.set_xlabel("Old NZGMDB")
        cur_ax.set_ylabel("New NZGMDB")
        cur_ax.set_title(cur_im)
        cur_ax.plot(
            [0, cur_max], [0, cur_max], color="black", linestyle="--", linewidth=0.5
        )
        cur_ax.set_xlim(0, cur_max)
        cur_ax.set_ylim(0, cur_max)

        # If the cur_im is PGA or includes pSA, set the x and y scales to log
        if cur_im == "PGA" or cur_im.startswith("pSA_"):
            cur_ax.set_xscale("log")
            cur_ax.set_yscale("log")
            cur_ax.set_xlim(0.001, cur_max)
            cur_ax.set_ylim(0.001, cur_max)

        cur_ax.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")

    # Convert to base64
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    # Embed in HTML
    html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
    html_parts.append("</div>")

    html_parts.append(
        """
    <h2>GMM Input Comparison</h2>
    <div class="fig-grid">
    """
    )
    # GMM Input comparison
    input_cols = [
        "ev_lat",
        "ev_lon",
        "ev_depth",
        "mag",
        "strike",
        "dip",
        "rake",
        "z_tor",
        "r_jb",
        "r_rup",
        "r_x",
        "Vs30",
        "Z1.0",
        "Z2.5",
    ]
    # Generate figures and embed as base64
    for i, col in enumerate(input_cols):
        cur_x_data = quality_old_record_index.loc[shared_record_ids, col].values
        cur_y_data = quality_new_record_index.loc[shared_record_ids, col].values
        nan_mask = np.isnan(cur_x_data) | np.isnan(cur_y_data)
        cur_x_data = cur_x_data[~nan_mask]
        cur_y_data = cur_y_data[~nan_mask]
        lims = (
            np.quantile(cur_x_data, 0.01),
            np.quantile(cur_x_data, 0.99),
        )
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.scatter(cur_x_data, cur_y_data, alpha=0.5, s=1)
        ax.plot(lims, lims, color="k")
        ax.set_xlabel("Old NZGMDB")
        ax.set_ylabel("New NZGMDB")
        ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_title(f"{col} - N: {len(cur_x_data)}")
        fig.tight_layout()

        # Convert to base64
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        # Embed in HTML
        html_parts.append(f'<img src="data:image/png;base64,{img_base64}">')
    # End of Section
    html_parts.append("</div>")

    html_parts.append("</body></html>")
    # Save report
    with open(output_file, "w") as f:
        f.write("".join(html_parts))
