"""
This module contains the functions to merge different flatfiles together to create the final flatfiles
"""

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure


def process_im_batch(
    batch_files: list[str],
    batch_id: int,
    im_dir: Path,
    tmp_dir: Path,
    is_parquet: bool,
    fas_columns: list[str],
):
    fas_comps = ["000", "090", "ver", "geom", "eas"]
    psa_comps = ["000", "090", "ver", "geom"]
    rotd_comps = ["rotd0", "rotd50", "rotd100"]

    filename_mapping = {
        "000_psa": file_structure.PreFlatfileNames.IM_MERGE_000,
        "000_fas": file_structure.PreFlatfileNames.IM_MERGE_000_FAS,
        "090_psa": file_structure.PreFlatfileNames.IM_MERGE_090,
        "090_fas": file_structure.PreFlatfileNames.IM_MERGE_090_FAS,
        "ver_psa": file_structure.PreFlatfileNames.IM_MERGE_VER,
        "ver_fas": file_structure.PreFlatfileNames.IM_MERGE_VER_FAS,
        "geom_psa": file_structure.PreFlatfileNames.IM_MERGE_GEOM,
        "geom_fas": file_structure.PreFlatfileNames.IM_MERGE_GEOM_FAS,
        "rotd0_psa": file_structure.PreFlatfileNames.IM_MERGE_ROTD0,
        "rotd50_psa": file_structure.PreFlatfileNames.IM_MERGE_ROTD50,
        "rotd100_psa": file_structure.PreFlatfileNames.IM_MERGE_ROTD100,
        "eas_fas": file_structure.PreFlatfileNames.IM_MERGE_EAS_FAS,
    }
    scalar_columns = [
        "PGA",
        "PGV",
        "PGD",
        "CAV",
        "CAV5",
        "AI",
        "Ds575",
        "Ds595",
    ]

    batch_dir = tmp_dir / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    writers = {}

    def write_chunk(df: pd.DataFrame, output_file: Path):
        if df.empty:
            return

        table = pa.Table.from_pandas(df, preserve_index=False)

        if output_file not in writers:
            writers[output_file] = pq.ParquetWriter(
                output_file,
                table.schema,
                compression="zstd",
            )

        writers[output_file].write_table(table)

    try:
        for rel_path in batch_files:

            im_file = im_dir / rel_path

            if not im_file.exists():
                continue

            if is_parquet:
                df = pd.read_parquet(im_file)
            else:
                df = pd.read_csv(im_file)

            # Split the record id into evid, sta, chan, loc
            record_parts = df["record_id"].str.split(
                "_",
                n=3,
                expand=True,
            )
            record_parts.columns = [
                "evid",
                "sta",
                "chan",
                "loc",
            ]
            df = pd.concat(
                [df, record_parts],
                axis=1,
            )

            psa_columns = [c for c in df.columns if c.startswith("pSA")]

            existing_fas_cols = [c for c in df.columns if c.startswith("FAS_")]

            fas_df = {}

            for col in existing_fas_cols:
                freq = float(col.removeprefix("FAS_"))
                new_col = f"FAS_{freq:.6g}"

                s = df[col]

                if s.dtype == object:
                    s = pd.to_numeric(
                        s.astype(str).str.replace("e/", "e-", regex=False),
                        errors="coerce",
                    )

                fas_df[new_col] = s

            if existing_fas_cols:
                df = df.drop(columns=existing_fas_cols)

            if fas_df:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            fas_df,
                            index=df.index,
                        ),
                    ],
                    axis=1,
                )

            non_fas_cols = [c for c in df.columns if not c.startswith("FAS_")]

            df = df.reindex(
                columns=[
                    *non_fas_cols,
                    *fas_columns,
                ]
            )

            columns_remove_rotd = [
                "CAV",
                "CAV5",
                "AI",
                "Ds575",
                "Ds595",
            ] + fas_columns

            columns_remove_fas = scalar_columns + psa_columns

            for comp, comp_rows in df.groupby("component"):

                if comp in fas_comps:

                    comp_rows_fas = comp_rows.drop(
                        columns=columns_remove_fas,
                        errors="ignore",
                    )

                    write_chunk(
                        comp_rows_fas,
                        batch_dir / filename_mapping[f"{comp}_fas"],
                    )

                if comp in psa_comps:

                    comp_rows_psa = comp_rows.drop(
                        columns=fas_columns,
                        errors="ignore",
                    )

                    write_chunk(
                        comp_rows_psa,
                        batch_dir / filename_mapping[f"{comp}_psa"],
                    )

                if comp in rotd_comps:

                    comp_rows_rotd = comp_rows.drop(
                        columns=columns_remove_rotd,
                        errors="ignore",
                    )

                    write_chunk(
                        comp_rows_rotd,
                        batch_dir / filename_mapping[f"{comp}_psa"],
                    )

    finally:
        for writer in writers.values():
            writer.close()


def merge_component_files(
    component_name: str,
    tmp_dir: Path,
    output_file: Path,
):
    files = sorted(tmp_dir.glob(f"batch_*/*{component_name}.parquet"))

    writer = None

    try:
        for ffp in files:

            table = pq.read_table(ffp)

            if writer is None:
                writer = pq.ParquetWriter(
                    output_file,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(table)

    finally:
        if writer:
            writer.close()


def merge_im_data(
    im_dir: Path,
    output_dir: Path,
    records_ffp: Path,
    n_procs: int = 1,
    batch_size: int = 50000,
    is_parquet: bool = False,
):
    """
    Merge the IM data into component / fas split files.
    """

    records_df = pd.read_csv(records_ffp)

    suffix = "parquet" if is_parquet else "csv"

    records_df["evid"] = records_df["record_id"].str.partition("_")[0]

    records_df["im_file"] = (
        records_df["evid"] + "/" + records_df["record_id"] + f"_IM.{suffix}"
    )

    batches = [
        records_df["im_file"].iloc[i : i + batch_size].tolist()
        for i in range(0, len(records_df), batch_size)
    ]

    config = cfg.Config()
    fas_frequencies = np.logspace(
        np.log10(config.get_value("common_frequency_start")),
        np.log10(config.get_value("common_frequency_end")),
        num=config.get_value("common_frequency_num"),
    )
    fas_columns = [f"FAS_{freq:.6g}" for freq in fas_frequencies]

    with mp.Pool(n_procs) as pool:
        pool.starmap(
            process_im_batch,
            [
                (
                    batch,
                    batch_id,
                    im_dir,
                    output_dir / "im_merge_batch_dir",
                    is_parquet,
                    fas_columns,
                )
                for batch_id, batch in enumerate(batches)
            ],
        )

    for component in [
        "000",
        "000_fas",
        "090",
        "090_fas",
        "ver",
        "ver_fas",
        "geom",
        "geom_fas",
        "rotd0",
        "rotd50",
        "rotd100",
        "eas_fas",
    ]:
        merge_component_files(
            component,
            output_dir / "im_merge_batch_dir",
            output_dir / f"im_merge_{component}.parquet",
        )


def add_ground_level(
    station_df: pd.DataFrame,
    gm_im_df_flat: pd.DataFrame,
):
    """
    Add in the is ground level location elevation information to the gm_im_df_flat dataframe

    Parameters
    ----------
    station_df : pd.DataFrame
        The station dataframe containing the station information such as loc_elev
    gm_im_df_flat : pd.DataFrame
        The ground motion IM dataframe to add the ground level information to

    Returns
    -------
    pd.DataFrame
        The ground motion IM dataframe with the ground level information added
    """
    # Get the recorders information for location codes
    config = cfg.Config()
    locations_url = config.get_value("locations_url")
    locations_df = pd.read_csv(locations_url)
    # Ensure the Station and Location pairings are unique
    locations_df = locations_df.drop_duplicates(subset=["Station", "Location"])

    # Merge the locations_df with the station_df to get extra loc_elev from locations_df
    station_df = station_df.merge(
        locations_df[["Station", "Location", "Depth"]],
        left_on=["sta", "loc"],
        right_on=["Station", "Location"],
        how="outer",
    )

    # Remove rows where sta is NaN
    station_df = station_df[station_df["sta"].notna()]

    # Fill the loc_elev with the locations depth when the loc_elev is NaN
    station_df["loc_elev"] = station_df["loc_elev"].fillna(station_df["Depth"])

    # Apply negative to the loc_elev column to convert to elevation
    station_df["loc_elev"] = -station_df["loc_elev"]

    # Fill NaN start and end times
    station_df["start_time"] = station_df["start_time"].fillna(
        pd.Timestamp("2000-01-01")
    )
    station_df["end_time"] = station_df["end_time"].fillna(pd.Timestamp.max)

    # Ensure datetime dtypes
    # def _to_py_datetime(val: object) -> object:
    #     """
    #     Convert UTCDateTime to python datetime.datetime if needed.
    #
    #     Parameters
    #     ----------
    #     val : object
    #         The value to convert.
    #
    #     Returns
    #     -------
    #     object
    #         The converted value.
    #     """
    #     if isinstance(val, UTCDateTime):
    #         return val.datetime
    #     return val
    #
    # station_df["start_time"] = station_df["start_time"].apply(_to_py_datetime)
    # station_df["end_time"] = station_df["end_time"].apply(_to_py_datetime)

    # def ensure_utc(series: pd.Series) -> pd.Series:
    #     """
    #     Ensure a pandas Series of datetimes is timezone-aware in UTC.
    #
    #     Parameters
    #     ----------
    #     series : pd.Series
    #         The pandas Series to ensure is timezone-aware in UTC.
    #
    #     Returns
    #     -------
    #     pd.Series
    #         The timezone-aware pandas Series in UTC.
    #     """
    #     # coerce to datetime first, then ensure UTC tz (convert if already tz-aware, localize if naive)
    #     s = pd.to_datetime(series, errors="coerce")
    #     if pd.api.types.is_datetime64tz_dtype(s.dtype):
    #         return s.dt.tz_convert("UTC")
    #     return s.dt.tz_localize("UTC")

    # Normalize both frames to UTC before sorting / merge_asof
    # station_df["start_time"] = ensure_utc(station_df["start_time"])
    # station_df["end_time"] = ensure_utc(station_df["end_time"])
    station_df["start_time"] = pd.to_datetime(
        station_df["start_time"], errors="coerce", utc=True
    )
    station_df["end_time"] = pd.to_datetime(
        station_df["end_time"], errors="coerce", utc=True
    )

    station_df["start_time"] = pd.to_datetime(station_df["start_time"])
    station_df["end_time"] = pd.to_datetime(station_df["end_time"])
    gm_im_df_flat["datetime"] = pd.to_datetime(gm_im_df_flat["datetime"])

    merged = pd.merge(
        gm_im_df_flat.reset_index(), station_df, on=["sta", "loc", "chan"], how="left"
    )
    valid_matches = merged[
        (merged["datetime"] >= merged["start_time"])
        & (merged["datetime"] <= merged["end_time"])
    ]
    # Sort to ensure deterministic selection of the first match, then drop duplicates.
    first_matches = valid_matches.sort_values("start_time").drop_duplicates(
        subset="index", keep="first"
    )
    # Map the loc_elev back to the gm_im_df_flat dataframe.
    gm_im_df_flat["loc_elev"] = gm_im_df_flat.index.map(
        first_matches.set_index("index")["loc_elev"]
    )

    # Replace -0.0 with 0.0 in the DataFrame
    gm_im_df_flat = gm_im_df_flat.replace(-0.0, 0.0)

    # Add in a flag for when the location elevation is 0
    # Group by 'evid', 'sta', and 'chan'
    grouped = gm_im_df_flat.groupby(["evid", "sta", "chan"])

    def custom_idxmin(group: pd.DataFrame):
        """
        Custom function to handle NaN values and find the index of the row with the loc_elev value closest to 0

        Parameters
        ----------
        group : pd.DataFrame
            The group of the DataFrame with the same 'evid', 'sta', and 'chan' values

        Returns
        -------
        int | None
            The index of the row with the loc_elev value closest to 0, or None if all values are NaN
        """
        # Filter out loc_elev values greater than 5 metres (In either direction)
        group = group[group["loc_elev"].abs() <= config.get_value("locations_max_elev")]
        if group["loc_elev"].isna().all():
            return None
        # Find the index of the row with the loc_elev value closest to 0
        return (group["loc_elev"].abs()).idxmin(skipna=True)

    # Find the index of the row with the smallest loc_elev value for each group, excluding NaN values
    idx_min_loc_elev = grouped.apply(custom_idxmin)

    gm_im_df_flat["is_ground_level"] = False
    if len(idx_min_loc_elev) > 0:
        # Set the flag to True for the rows with the smallest loc_elev value
        record_ids = gm_im_df_flat.loc[idx_min_loc_elev.dropna(), "record_id"]
        gm_im_df_flat.loc[
            gm_im_df_flat["record_id"].isin(record_ids), "is_ground_level"
        ] = True

    # For Locations not found in the dataframe, set the loc_elev to 0 only if there is just 1 location
    # Also set the is_ground_level to True
    gm_im_df_flat.loc[
        gm_im_df_flat["loc_elev"].isna()
        & gm_im_df_flat.groupby(["evid", "sta", "chan"])["loc"]
        .transform("nunique")
        .eq(1),
        ["is_ground_level", "loc_elev"],
    ] = [True, 0.0]

    return gm_im_df_flat


def merge_flatfiles(main_dir: Path, bypass_records_ffp: Path = None):
    """
    Merge the flatfiles into the final flatfiles, separating the components
    and ensuring that the data contains only the unique events and sites that made it to the IM calculation

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    bypass_records_ffp : Path
        The full file path to the bypass records file, which includes a custom fmin, fmax, and p_wave_ix
    """
    # Get the flatfile directory
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    # flatfile_dir = main_dir / "tmp"

    # Split record_id
    # new_cols = df["record_id"].str.split("_", expand=True)
    # new_cols.columns = ["evid", "sta", "chan", "loc"]
    #
    # df = pd.concat([df, new_cols], axis=1)

    # Load the files
    gmc_ffp = flatfile_dir / file_structure.FlatfileNames.GMC_PREDICTIONS
    if not gmc_ffp.exists():
        gmc_df = pd.DataFrame(
            columns=[
                "record",
                "score_mean_X",
                "fmin_mean_X",
                "multi_mean_X",
                "score_mean_Y",
                "fmin_mean_Y",
                "multi_mean_Y",
                "score_mean_Z",
                "fmin_mean_Z",
                "multi_mean_Z",
            ]
        )
    else:
        gmc_df = pd.read_csv(gmc_ffp)

        # Define the columns to be grouped
        columns = ["score_mean", "fmin_mean", "multi_mean"]

        # Group by 'record' and 'component', then aggregate the columns
        gmc_df = gmc_df.groupby(["record", "component"])[columns].mean().unstack()

        # Join the column names to score_mean_X etc.
        gmc_df.columns = ["_".join(col) for col in gmc_df.columns]

        gmc_df = gmc_df.reset_index()

    fmax_ffp = flatfile_dir / file_structure.FlatfileNames.FMAX
    fmax_df = (
        pd.DataFrame(columns=["record_id", "fmax_000", "fmax_090", "fmax_ver"])
        if fmax_ffp is None or not fmax_ffp.stat().st_size
        else pd.read_csv(fmax_ffp)
    )
    event_df = pd.read_csv(
        flatfile_dir
        / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_AFTERSHOCKS,
        dtype={"evid": str},
    )
    geo_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_GEOMETRY,
        dtype={"evid": str},
    )
    sta_mag_df = pd.read_csv(
        flatfile_dir
        / file_structure.PreFlatfileNames.STATION_MAGNITUDE_TABLE_EXTRACTION,
        dtype={"evid": str},
    )
    if (flatfile_dir / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE).exists():
        phase_table_df = pd.read_csv(
            flatfile_dir / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE
        )
    else:
        phase_table_df = pd.DataFrame(
            columns=[
                "record_id",
                "p_wave_ix",
                "p_wave_datetime",
                "p_wave_prob",
                "s_wave_ix",
                "s_wave_datetime",
                "s_wave_prob",
                "evid_datetime",
            ]
        )
    prop_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.PROPAGATION_TABLE,
        dtype={"evid": str},
    )
    im_df = pd.read_parquet(
        flatfile_dir
        / "im_merge_batch_dir"
        / file_structure.PreFlatfileNames.IM_MERGE_ROTD50,
        columns=["record_id", "evid", "sta", "chan", "loc"],
    )
    site_basin_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.SITE_TABLE
    )
    station_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.STATION_TABLE
    )
    station_extraction_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.STATION_EXTRACTION_TABLE_GEONET,
        dtype={"evid": str},
    )
    multi_event_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.MULTI_EVENT_TABLE,
    )

    # Ensure correct strike and rake values
    event_df.loc[event_df.strike == 360, "strike"] = 0
    event_df.loc[event_df.rake > 180, "rake"] -= 360

    # Get unique events that made it to the IM calculation
    unique_events = im_df.evid.unique()
    # Ensure that the other dfs only have the unique events
    event_df = event_df[event_df.evid.isin(unique_events)]
    geo_df = geo_df[geo_df.evid.isin(unique_events)]

    phase_table_df = phase_table_df[
        phase_table_df["record_id"].isin(im_df["record_id"])
    ]

    multi_event_df = multi_event_df[
        multi_event_df["record_id"].isin(im_df["record_id"])
    ]

    # Ensure that the site_basin_df only has the unique sites found in the im_df
    unique_sites = im_df["sta"].unique()
    site_basin_df = site_basin_df[site_basin_df["sta"].isin(unique_sites)]
    station_df = station_df[station_df["sta"].isin(unique_sites)]

    # Ensure the station magnitude table only has values of events and station pairs available in the im_df
    unique_pairs_df = im_df[["evid", "sta"]].drop_duplicates()
    sta_mag_df = pd.merge(sta_mag_df, unique_pairs_df, on=["evid", "sta"], how="inner")

    # Ensure the station extraction table only has values of events and station pairs available in the im_df
    station_extraction_df = pd.merge(
        station_extraction_df, unique_pairs_df, on=["evid", "sta"], how="inner"
    )

    # Get a list of sites not found in the site basin df
    missing_sites = set(unique_sites) - set(site_basin_df["sta"].unique())
    # Save the missing sites
    missing_sites_df = pd.DataFrame(missing_sites, columns=["sta"])
    missing_sites_df.to_csv(
        flatfile_dir / file_structure.SkippedRecordFilenames.MISSING_SITES, index=False
    )

    # Merge in the gmc df
    gmc_df = gmc_df.rename(
        columns={
            "record": "record_id",
            "score_mean_X": "score_X",
            "fmin_mean_X": "fmin_X",
            "fmax_mean_X": "fmax_X",
            "multi_mean_X": "multi_X",
            "score_mean_Y": "score_Y",
            "fmin_mean_Y": "fmin_Y",
            "fmax_mean_Y": "fmax_Y",
            "multi_mean_Y": "multi_Y",
            "score_mean_Z": "score_Z",
            "fmin_mean_Z": "fmin_Z",
            "fmax_mean_Z": "fmax_Z",
            "multi_mean_Z": "multi_Z",
        }
    )
    im_df = im_df.merge(gmc_df, on="record_id", how="left")

    # Merge in the fmax df
    fmax_df = fmax_df.rename(
        columns={
            "fmax_000": "fmax_X",
            "fmax_090": "fmax_Y",
            "fmax_ver": "fmax_Z",
        }
    )
    im_df = im_df.merge(fmax_df, on="record_id", how="left")

    # Merge event data with the IM data
    gm_im_df_flat = im_df.merge(
        event_df[
            [
                "evid",
                "datetime",
                "lat",
                "lon",
                "depth",
                "mag",
                "mag_type",
                "tect_class",
                "reloc",
                "domain_no",
                "domain_type",
                "strike",
                "dip",
                "rake",
                "f_length",
                "f_width",
                "f_type",
                "z_tor",
                "z_bor",
                "aftershock_flag_crjb0",
                "cluster_flag_crjb0",
                "aftershock_flag_crjb2",
                "cluster_flag_crjb2",
                "aftershock_flag_crjb5",
                "cluster_flag_crjb5",
                "aftershock_flag_crjb10",
                "cluster_flag_crjb10",
            ]
        ],
        on="evid",
        how="left",
    )
    gm_im_df_flat = gm_im_df_flat.rename(
        columns={"lat": "ev_lat", "lon": "ev_lon", "depth": "ev_depth"}
    )

    # Create the site basin df to merge with only 1 sta value
    merge_site_table = site_basin_df.drop_duplicates(subset=["sta"])

    # Merge in the site data
    gm_im_df_flat = gm_im_df_flat.merge(
        merge_site_table[
            [
                "sta",
                "lat",
                "lon",
                "elev",
                "Vs30",
                "Vs30_std",
                "Q_Vs30",
                "T0",
                "T0_std",
                "Q_T0",
                "Z1.0",
                "Z1.0_std",
                "Q_Z1.0",
                "Z2.5",
                "Z2.5_std",
                "Q_Z2.5",
                "site_domain_no",
            ]
        ],
        on="sta",
        how="left",
    )
    gm_im_df_flat = gm_im_df_flat.rename(
        columns={"lat": "sta_lat", "lon": "sta_lon", "elev": "sta_elev"}
    )

    # Add in the ground level location elevation information
    gm_im_df_flat = add_ground_level(station_df, gm_im_df_flat)

    # Add in multi_event information
    gm_im_df_flat = gm_im_df_flat.merge(
        multi_event_df[
            [
                "record_id",
                "stalta_score",
                "sync_event",
            ]
        ],
        on="record_id",
        how="left",
    )

    # Remove duplicated columns in prop_df
    prop_df["evid_sta"] = prop_df["evid"].astype(str) + "_" + prop_df["sta"].astype(str)
    prop_df = prop_df.drop_duplicates(subset=["evid_sta"])
    prop_df = prop_df.drop(columns=["evid_sta"])

    # Merge in the propagation data
    gm_im_df_flat = gm_im_df_flat.merge(
        prop_df[
            [
                "evid",
                "sta",
                "r_epi",
                "r_hyp",
                "r_jb",
                "r_rup",
                "r_avg",
                "r_x",
                "r_y",
                "r_tvz",
                "r_xvf",
            ]
        ],
        on=["evid", "sta"],
        how="left",
    )

    # Merge in the bypass information
    if bypass_records_ffp is not None:
        bypass_df = pd.read_csv(bypass_records_ffp)
        gm_im_df_flat = gm_im_df_flat.merge(
            bypass_df[
                [
                    "record_id",
                    "fmax_000",
                    "fmax_090",
                    "fmax_ver",
                    "fmin_000",
                    "fmin_090",
                    "fmin_ver",
                ]
            ],
            on="record_id",
            how="left",
            suffixes=("", "_bypass"),
        )
        for bypass_col, col in [
            ("fmin_000", "fmin_X"),
            ("fmin_090", "fmin_Y"),
            ("fmin_ver", "fmin_Z"),
            ("fmax_000", "fmax_X"),
            ("fmax_090", "fmax_Y"),
            ("fmax_ver", "fmax_Z"),
        ]:
            gm_im_df_flat[col] = gm_im_df_flat[col].fillna(gm_im_df_flat[bypass_col])
        gm_im_df_flat = gm_im_df_flat.drop(
            columns=[
                "fmax_000",
                "fmax_090",
                "fmax_ver",
                "fmin_000",
                "fmin_090",
                "fmin_ver",
            ]
        )

        # Add any extra p_wave_ix values to the phase_table_df
        new_records = bypass_df[
            ~bypass_df["record_id"].isin(phase_table_df["record_id"])
        ]
        # remove p_wave_ix of nan
        new_records = new_records.dropna(subset=["p_wave_ix"])
        new_records = new_records[["record_id", "p_wave_ix"]]
        phase_table_df = pd.concat([phase_table_df, new_records])

    # Add in the default fmin values if they are nan
    config = cfg.Config()
    default_fmin = config.get_value("low_cut_default")
    for col in ["fmin_X", "fmin_Y", "fmin_Z"]:
        gm_im_df_flat[col] = gm_im_df_flat[col].fillna(default_fmin)

    # Add in colunms for fmin_max and fmin_highpass
    gm_im_df_flat["fmin_max_h"] = gm_im_df_flat[["fmin_X", "fmin_Y"]].max(axis=1)
    gm_im_df_flat["HPF_h"] = gm_im_df_flat["fmin_max_h"] / 1.25
    gm_im_df_flat["HPF_v"] = gm_im_df_flat["fmin_Z"] / 1.25

    gm_im_df_flat["LPF_h"] = gm_im_df_flat[["fmax_X", "fmax_Y"]].min(axis=1)
    gm_im_df_flat["LPF_v"] = gm_im_df_flat["fmax_Z"]

    # Save final outputs
    event_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.EARTHQUAKE_SOURCE_TABLE, index=False
    )
    geo_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.EARTHQUAKE_SOURCE_GEOMETRY,
        index=False,
    )
    sta_mag_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.STATION_MAGNITUDE_TABLE, index=False
    )
    station_extraction_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.STATION_EXTRACTION_TABLE,
        index=False,
    )
    multi_event_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.MULTI_EVENT_TABLE, index=False
    )
    phase_table_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.PHASE_ARRIVAL_TABLE, index=False
    )
    site_basin_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.SITE_TABLE, index=False
    )
    station_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.STATION_TABLE, index=False
    )
    prop_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.PROPAGATION_TABLE, index=False
    )

    # Sort the rows
    gm_im_df_flat = gm_im_df_flat.sort_values(["datetime", "sta"])

    psa_periods = np.asarray(config.get_value("psa_periods"))
    fas_frequencies = np.logspace(
        np.log10(config.get_value("common_frequency_start")),
        np.log10(config.get_value("common_frequency_end")),
        num=config.get_value("common_frequency_num"),
    )
    psa_columns = [f"pSA_{p}" for p in psa_periods]
    fas_columns = [f"FAS_{f}" for f in fas_frequencies]
    columns = (
        [
            "record_id",
            "datetime",
            "evid",
            "sta",
            "loc",
            "chan",
            "ev_lat",
            "ev_lon",
            "ev_depth",
            "mag",
            "mag_type",
            "tect_class",
            "reloc",
            "domain_no",
            "domain_type",
            "strike",
            "dip",
            "rake",
            "f_length",
            "f_width",
            "f_type",
            "z_tor",
            "z_bor",
            "sta_lat",
            "sta_lon",
            "sta_elev",
            "loc_elev",
            "is_ground_level",
            "r_epi",
            "r_hyp",
            "r_jb",
            "r_rup",
            "r_avg",
            "r_x",
            "r_y",
            "r_tvz",
            "r_xvf",
            "Vs30",
            "Vs30_std",
            "Q_Vs30",
            "T0",
            "T0_std",
            "Q_T0",
            "Z1.0",
            "Z1.0_std",
            "Q_Z1.0",
            "Z2.5",
            "Z2.5_std",
            "Q_Z2.5",
            "site_domain_no",
            "PGA",
            "PGV",
            "PGD",
            "CAV",
            "CAV5",
            "AI",
            "Ds575",
            "Ds595",
            "score_X",
            "fmin_X",
            "fmax_X",
            "multi_X",
            "score_Y",
            "fmin_Y",
            "fmax_Y",
            "multi_Y",
            "score_Z",
            "fmin_Z",
            "fmax_Z",
            "multi_Z",
            "stalta_score",
            "sync_event",
            "HPF_h",
            "HPF_v",
            "LPF_h",
            "LPF_v",
            "aftershock_flag_crjb0",
            "cluster_flag_crjb0",
            "aftershock_flag_crjb2",
            "cluster_flag_crjb2",
            "aftershock_flag_crjb5",
            "cluster_flag_crjb5",
            "aftershock_flag_crjb10",
            "cluster_flag_crjb10",
        ]
        + psa_columns
        + fas_columns
    )

    filename_mapping = {
        file_structure.PreFlatfileNames.IM_MERGE_000: file_structure.FlatfileNames.GROUND_MOTION_IM_000_FLAT,
        file_structure.PreFlatfileNames.IM_MERGE_000_FAS: file_structure.FlatfileNames.GROUND_MOTION_IM_000_FAS_FLAT,
        file_structure.PreFlatfileNames.IM_MERGE_090: file_structure.FlatfileNames.GROUND_MOTION_IM_090_FLAT,
        file_structure.PreFlatfileNames.IM_MERGE_090_FAS: file_structure.FlatfileNames.GROUND_MOTION_IM_090_FAS_FLAT,
        file_structure.PreFlatfileNames.IM_MERGE_VER: file_structure.FlatfileNames.GROUND_MOTION_IM_VER_FLAT,
        file_structure.PreFlatfileNames.IM_MERGE_VER_FAS: file_structure.FlatfileNames.GROUND_MOTION_IM_VER_FAS_FLAT,
        file_structure.PreFlatfileNames.IM_MERGE_GEOM: file_structure.FlatfileNames.GROUND_MOTION_IM_GEOM_FLAT,
        file_structure.PreFlatfileNames.IM_MERGE_GEOM_FAS: file_structure.FlatfileNames.GROUND_MOTION_IM_GEOM_FAS_FLAT,
        file_structure.PreFlatfileNames.IM_MERGE_ROTD0: file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD0_FLAT,
        file_structure.PreFlatfileNames.IM_MERGE_ROTD50: file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD50_FLAT,
        file_structure.PreFlatfileNames.IM_MERGE_ROTD100: file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD100_FLAT,
        file_structure.PreFlatfileNames.IM_MERGE_EAS_FAS: file_structure.FlatfileNames.GROUND_MOTION_IM_EAS_FAS_FLAT,
    }

    # Save the flatfiles
    gm_im_df_flat = gm_im_df_flat.drop(columns=["evid", "sta", "chan", "loc"])

    for im_merge, final_output in filename_mapping.items():
        # Read parquet
        df = pd.read_parquet(flatfile_dir / "im_merge_batch_dir" / Path(im_merge).stem)

        # Drop component column
        df = df.drop(columns=["component"])

        # Merge in the gm_im_df_flat columns
        df = df.merge(gm_im_df_flat, on="record_id", how="left")

        # Keep only columns that exist in the dataframe
        existing_columns = [col for col in columns if col in df.columns]

        # Reorder columns
        df = df[existing_columns]

        # Save with gzip compression
        df.to_parquet(
            flatfile_dir / final_output,
            compression="gzip",
            index=False,
        )


def merge_dbs(
    flatfile_db_dir: Path,
    to_merge_db_dir: Path,
    output_dir: Path,
):
    """
    Merge the databases into a single database, where all the results from the to_merge_db_dir
    are adding or replacing the flatfile_db_dir results. The output is saved to the output_dir

    Parameters
    ----------
    flatfile_db_dir : Path
        The main database directory
    to_merge_db_dir : Path
        The directory of the database to merge into the main database
    output_dir : Path
        The directory to save the merged database
    """
    # For each file in the flatfiles, merge the to_merge_db_dir into the main_db_dir
    for flatfile_name in file_structure.FlatfileNames:
        main_df = pd.read_csv(flatfile_db_dir / flatfile_name, dtype={"evid": str})
        to_merge_df = pd.read_csv(to_merge_db_dir / flatfile_name, dtype={"evid": str})

        if flatfile_name in [
            file_structure.FlatfileNames.EARTHQUAKE_SOURCE_TABLE,
            file_structure.FlatfileNames.EARTHQUAKE_SOURCE_GEOMETRY,
        ]:
            # Merge based on evid, replace values if they exist and append new ones
            main_df = pd.concat([main_df, to_merge_df]).drop_duplicates(
                subset=["evid"], keep="last"
            )
            # Re-sort based on evid
            main_df = main_df.sort_values("datetime")
        elif flatfile_name == file_structure.FlatfileNames.STATION_MAGNITUDE_TABLE:
            # Make the unique record_id col with the columns evid_sta_chan_loc
            main_df["record_id"] = (
                main_df["evid"]
                + "_"
                + main_df["sta"]
                + "_"
                + main_df["chan"]
                + "_"
                + main_df["loc"].astype(str)
            )
            to_merge_df["record_id"] = (
                to_merge_df["evid"]
                + "_"
                + to_merge_df["sta"]
                + "_"
                + to_merge_df["chan"]
                + "_"
                + to_merge_df["loc"].astype(str)
            )
            # Merge on record_id, replace values if they exist and append new ones
            main_df = pd.concat([main_df, to_merge_df]).drop_duplicates(
                subset=["record_id"], keep="last"
            )
            # Re-sort based on record_id
            main_df = main_df.sort_values("record_id")
            # Remove the record_id column
            main_df = main_df.drop(columns=["record_id"])
        elif flatfile_name == file_structure.FlatfileNames.SITE_TABLE:
            # Merge based on sta, replace values if they exist and append new ones
            main_df = pd.concat([main_df, to_merge_df]).drop_duplicates(
                subset=["sta"], keep="last"
            )
            # Re-sort based on sta
            main_df = main_df.sort_values("sta")
        elif flatfile_name == file_structure.FlatfileNames.PROPAGATION_TABLE:
            # Merge based on evid_sta, replace values if they exist and append new ones
            main_df["evid_sta"] = main_df["evid"] + "_" + main_df["sta"].astype(str)
            to_merge_df["evid_sta"] = (
                to_merge_df["evid"] + "_" + to_merge_df["sta"].astype(str)
            )
            main_df = pd.concat([main_df, to_merge_df]).drop_duplicates(
                subset=["evid_sta"], keep="last"
            )
            # Re-sort based on evid_sta
            main_df = main_df.sort_values("evid_sta")
            # Remove the record_id column
            main_df = main_df.drop(columns=["evid_sta"])
        else:
            # Merge on record_id, replace values if they exist and append new ones
            main_df = pd.concat([main_df, to_merge_df]).drop_duplicates(
                subset=["record_id"], keep="last"
            )
            # If the name of the file contains "flat" sort by datetime ,sta
            if "flat" in flatfile_name:
                main_df = main_df.sort_values(["datetime", "sta"])
            else:
                # Re-sort based on record_id
                main_df = main_df.sort_values("record_id")

        # Save the merged database
        main_df.to_csv(output_dir / flatfile_name, index=False)
