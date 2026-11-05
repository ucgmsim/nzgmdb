"""Tests for merging flatfile databases and filtering them into the quality database."""

from pathlib import Path

import pandas as pd

from nzgmdb.data_processing import merge_flatfiles, quality_db
from nzgmdb.management import file_structure


def make_flatfile(record_ids: list[str], value: float) -> pd.DataFrame:
    """
    Make a flatfile with every column the merge and filter steps key on.

    Parameters
    ----------
    record_ids : list[str]
        The record ids in the form evid_sta_chan_loc
    value : float
        Value for the PGA column, to check which database a row came from

    Returns
    -------
    pd.DataFrame
        The flatfile dataframe
    """
    parts = [record_id.split("_") for record_id in record_ids]
    return pd.DataFrame(
        {
            "record_id": record_ids,
            "record": record_ids,
            "component": "X",
            "evid": [part[0] for part in parts],
            "sta": [part[1] for part in parts],
            "chan": [part[2] for part in parts],
            "loc": [part[3] for part in parts],
            "provider": "GeoNet",
            "net": "NZ",
            "start_time": "2000-01-01",
            "datetime": "2020-01-01",
            "PGA": value,
        }
    )


def write_db(db_dir: Path, record_ids: list[str], value: float):
    """
    Write every final flatfile for a database.

    Parameters
    ----------
    db_dir : Path
        The directory to write the flatfiles to
    record_ids : list[str]
        The record ids in the form evid_sta_chan_loc
    value : float
        Value for the PGA column
    """
    db_dir.mkdir(parents=True, exist_ok=True)
    df = make_flatfile(record_ids, value)
    for flatfile_name in file_structure.FlatfileNames:
        merge_flatfiles.write_flatfile(df, db_dir / flatfile_name)


def test_merge_dbs(tmp_path: Path):
    """Check the merged database replaces and appends rows and keeps each file format."""
    main_dir = tmp_path / "main"
    to_merge_dir = tmp_path / "to_merge"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    write_db(main_dir, ["1_STA1_HN_20", "1_STA2_HN_20"], 1.0)
    write_db(to_merge_dir, ["1_STA2_HN_20", "2_STA3_HN_20"], 2.0)

    merge_flatfiles.merge_dbs(main_dir, to_merge_dir, output_dir)

    # The event tables are merged per evid, the rest per record / station
    event_tables = [
        file_structure.FlatfileNames.EARTHQUAKE_SOURCE_TABLE,
        file_structure.FlatfileNames.EARTHQUAKE_SOURCE_GEOMETRY,
    ]
    for flatfile_name in file_structure.FlatfileNames:
        merged_df = merge_flatfiles.read_flatfile(output_dir / flatfile_name)
        if flatfile_name in event_tables:
            pga = merged_df.set_index("evid")["PGA"].to_dict()
            assert pga == {"1": 2.0, "2": 2.0}, flatfile_name
        else:
            pga = merged_df.set_index("sta")["PGA"].to_dict()
            assert pga == {"STA1": 1.0, "STA2": 2.0, "STA3": 2.0}, flatfile_name

    assert (
        output_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD0_ORIENTATION_FLAT
    ).suffix == ".parquet"


def test_filter_flatfiles_on_catalouge(tmp_path: Path):
    """Check every flatfile, including the station table, is filtered and saved as CSV."""
    flatfile_dir = tmp_path / "flatfiles"
    output_dir = tmp_path / "quality_db"
    output_dir.mkdir()
    write_db(flatfile_dir, ["1_STA1_HN_20", "1_STA2_HN_20"], 1.0)
    rotd50_flat = make_flatfile(["1_STA1_HN_20"], 1.0)

    quality_db.filter_flatfiles_on_catalouge(flatfile_dir, output_dir, rotd50_flat)

    for flatfile_name in [
        file_structure.FlatfileNames.STATION_TABLE,
        file_structure.FlatfileNames.MULTI_EVENT_TABLE,
        file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD0_ORIENTATION_FLAT,
        file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD100_ORIENTATION_FLAT,
    ]:
        filtered_df = pd.read_csv((output_dir / flatfile_name).with_suffix(".csv"))
        assert filtered_df["sta"].tolist() == ["STA1"], flatfile_name
    assert not list(output_dir.glob("*.parquet"))


def test_filter_station_table_on_chan_loc(tmp_path: Path):
    """Check the station table keeps only the sta, chan and loc used by the records."""
    flatfile_dir = tmp_path / "flatfiles"
    output_dir = tmp_path / "quality_db"
    output_dir.mkdir()
    write_db(flatfile_dir, ["1_STA1_HN_20"], 1.0)
    # Same station with several channel and location codes
    station_df = pd.DataFrame(
        {
            "sta": ["STA1", "STA1", "STA1", "STA1", "STA2"],
            "chan": ["HN", "HN", "BN", "HN", "HN"],
            "loc": ["20", "10", "20", "00", "20"],
        }
    )
    station_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.STATION_TABLE, index=False
    )
    rotd50_flat = make_flatfile(["1_STA1_HN_20", "2_STA1_HN_00"], 1.0)

    quality_db.filter_flatfiles_on_catalouge(flatfile_dir, output_dir, rotd50_flat)

    filtered_df = pd.read_csv(
        output_dir / file_structure.FlatfileNames.STATION_TABLE, dtype={"loc": str}
    )
    assert filtered_df[["sta", "chan", "loc"]].values.tolist() == [
        ["STA1", "HN", "20"],
        ["STA1", "HN", "00"],
    ]
