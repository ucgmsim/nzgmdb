"""Tests for the batched IM calculation outputs, skips and checkpoints."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import qcore.timeseries as ts
from IM import im_calculation
from IM.ims import IM
from nzgmdb.calculation import ims
from nzgmdb.management import file_structure

INTENSITY_MEASURES = [IM.PGA, IM.CAV, IM.pSA, IM.FAS]
IM_COMPONENTS = [
    "000",
    "090",
    "ver",
    "geom",
    "rotd0",
    "rotd50",
    "rotd100",
    "rotd0_orientation",
    "rotd100_orientation",
    "eas",
]


def write_record(
    main_dir: Path, record_id: str, dt: float = 0.01, nan_middle: bool = False
):
    """
    Write the processed 000, 090 and ver ascii files for a record.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results
    record_id : str
        The record_id, as evid_sta_chan_loc
    dt : float, optional
        The time step of the record, by default 0.01
    nan_middle : bool, optional
        Whether to put a NaN in the middle of the 000 component, by default False
    """
    event_id, station = record_id.split("_")[:2]
    processed_dir = file_structure.get_processed_dir(main_dir, 2020, event_id)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for comp in ["000", "090", "ver"]:
        data = np.random.default_rng(0).standard_normal(500)
        if nan_middle and comp == "000":
            data[250] = np.nan
        ts.timeseries_to_text(
            data, processed_dir / f"{record_id}.{comp}", dt, station, comp
        )


@pytest.fixture
def fake_calculate_ims(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """
    Replace the IM calculation with a fast fake that has the IM_calculation output layout.

    The fake keeps FAS frequencies at or below Nyquist, as IM_calculation does.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The pytest monkeypatch fixture

    Returns
    -------
    list[float]
        The dt of each record the fake was called for
    """
    calls = []

    def calculate_ims(
        waveform: np.ndarray,
        dt: float,
        ims_list: list[IM],
        periods: np.ndarray,
        frequencies: np.ndarray,
        bandwidth: float,
    ) -> pd.DataFrame:
        calls.append(dt)
        columns = ["PGA", "CAV"] + [f"pSA_{period}" for period in periods]
        columns += [
            f"FAS_{im_calculation.frequency_label(freq)}"
            for freq in frequencies
            if freq <= 0.5 / dt
        ]
        im_df = pd.DataFrame(1.0, index=IM_COMPONENTS, columns=columns)
        im_df.loc[["rotd0_orientation", "rotd100_orientation"]] = 45.0
        return im_df

    monkeypatch.setattr(ims.im_calculation, "calculate_ims", calculate_ims)
    return calls


def read_component(main_dir: Path, component: str) -> pd.DataFrame:
    """
    Read a component directory as one dataset, as the later pipeline steps do.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results
    component : str
        The IM component, e.g. "rotd50" or "000_fas"

    Returns
    -------
    pd.DataFrame
        The IMs of every batch for the component
    """
    return pd.read_parquet(file_structure.get_im_component_dir(main_dir, component))


def read_skipped(main_dir: Path) -> pd.DataFrame:
    """
    Read the combined IM calculation skipped records.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results

    Returns
    -------
    pd.DataFrame
        The skipped records
    """
    return pd.read_csv(
        file_structure.get_flatfile_dir(main_dir)
        / file_structure.SkippedRecordFilenames.IM_CALC_SKIPPED_RECORDS
    )


def test_component_tables(tmp_path: Path, fake_calculate_ims: list[float]):
    """Each component directory has the right records, record columns and IM columns."""
    record_ids = ["2020p000001_ABC_HN_20", "2020p000002_DEF_BN_00"]
    for record_id in record_ids:
        write_record(tmp_path, record_id)

    ims.compute_ims_for_all_processed_records(
        tmp_path, intensity_measures=INTENSITY_MEASURES
    )

    im_merge_dir = file_structure.get_im_merge_dir(tmp_path)
    expected_dirs = {
        f"im_merge_{comp}"
        for comp in ims.PSA_COMPONENTS
        + ims.ROTD_COMPONENTS
        + ims.ORIENTATION_COMPONENTS
        + [f"{comp}_fas" for comp in ims.FAS_COMPONENTS]
    }
    found_dirs = {p.name for p in im_merge_dir.glob("im_merge_*")}
    assert found_dirs == expected_dirs

    # Scalar / pSA components keep every IM except FAS
    df_000 = read_component(tmp_path, "000")
    assert list(df_000.columns[:6]) == ims.RECORD_COLUMNS
    assert sorted(df_000["record_id"]) == record_ids
    assert set(df_000["component"]) == {"000"}
    assert "CAV" in df_000.columns
    assert not any(col.startswith("FAS_") for col in df_000.columns)
    row = df_000[df_000["record_id"] == record_ids[0]].iloc[0]
    assert (row["evid"], row["sta"], row["chan"], row["loc"]) == (
        "2020p000001",
        "ABC",
        "HN",
        "20",
    )

    # RotD components drop the IMs that are not calculated for them
    df_rotd50 = read_component(tmp_path, "rotd50")
    assert "PGA" in df_rotd50.columns
    assert "CAV" not in df_rotd50.columns

    # FAS components only have FAS columns, including eas
    df_eas = read_component(tmp_path, "eas_fas")
    assert set(df_eas["component"]) == {"eas"}
    assert not {"PGA", "CAV"} & set(df_eas.columns)
    assert any(col.startswith("FAS_") for col in df_eas.columns)

    # Orientations are their own components with the RotD IM columns, holding the angles
    for component in ims.ORIENTATION_COMPONENTS:
        df_orientation = read_component(tmp_path, component)
        assert list(df_orientation.columns) == list(df_rotd50.columns)
        assert set(df_orientation["component"]) == {component}
        assert (df_orientation["PGA"] == 45.0).all()
    assert (read_component(tmp_path, "rotd0")["PGA"] == 1.0).all()
    assert not any(col.endswith("_orientation") for col in df_rotd50.columns)


def test_no_fas_tables_without_fas(tmp_path: Path, fake_calculate_ims: list[float]):
    """No FAS directories are made when FAS is not calculated."""
    write_record(tmp_path, "2020p000001_ABC_HN_20")

    ims.compute_ims_for_all_processed_records(
        tmp_path, intensity_measures=[IM.PGA, IM.pSA]
    )

    im_merge_dir = file_structure.get_im_merge_dir(tmp_path)
    assert not list(im_merge_dir.glob("im_merge_*_fas"))
    assert (im_merge_dir / "im_merge_rotd50").is_dir()


def test_fas_columns_match_across_records(
    tmp_path: Path, fake_calculate_ims: list[float]
):
    """Records and batches with a lower Nyquist have every FAS column, with NaN above Nyquist."""
    write_record(tmp_path, "2020p000001_ABC_HN_20", dt=0.01)
    write_record(tmp_path, "2020p000002_DEF_HN_20", dt=0.02)

    ims.compute_ims_for_all_processed_records(
        tmp_path, intensity_measures=INTENSITY_MEASURES, batch_size=1
    )

    batch_files = sorted(
        file_structure.get_im_component_dir(tmp_path, "000_fas").glob("*.parquet")
    )
    assert len(batch_files) == 2
    columns = [list(pd.read_parquet(f).columns) for f in batch_files]
    assert columns[0] == columns[1]

    # 30.2 Hz is below Nyquist for dt = 0.01 (50 Hz) and above it for dt = 0.02 (25 Hz)
    df = read_component(tmp_path, "000_fas").set_index("record_id")
    assert df.loc["2020p000001_ABC_HN_20", "FAS_30.1995"] == 1.0
    assert np.isnan(df.loc["2020p000002_DEF_HN_20", "FAS_30.1995"])
    assert df.loc["2020p000002_DEF_HN_20", "FAS_1"] == 1.0


def test_skipped_records(tmp_path: Path, fake_calculate_ims: list[float]):
    """Missing component files and NaN values are skipped with a reason."""
    write_record(tmp_path, "2020p000001_ABC_HN_20")
    write_record(tmp_path, "2020p000002_DEF_HN_20")
    write_record(tmp_path, "2020p000003_GHI_HN_20", nan_middle=True)
    (
        file_structure.get_processed_dir(tmp_path, 2020, "2020p000002")
        / "2020p000002_DEF_HN_20.090"
    ).unlink()

    ims.compute_ims_for_all_processed_records(
        tmp_path, intensity_measures=INTENSITY_MEASURES
    )

    skipped = read_skipped(tmp_path).set_index("record_id")["reason"].to_dict()
    assert skipped == {
        "2020p000002_DEF_HN_20": "Failed to find the waveform ascii files",
        "2020p000003_GHI_HN_20": "Components contain NaN values",
    }
    assert read_component(tmp_path, "000")["record_id"].tolist() == [
        "2020p000001_ABC_HN_20"
    ]


def test_checkpoint_resumes_incomplete_batches(
    tmp_path: Path, fake_calculate_ims: list[float]
):
    """A re-run only calculates the batches that did not complete."""
    record_ids = [f"2020p00000{i}_ABC_HN_20" for i in range(1, 4)]
    for record_id in record_ids:
        write_record(tmp_path, record_id)

    ims.compute_ims_for_all_processed_records(
        tmp_path, intensity_measures=INTENSITY_MEASURES, batch_size=1
    )
    assert len(fake_calculate_ims) == 3

    # Everything is complete, so nothing is calculated again
    ims.compute_ims_for_all_processed_records(
        tmp_path, intensity_measures=INTENSITY_MEASURES, batch_size=1
    )
    assert len(fake_calculate_ims) == 3

    # Only the batch without a skipped records file is calculated again
    (ims.get_skipped_records_dir(tmp_path) / "batch_1.csv").unlink()
    ims.compute_ims_for_all_processed_records(
        tmp_path, intensity_measures=INTENSITY_MEASURES, batch_size=1
    )
    assert len(fake_calculate_ims) == 4
    assert sorted(read_component(tmp_path, "rotd50")["record_id"]) == record_ids


def test_changed_batches_are_recalculated(
    tmp_path: Path, fake_calculate_ims: list[float]
):
    """A different batch size removes the earlier batches instead of resuming or mixing them in."""
    record_ids = [f"2020p00000{i}_ABC_HN_20" for i in range(1, 4)]
    for record_id in record_ids:
        write_record(tmp_path, record_id)

    ims.compute_ims_for_all_processed_records(
        tmp_path, intensity_measures=INTENSITY_MEASURES, batch_size=1
    )
    ims.compute_ims_for_all_processed_records(
        tmp_path, intensity_measures=INTENSITY_MEASURES, batch_size=2
    )
    assert len(fake_calculate_ims) == 6

    component_dir = file_structure.get_im_component_dir(tmp_path, "rotd50")
    assert sorted(p.name for p in component_dir.iterdir()) == [
        "batch_0.parquet",
        "batch_1.parquet",
    ]
    assert sorted(read_component(tmp_path, "rotd50")["record_id"]) == record_ids


def test_no_records(tmp_path: Path, fake_calculate_ims: list[float]):
    """A run with no records writes an empty skipped records file without failing."""
    file_structure.get_waveform_dir(tmp_path).mkdir(parents=True)

    ims.compute_ims_for_all_processed_records(
        tmp_path, intensity_measures=INTENSITY_MEASURES
    )

    assert read_skipped(tmp_path).empty
    assert len(fake_calculate_ims) == 0
