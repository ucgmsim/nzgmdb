import pandas as pd
import numpy as np
import pytest

from pathlib import Path
from nzgmdb.data_processing.quality_db import (
    filter_has_score_mean,
    filter_score_mean,
    filter_multi_mean,
    filter_fmax,
    filter_fmin,
    filter_missing_sta_info,
    filter_ground_level_locations,
    apply_clipNet_filter,
    filter_troublesome_sensitivity,
    filter_empirical_predictions,
    filter_duplicate_channels,
)

@pytest.fixture
def sample_catalogue():
    return pd.DataFrame({
        "record_id": ["rec1", "rec2", "rec3", "rec4"],
        "score_X": [np.nan, 0.6, 0.2, 0.9],
        "score_Y": [np.nan, 0.6, 0.3, 0.9],
        "score_Z": [np.nan, 0.6, 0.4, 0.9],
        "multi_X": [0.2, 0.6, 0.7, 0.5],
        "multi_Y": [0.2, 0.6, 0.8, 0.5],
        "multi_Z": [0.2, 0.6, 0.9, 0.5],
        "fmax_X": [5.0, 10.0, 1.0, 6.0],
        "fmax_Y": [5.0, 10.0, 1.0, 6.0],
        "fmin_X": [0.1, 0.5, 2.0, 0.3],
        "fmin_Y": [0.1, 0.5, 2.1, 0.3],
        "Vs30": [300, np.nan, 450, 500],
        "is_ground_level": [True, False, True, True],
        "evid": ["e1", "e2", "e3", "e4"],
        "sta": ["s1", "s2", "s3", "s4"],
        "chan": ["HNZ", "BNZ", "HHZ", "HNZ"],
        "loc": ["00", "01", "00", "00"],
        "datetime": pd.to_datetime(["2021-01-01"] * 4),
        "tect_class": ["Interface", "Slab", "Active", "Slab"],
        "r_rup": [10, 15, 20, 25],
        "mag": [5.5, 6.0, 5.8, 6.5],
        "Z1.0": [300, 350, 400, 450],
        "pSA_0.01": [0.1, 0.2, 0.05, 0.15],
        "pSA_0.1": [0.2, 0.3, 0.1, 0.25],
        "pSA_1.0": [0.3, 0.5, 0.2, 0.35],
    })


def test_filter_has_score_mean(sample_catalogue):
    cat, skipped = filter_has_score_mean(sample_catalogue)
    assert "rec1" not in cat["record_id"].values
    assert "rec1" in skipped["record_id"].values


def test_filter_score_mean(sample_catalogue):
    cat, skipped = filter_score_mean(sample_catalogue, score_min=0.5)
    assert "rec3" not in cat["record_id"].values
    assert "rec3" in skipped["record_id"].values


def test_filter_multi_mean(sample_catalogue):
    cat, skipped = filter_multi_mean(sample_catalogue, multi_max=0.6, include_z=True)
    assert "rec3" not in cat["record_id"].values
    assert "rec3" in skipped["record_id"].values


def test_filter_fmax(sample_catalogue):
    cat, skipped = filter_fmax(sample_catalogue, fmax_min=4.0)
    assert "rec3" not in cat["record_id"].values
    assert "rec3" in skipped["record_id"].values


def test_filter_fmin(sample_catalogue):
    cat, skipped = filter_fmin(sample_catalogue, fmin_max=1.0)
    assert "rec3" not in cat["record_id"].values
    assert "rec3" in skipped["record_id"].values


def test_filter_missing_sta_info(sample_catalogue):
    cat, skipped = filter_missing_sta_info(sample_catalogue)
    assert "rec2" not in cat["record_id"].values
    assert "rec2" in skipped["record_id"].values


def test_filter_ground_level_locations(sample_catalogue):
    cat, skipped = filter_ground_level_locations(sample_catalogue)
    assert "rec2" not in cat["record_id"].values
    assert "rec2" in skipped["record_id"].values


def test_filter_duplicate_channels(sample_catalogue):
    bypass = np.array(["rec3"])
    cat, skipped = filter_duplicate_channels(sample_catalogue, bypass_records=bypass)
    assert "rec3" in cat["record_id"].values
    assert all(r in ["rec1", "rec2", "rec4", "rec3"] for r in cat["record_id"])


def test_apply_clipNet_filter(tmp_path, sample_catalogue):
    # Simulate clipped file
    clipped_file = tmp_path / "clipped.csv"
    pd.DataFrame({"record_id": ["rec3"]}).to_csv(clipped_file, index=False)

    cat, skipped = apply_clipNet_filter(sample_catalogue.copy(), clipped_file)
    assert "rec3" not in cat["record_id"].values
    assert "rec3" in skipped["record_id"].values


def test_filter_troublesome_sensitivity(monkeypatch, sample_catalogue, tmp_path):
    # Mock NZGMDB_DATA.fetch
    dummy_sens_file = tmp_path / "sensitivity_ignore.csv"
    pd.DataFrame({
        "sta": ["s3"],
        "chan": ["HHZ"],
        "loc": ["00"],
        "start_date": ["2020-01-01"],
        "end_date": ["2022-01-01"],
    }).to_csv(dummy_sens_file, index=False)

    from nzgmdb.management.data_registry import NZGMDB_DATA
    monkeypatch.setattr(NZGMDB_DATA, "fetch", lambda name: dummy_sens_file)

    cat, skipped = filter_troublesome_sensitivity(sample_catalogue.copy())
    assert "rec3" not in cat["record_id"].values
    assert "rec3" in skipped["record_id"].values
