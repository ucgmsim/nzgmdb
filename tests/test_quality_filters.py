import pandas as pd
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
    return pd.read_csv(Path(__file__).parent / "quality_db_testing.csv")


def test_filter_has_score_mean(sample_catalogue):
    cat, skipped = filter_has_score_mean(sample_catalogue)
    assert "1493377_NELS_HN_20" not in cat["record_id"].values
    assert "1493377_NELS_HN_20" in skipped["record_id"].values


def test_filter_score_mean(sample_catalogue):
    cat, skipped = filter_score_mean(sample_catalogue, score_min=0.5)
    assert "2016p858076_BMTS_HN_20" not in cat["record_id"].values
    assert "2016p858076_BMTS_HN_20" in skipped["record_id"].values


def test_filter_multi_mean(sample_catalogue):
    cat, skipped = filter_multi_mean(sample_catalogue, multi_max=0.2)
    assert "2126295_MLZ_HH_10" not in cat["record_id"].values
    assert "2126295_MLZ_HH_10" in skipped["record_id"].values


def test_filter_fmax(sample_catalogue):
    cat, skipped = filter_fmax(sample_catalogue, fmax_min=4.1)
    assert "2016p858116_QRZ_HH_10" not in cat["record_id"].values
    assert "2016p858116_QRZ_HH_10" in skipped["record_id"].values


def test_filter_fmin(sample_catalogue):
    cat, skipped = filter_fmin(sample_catalogue, fmin_max=2.0)
    assert not all(
        rec in cat["record_id"].values
        for rec in ["2016p858076_BMTS_HN_20", "2014p001738_POKS_HN_20"]
    )
    assert all(
        rec in skipped["record_id"].values
        for rec in ["2016p858076_BMTS_HN_20", "2014p001738_POKS_HN_20"]
    )


def test_filter_missing_sta_info(sample_catalogue):
    cat, skipped = filter_missing_sta_info(sample_catalogue)
    assert "3151675_RIZ_HH_10" not in cat["record_id"].values
    assert "3151675_RIZ_HH_10" in skipped["record_id"].values


def test_filter_ground_level_locations(sample_catalogue):
    cat, skipped = filter_ground_level_locations(sample_catalogue)
    assert "2013p707091_CPLB_BN_2D" not in cat["record_id"].values
    assert "2013p707091_CPLB_BN_2D" in skipped["record_id"].values


def test_filter_duplicate_channels(sample_catalogue):
    cat, skipped = filter_duplicate_channels(sample_catalogue)
    assert "2016p858848_POTS_BN_20" not in cat["record_id"].values
    assert "2016p858848_POTS_BN_20" in skipped["record_id"].values


def test_apply_clipNet_filter(sample_catalogue):
    clipped_file = Path(__file__).parent / "clipped_testing.csv"
    cat, skipped = apply_clipNet_filter(sample_catalogue.copy(), clipped_file)
    assert "1493377_NELS_HN_20" not in cat["record_id"].values
    assert "1493377_NELS_HN_20" in skipped["record_id"].values


def test_filter_troublesome_sensitivity(sample_catalogue):
    cat, skipped = filter_troublesome_sensitivity(sample_catalogue.copy())
    assert not all(
        rec in cat["record_id"].values
        for rec in ["1470798_TOZ_HH_10", "1632127_KNZ_HH_10"]
    )
    assert all(
        rec in skipped["record_id"].values
        for rec in ["1470798_TOZ_HH_10", "1632127_KNZ_HH_10"]
    )


def test_filter_empirical_predictions(sample_catalogue):
    cat, skipped = filter_empirical_predictions(
        sample_catalogue.copy(), max_residual_threshold=3.0
    )
    assert "2014p001738_POKS_HN_20" not in cat["record_id"].values
    assert "2014p001738_POKS_HN_20" in skipped["record_id"].values
