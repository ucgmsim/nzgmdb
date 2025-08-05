from pathlib import Path

import pandas as pd
import pytest

from nzgmdb.data_processing import quality_db


@pytest.fixture
def sample_catalogue():
    return pd.read_csv(Path(__file__).parent / "quality_db_testing.csv")


def test_filter_has_score_mean(sample_catalogue: pd.DataFrame):
    """
    Test the filter_has_score_mean function.

    Parameters
    ----------
    sample_catalogue : pd.DataFrame
        A sample catalogue DataFrame to test the filtering function.
    """
    cat, skipped = quality_db.filter_has_score_mean(sample_catalogue)
    assert "1493377_NELS_HN_20" not in cat["record_id"].values
    assert "1493377_NELS_HN_20" in skipped["record_id"].values


def test_filter_score_mean(sample_catalogue: pd.DataFrame):
    """
    Test the filter_score_mean function.

    Parameters
    ----------
    sample_catalogue : pd.DataFrame
        A sample catalogue DataFrame to test the filtering function.
    """
    cat, skipped = quality_db.filter_score_mean(sample_catalogue, score_min=0.5)
    assert "2016p858076_BMTS_HN_20" not in cat["record_id"].values
    assert "2016p858076_BMTS_HN_20" in skipped["record_id"].values


def test_filter_multi_mean(sample_catalogue: pd.DataFrame):
    """
    Test the filter_multi_mean function.

    Parameters
    ----------
    sample_catalogue : pd.DataFrame
        A sample catalogue DataFrame to test the filtering function.
    """
    cat, skipped = quality_db.filter_multi_mean(sample_catalogue, multi_max=0.2)
    assert "2126295_MLZ_HH_10" not in cat["record_id"].values
    assert "2126295_MLZ_HH_10" in skipped["record_id"].values


def test_filter_fmax(sample_catalogue: pd.DataFrame):
    """
    Test the filter_fmax function.

    Parameters
    ----------
    sample_catalogue : pd.DataFrame
        A sample catalogue DataFrame to test the filtering function.
    """
    cat, skipped = quality_db.filter_fmax(sample_catalogue, fmax_min=4.1)
    assert "2016p858116_QRZ_HH_10" not in cat["record_id"].values
    assert "2016p858116_QRZ_HH_10" in skipped["record_id"].values


def test_filter_fmin(sample_catalogue: pd.DataFrame):
    """
    Test the filter_fmin function.

    Parameters
    ----------
    sample_catalogue : pd.DataFrame
        A sample catalogue DataFrame to test the filtering function.
    """
    cat, skipped = quality_db.filter_fmin(sample_catalogue, fmin_max=2.0)
    assert not all(
        rec in cat["record_id"].values
        for rec in ["2016p858076_BMTS_HN_20", "2014p001738_POKS_HN_20"]
    )
    assert all(
        rec in skipped["record_id"].values
        for rec in ["2016p858076_BMTS_HN_20", "2014p001738_POKS_HN_20"]
    )


def test_filter_missing_sta_info(sample_catalogue: pd.DataFrame):
    """
    Test the filter_missing_sta_info function.

    Parameters
    ----------
    sample_catalogue : pd.DataFrame
        A sample catalogue DataFrame to test the filtering function.
    """
    cat, skipped = quality_db.filter_missing_sta_info(sample_catalogue)
    assert "3151675_RIZ_HH_10" not in cat["record_id"].values
    assert "3151675_RIZ_HH_10" in skipped["record_id"].values


def test_filter_ground_level_locations(sample_catalogue: pd.DataFrame):
    """
    Test the filter_ground_level_locations function.

    Parameters
    ----------
    sample_catalogue : pd.DataFrame
        A sample catalogue DataFrame to test the filtering function.
    """
    cat, skipped = quality_db.filter_ground_level_locations(sample_catalogue)
    assert "2013p707091_CPLB_BN_2D" not in cat["record_id"].values
    assert "2013p707091_CPLB_BN_2D" in skipped["record_id"].values


def test_filter_duplicate_channels(sample_catalogue: pd.DataFrame):
    """
    Test the filter_duplicate_channels function.

    Parameters
    ----------
    sample_catalogue : pd.DataFrame
        A sample catalogue DataFrame to test the filtering function.
    """
    cat, skipped = quality_db.filter_duplicate_channels(sample_catalogue)
    assert "2016p858848_POTS_BN_20" not in cat["record_id"].values
    assert "2016p858848_POTS_BN_20" in skipped["record_id"].values


def test_apply_clipNet_filter(sample_catalogue: pd.DataFrame):
    """
    Test the apply_clipNet_filter function.

    Parameters
    ----------
    sample_catalogue : pd.DataFrame
        A sample catalogue DataFrame to test the filtering function.
    """
    clipped_file = Path(__file__).parent / "clipped_testing.csv"
    cat, skipped = quality_db.apply_clipNet_filter(
        sample_catalogue.copy(), clipped_file
    )
    assert "1493377_NELS_HN_20" not in cat["record_id"].values
    assert "1493377_NELS_HN_20" in skipped["record_id"].values


def test_filter_troublesome_sensitivity(sample_catalogue: pd.DataFrame):
    """
    Test the filter_troublesome_sensitivity function.

    Parameters
    ----------
    sample_catalogue : pd.DataFrame
        A sample catalogue DataFrame to test the filtering function.
    """
    cat, skipped = quality_db.filter_troublesome_sensitivity(sample_catalogue.copy())
    assert not all(
        rec in cat["record_id"].values
        for rec in ["1470798_TOZ_HH_10", "1632127_KNZ_HH_10"]
    )
    assert all(
        rec in skipped["record_id"].values
        for rec in ["1470798_TOZ_HH_10", "1632127_KNZ_HH_10"]
    )


def test_filter_empirical_predictions(sample_catalogue: pd.DataFrame):
    """
    Test the filter_empirical_predictions function.

    Parameters
    ----------
    sample_catalogue : pd.DataFrame
        A sample catalogue DataFrame to test the filtering function.
    """
    cat, skipped = quality_db.filter_empirical_predictions(
        sample_catalogue.copy(), max_residual_threshold=3.0
    )
    assert "2014p001738_POKS_HN_20" not in cat["record_id"].values
    assert "2014p001738_POKS_HN_20" in skipped["record_id"].values
