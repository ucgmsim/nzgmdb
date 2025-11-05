import tempfile
from pathlib import Path

import pandas as pd

from nzgmdb.data_retrieval.waveform_extraction import extract_waveforms
from nzgmdb.management import file_structure


def test_extract_waveforms_outputs():
    with tempfile.TemporaryDirectory() as tmpdirname:
        main_dir = Path(tmpdirname) / "output"
        main_dir.mkdir()
        test_dir = Path(__file__).parent
        station_extraction_table = test_dir / "waveform_extraction_table_tests.csv"

        extract_waveforms(
            main_dir=main_dir,
            station_extraction_table_ffp=station_extraction_table,
        )

        # Check for the existence of expected output files
        flatfile_dir = file_structure.get_flatfile_dir(main_dir)
        expected_files = [
            flatfile_dir
            / file_structure.PreFlatfileNames.STATION_MAGNITUDE_TABLE_EXTRACTION,
            flatfile_dir
            / file_structure.SkippedRecordFilenames.EXTRACTION_SKIPPED_RECORDS,
            flatfile_dir / file_structure.SkippedRecordFilenames.CLIPPED_RECORDS,
            flatfile_dir
            / file_structure.SkippedRecordFilenames.MULTI_TRACE_ISSUE_RECORDS,
        ]
        for file in expected_files:
            assert file.exists(), f"Expected file {file} does not exist."

        # Check that these record_ids and reasons were skipped in the extraction skipped records file
        skipped_records_df = pd.read_csv(
            flatfile_dir
            / file_structure.SkippedRecordFilenames.EXTRACTION_SKIPPED_RECORDS
        )
        expected_skipped = {
            "3713218_ASHS_HN_20": "No data from ptime_est to ds595 + 1std",
            "3713218_PPHS_HN_20": "Could not agree on best traces using Arias Intensity selection",
            "2016p863743_VUWB_BN_26": "Offset in traces missing data between noise and ds595 + 1std",
            "2020p053330_TRAB_HNX_28": "All 0's",
            "2018p681807_LHBS_HN_20": "Start time after end time when trimming to common length",
            "2015p150076_VUWB_BN_26": "Could not agree on best traces using Arias Intensity selection",
            "2015p150076_VUWB_BN_2A": "Could not agree on best traces using Arias Intensity selection",
        }
        for record_id, expected_reason in expected_skipped.items():
            rows = skipped_records_df[skipped_records_df["record_id"] == record_id]
            assert (
                not rows.empty
            ), f"Expected skipped record_id {record_id} not found in skipped records."
            if not any(rows["reason"] == expected_reason):
                actual_reasons = rows["reason"].tolist()
                raise AssertionError(
                    f"Expected reason '{expected_reason}' for {record_id}, got {actual_reasons}"
                )

        # Check that the multi-trace issue records file has certain record_ids and reasons
        multi_trace_issues_df = pd.read_csv(
            flatfile_dir
            / file_structure.SkippedRecordFilenames.MULTI_TRACE_ISSUE_RECORDS
        )
        expected_multi_trace = {
            "2016p863743_VUWB_BN_2C": "Reduced to 3 traces using Arias Intensity selection",
            "2016p863743_VUWB_BN_23": "Selected trace: 2",
            "2020p053330_TRAB_HN_23": "Small overlapping data between traces",
            "2020p053330_TRAB_HN_27": "Kept longest trace for each channel due to overlapping duplicate data",
            "2013p543832_PRKS_HN_20": "Found change in timestep during main_intensity section of waveform",
            "2017p795065_WEL_BN_20": "Multiple horizontal channels, selected pair: 1, 2",
            "3469034_CRLZ_HN_20": "Large overlapping data between traces",
        }
        for record_id, expected_reason in expected_multi_trace.items():
            rows = multi_trace_issues_df[
                multi_trace_issues_df["record_id"] == record_id
            ]
            assert (
                not rows.empty
            ), f"Expected multi-trace issue record_id {record_id} not found in multi-trace issues."
            if not any(rows["reason"] == expected_reason):
                actual_reasons = rows["reason"].tolist()
                raise AssertionError(
                    f"Expected reason '{expected_reason}' for {record_id}, got {actual_reasons}"
                )
