from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from obspy.clients.fdsn import Client as FDSN_Client

from nzgmdb.data_retrieval import geonet, inventory_xml
from nzgmdb.management.data_registry import NZGMDB_DATA


@pytest.mark.parametrize("evid", ["2016p858000"])
def test_fetch_event_data_against_benchmark(evid):
    """
    Test `geonet.fetch_event_data` for event `2016p858000` by comparing r_hyp
    values in the station_extraction_table to the benchmark propagation CSV.
    """

    tests_dir = Path(__file__).parent
    # Read event info (same small table used in other tests)
    event_info = pd.read_csv(tests_dir / "distance_evid_info.csv", dtype=str)
    event_row = event_info[event_info.evid == evid].iloc[0]

    client_NZ = FDSN_Client("GEONET")
    event_cat = client_NZ.get_events(eventid=evid)[0]

    inventory = inventory_xml.get_full_inventory(level="station")

    # site_table used by fetch_sta_extraction - empty (so code will fall back to default vs30)
    site_table = pd.DataFrame(columns=["net", "sta", "Vs30", "Z1.0"])

    # Load Mw_rrup data used to compute maxradius
    mw_rrup_data = np.loadtxt(NZGMDB_DATA.fetch("Mw_rrup.txt"))

    # Call the function under test
    event_id = evid
    event_data = geonet.fetch_event_data(
        event_id=event_id,
        event_cat=event_cat,
        inventory=inventory,
        site_table=site_table,
        mw_rrup_data=mw_rrup_data,
        only_sites=None,
        only_record_ids=None,
        n_procs=1,
    )

    # Unpack result
    event_line, station_extraction_table, skipped_records = event_data

    # Basic sanity checks
    assert event_line is not None, "fetch_event_data returned no event_line"
    assert isinstance(station_extraction_table, pd.DataFrame)
    assert not station_extraction_table.empty, "station_extraction_table is empty"
    assert skipped_records.empty, "skipped_records is not empty"

    # Load benchmark data
    benchmark_extraction_table = pd.read_csv(tests_dir / "geonet_extraction_table.csv")

    # Compare results
    pd.testing.assert_frame_equal(
        station_extraction_table,
        benchmark_extraction_table,
        atol=1e-3,
        check_dtype=False,
        check_index_type=False,
        check_names=False,
    )

    expected_time = datetime(2016, 11, 13, 11, 2, 56, 346094)
    expected_lat = -42.6925354
    expected_lon = 173.0221405
    expected_depth_km = 15.11445332
    expected_mag = 7.820379733
    expected_mag_unc = 0.4
    expected_ndef = 94
    expected_nsta = 93
    expected_std = 1.822964596

    assert event_line[0] == evid
    assert event_line[1] == expected_time
    np.testing.assert_allclose(event_line[2], expected_lat, atol=1e-8)
    np.testing.assert_allclose(event_line[3], expected_lon, atol=1e-8)
    np.testing.assert_allclose(event_line[4], expected_depth_km, atol=1e-8)
    np.testing.assert_allclose(event_line[7], expected_mag, atol=1e-9)
    np.testing.assert_allclose(event_line[10], expected_mag_unc, atol=1e-9)
    assert event_line[14] == expected_ndef
    assert event_line[15] == expected_nsta
    np.testing.assert_allclose(event_line[17], expected_std, atol=1e-9)
