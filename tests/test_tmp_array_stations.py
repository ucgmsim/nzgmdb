import pytest
from obspy import UTCDateTime

from nzgmdb.tmp_array import stations

START = UTCDateTime("2026-01-01")
END = UTCDateTime("2027-01-01")
# (min_lat, max_lat, min_lon, max_lon), wrapping across 180 to include the Chathams
BBOX = (-49.0, -32.0, 165.0, -176.9)

# Station code, lat, lon
CHRISTCHURCH = ("CHCH", -43.53, 172.63)
TASMAN_SEA = ("TASM", -40.0, 170.0)
CHATHAM = ("CHAT", -43.912, -176.507)


def _fake_query_provider(
    provider: str,
    start: UTCDateTime,
    end: UTCDateTime,
    bbox: tuple[float, float, float, float],
) -> list[list]:
    return [
        [provider, "XX", sta, lat, lon, 10.0, "HH", "", START, None, "False"]
        for sta, lat, lon in (CHRISTCHURCH, TASMAN_SEA, CHATHAM)
    ]


@pytest.mark.parametrize(
    "coastline_filter, expected",
    [(True, ["CHCH"]), (False, ["CHAT", "CHCH", "TASM"])],
)
def test_get_station_table_coastline_filter(
    monkeypatch: pytest.MonkeyPatch, coastline_filter: bool, expected: list[str]
):
    monkeypatch.setattr(stations, "query_provider", _fake_query_provider)
    station_df = stations.get_station_table(
        START,
        END,
        ["IRIS"],
        ["HH"],
        [],
        BBOX,
        coastline_filter=coastline_filter,
    )
    assert sorted(station_df["sta"]) == expected
    assert (station_df["end_date"] == "").all()
