from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from nzgmdb.data_retrieval import inventory_xml, sites


def _make_test_geotiff(
    width: int = 10,
    height: int = 10,
    nodata: float = -9999.0,
) -> MemoryFile:
    """
    Create an in-memory single-band GeoTIFF (gradient values, one NoData pixel).

    Parameters
    ----------
    width : int, default=10
        Raster width in pixels.
    height : int, default=10
        Raster height in pixels.
    nodata : float, default=-9999.0
        NoData value, also written to the top-left pixel.

    Returns
    -------
    MemoryFile
        In-memory GeoTIFF.
    """
    data = np.arange(width * height, dtype=np.float32).reshape(height, width)
    data[0, 0] = nodata
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(0.0, 10.0, 1.0, 1.0),
        nodata=nodata,
    ) as ds:
        ds.write(data, 1)
    return memfile


def _inventory_row(
    sta: str,
    lat: float,
    lon: float,
    net: str = "NZ",
    provider: str = "GEONET",
    chan: str = "HN",
    loc: str = "20",
) -> dict:
    """
    One station / channel row as returned by `inventory_xml.get_full_inventory(return_df=True)`.

    Parameters
    ----------
    sta : str
        Station code.
    lat : float
        Station latitude.
    lon : float
        Station longitude.
    net : str, optional
        Network code.
    provider : str, optional
        FDSN provider.
    chan : str, optional
        Two letter channel code.
    loc : str, optional
        Location code.

    Returns
    -------
    dict
        Inventory row.
    """
    return {
        "provider": provider,
        "net": net,
        "sta": sta,
        "lat": lat,
        "lon": lon,
        "elev": 10.0,
        "creation_date": "2000-01-01T00:00:00",
        "end_date": None,
        "chan": chan,
        "loc": loc,
        "loc_elev": 0.0,
        "start_time": "2000-01-01T00:00:00",
        "end_time": None,
    }


def _metadata_row(
    sta: str,
    vs30: float = 400.0,
    q_vs30: str = "Q1",
    z1: float = 200.0,
    q_z1: str = "Q1",
    z25: float = 1500.0,
) -> dict:
    """
    One GeoNet metadata summary row (Z1.0 / Z2.5 in m), measured Q1 values by default.

    Parameters
    ----------
    sta : str
        Station code.
    vs30 : float, optional
        Vs30 (m/s).
    q_vs30 : str, optional
        Vs30 quality.
    z1 : float, optional
        Z1.0 (m).
    q_z1 : str, optional
        Z1.0 / Z2.5 quality.
    z25 : float, optional
        Z2.5 (m).

    Returns
    -------
    dict
        Metadata row.
    """
    return {
        "Name": sta,
        "NZS1170SiteClass": "C",
        "Vs30_median": vs30,
        "Sigmaln_Vs30": 0.1,
        "Q_Vs30": q_vs30,
        "Vs30_Ref": "GeoNet",
        "T_median": 0.5,
        "sigmaln_T": 0.1,
        "Q_T": "Q1",
        "D_T": "I",
        "T_Ref": "GeoNet",
        "Z1.0_median": z1,
        "sigmaln_Z1.0": 0.2,
        "Q_Z1.0": q_z1,
        "Z1.0_Ref": "GeoNet",
        "Z2.5_median": z25,
        "sigmaln_Z2.5": 0.2,
        "Q_Z2.5": q_z1,
        "Z2.5_Ref": "GeoNet",
    }


@pytest.fixture
def run_site_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable:
    """
    Run `create_site_table_response` with external data sources replaced.

    The real tectonic domain code runs (with no domain shapes, so every site is
    Oceanic). NZCVM returns Z1.0 = |lat| / 50 km, Z2.5 = |lat| / 10 km and
    sigma = 0.3, in reversed station order to check results are aligned by station.
    The Vs30 map returns `vs30_map` values (default 250) in point order.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Callable
        `run(inventory, metadata, vs30_map=None, nzcvm_error=None)` returning
        `(site_df, station_df, calls)`.
    """

    def run(
        inventory: list[dict],
        metadata: list[dict],
        vs30_map: list[float] | None = None,
        nzcvm_error: Exception | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        calls = {"nzcvm_stations": [], "vs30_points": 0}

        metadata_ffp = tmp_path / "Geonet_Metadata_Summary_v1.4.csv"
        pd.DataFrame(metadata, columns=list(_metadata_row("X"))).to_csv(
            metadata_ffp, index=False
        )
        monkeypatch.setattr(
            sites.NZGMDB_DATA,
            "fetch",
            lambda name, *_a, **_k: str(
                metadata_ffp if name == metadata_ffp.name else tmp_path / name
            ),
        )
        monkeypatch.setattr(
            sites.inventory_xml,
            "get_full_inventory",
            lambda **_k: pd.DataFrame(
                inventory, columns=inventory_xml.STATION_INFO_COLUMNS
            ),
        )
        monkeypatch.setattr(sites.fiona, "open", lambda *_a, **_k: _EmptyCollection())

        def _thresholds(stations: pd.DataFrame, model_version: str) -> pd.DataFrame:
            if nzcvm_error is not None:
                raise nzcvm_error
            calls["nzcvm_stations"] = list(stations.index)
            stations = stations.iloc[::-1]
            return pd.DataFrame(
                {
                    "Z1.0(km)": stations["lat"].abs() / 50,
                    "Z2.5(km)": stations["lat"].abs() / 10,
                    "sigma": 0.3,
                },
                index=stations.index,
            )

        monkeypatch.setattr(sites.threshold, "compute_station_thresholds", _thresholds)

        def _vs30_map(_file_path: Path, points: np.ndarray) -> np.ndarray:
            calls["vs30_points"] = len(points)
            values = vs30_map if vs30_map is not None else [250.0] * len(points)
            return np.asarray(values, dtype=float).reshape(-1, 1)

        monkeypatch.setattr(sites, "sample_points_from_geotiff", _vs30_map)

        site_df, station_df = sites.create_site_table_response()
        return site_df.set_index("sta"), station_df, calls

    return run


class _EmptyCollection(list):
    """Fiona collection stand-in with no shapes."""

    def __enter__(self) -> "_EmptyCollection":  # noqa: D105
        return self

    def __exit__(self, *_args: object) -> bool:  # noqa: D105
        return False


def test_sample_points_from_geotiff_inside_outside_and_nodata() -> None:
    """In-bounds points get values; out-of-bounds and NoData pixels give NaN."""
    memfile = _make_test_geotiff()
    with memfile.open() as ds:
        points = np.array(
            [[9.5, 0.5], [5.5, 5.5], [20.0, 5.0], [-5.0, 5.0], [5.0, 20.0]]
        )
        out = sites.sample_points_from_geotiff(ds.name, points).ravel()

    assert out.shape == (len(points),)
    assert np.isnan(out[0])  # NoData pixel
    assert np.isfinite(out[1])
    assert np.isnan(out[2:]).all()  # outside the raster


def test_fill_gaps_with_nearest_fills_nans_and_preserves_finite() -> None:
    """NaNs are filled from neighbours and existing values are unchanged."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    values = np.array([10.0, 20.0, np.nan, 40.0])

    filled = sites.fill_gaps_with_nearest(coords, values, k=3)

    np.testing.assert_array_equal(filled[[0, 1, 3]], [10.0, 20.0, 40.0])
    assert filled[2] == pytest.approx(np.mean([10.0, 20.0, 40.0]))


def test_fill_gaps_with_nearest_all_nan_stays_nan() -> None:
    """With no valid values there is nothing to fill from."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert np.isnan(sites.fill_gaps_with_nearest(coords, [np.nan, np.nan])).all()


def test_one_site_row_per_station_and_all_channels_kept(run_site_table: Callable):
    """Channels collapse to one site row; station table keeps channels and maps '' to '00'."""
    site_df, station_df, calls = run_site_table(
        inventory=[
            _inventory_row("AAA", -41.0, 174.0, chan="HN", loc="20"),
            _inventory_row("AAA", -41.0, 174.0, chan="HH", loc="10"),
            _inventory_row("BBB", -42.0, 173.0, chan="HN", loc=""),
        ],
        metadata=[_metadata_row("AAA"), _metadata_row("BBB")],
    )

    assert sorted(site_df.index) == ["AAA", "BBB"]
    assert len(station_df) == 3
    assert station_df.loc[station_df["sta"] == "BBB", "loc"].tolist() == ["00"]
    assert (site_df["site_domain_no"] == 0).all()  # outside every domain -> Oceanic
    # All measured Q1 values: no model calls
    assert calls == {"nzcvm_stations": [], "vs30_points": 0}


def test_same_station_in_two_networks_is_one_site(run_site_table: Callable):
    """A station listed by two networks with identical channels is only kept once."""
    site_df, station_df, _ = run_site_table(
        inventory=[
            _inventory_row("SNZO", -41.3, 174.7, net="IU"),
            _inventory_row("SNZO", -41.3, 174.7, net="NZ"),
        ],
        metadata=[_metadata_row("SNZO")],
    )

    assert len(site_df) == 1
    assert site_df.loc["SNZO", "net"] == "IU"  # first network listed is kept
    assert len(station_df) == 1


def test_only_inventory_stations_are_included(run_site_table: Callable):
    """Metadata-only stations are dropped; inventory-only stations are filled from the models."""
    site_df, _, calls = run_site_table(
        inventory=[_inventory_row("CCC", -40.0, 175.0)],
        metadata=[_metadata_row("ZZZ")],
    )

    assert list(site_df.index) == ["CCC"]
    assert calls["nzcvm_stations"] == ["CCC"]
    ccc = site_df.loc["CCC"]
    assert ccc["Z1.0"] == pytest.approx(40.0 / 50 * 1000)  # m
    assert ccc["Z2.5"] == pytest.approx(40.0 / 10)  # km
    assert (ccc["Z1.0_ref"], ccc["Q_Z1.0"]) == ("NZCVM (2026)", "Q3")
    assert (ccc["Vs30"], ccc["Vs30_Ref"], ccc["Q_Vs30"]) == (
        250.0,
        "Vs30 Map v1.0 (2026)",
        "Q3",
    )


def test_q3_z_values_are_replaced_by_nzcvm(run_site_table: Callable):
    """
    Q3 GeoNet Z1.0 / Z2.5 are overwritten by NZCVM (not just labelled as NZCVM),
    Q1 values are untouched, and a measured Vs30 is kept even when Z1.0 is Q3.
    """
    site_df, _, calls = run_site_table(
        inventory=[
            _inventory_row("AAA", -41.0, 174.0),
            _inventory_row("BBB", -42.0, 173.0),
        ],
        metadata=[
            _metadata_row("AAA", z1=200.0, z25=1500.0, q_z1="Q1"),
            _metadata_row("BBB", z1=300.0, z25=1600.0, q_z1="Q3", vs30=500.0),
        ],
    )

    assert calls["nzcvm_stations"] == ["BBB"]

    aaa, bbb = site_df.loc["AAA"], site_df.loc["BBB"]
    # Q1 GeoNet values kept (Z2.5 converted m -> km)
    assert (aaa["Z1.0"], aaa["Z2.5"], aaa["Z1.0_ref"]) == (200.0, 1.5, "GeoNet")
    # Q3 GeoNet values replaced, value and label agree
    assert bbb["Z1.0"] == pytest.approx(42.0 / 50 * 1000)
    assert bbb["Z2.5"] == pytest.approx(42.0 / 10)
    assert (bbb["Z1.0_std"], bbb["Z2.5_std"]) == (0.3, 0.3)
    assert (bbb["Z1.0_ref"], bbb["Z2.5_ref"]) == ("NZCVM (2026)", "NZCVM (2026)")
    # Measured Vs30 kept even though Z1.0 was Q3
    assert (bbb["Vs30"], bbb["Q_Vs30"], bbb["Vs30_Ref"]) == (500.0, "Q1", "GeoNet")
    assert calls["vs30_points"] == 0


def test_q3_vs30_is_replaced_without_touching_z(run_site_table: Callable):
    """Vs30 follows its own quality: Q3 Vs30 is replaced while measured Z values stay."""
    site_df, _, calls = run_site_table(
        inventory=[_inventory_row("AAA", -41.0, 174.0)],
        metadata=[_metadata_row("AAA", vs30=600.0, q_vs30="Q3")],
    )

    aaa = site_df.loc["AAA"]
    assert (aaa["Vs30"], aaa["Vs30_Ref"]) == (250.0, "Vs30 Map v1.0 (2026)")
    assert (aaa["Z1.0"], aaa["Z1.0_ref"]) == (200.0, "GeoNet")
    assert calls["nzcvm_stations"] == []


def test_vs30_map_gaps_are_filled_from_neighbours(run_site_table: Callable):
    """A NaN from the Vs30 map is filled from nearby stations and rounded."""
    site_df, _, _ = run_site_table(
        inventory=[
            _inventory_row("AAA", -41.00, 174.00),
            _inventory_row("BBB", -41.01, 174.01),
            _inventory_row("CCC", -41.02, 174.02),
        ],
        metadata=[],
        vs30_map=[200.0, np.nan, 310.0],
    )

    assert site_df.loc["BBB", "Vs30"] == 255.0
    assert (site_df["Vs30_Ref"] == "Vs30 Map v1.0 (2026)").all()


def test_vs30_left_missing_when_map_has_no_values(run_site_table: Callable):
    """If the map has no values at all, Vs30 stays NaN and is not labelled."""
    site_df, _, _ = run_site_table(
        inventory=[_inventory_row("AAA", -41.0, 174.0)],
        metadata=[],
        vs30_map=[np.nan],
    )

    assert np.isnan(site_df.loc["AAA", "Vs30"])
    assert pd.isna(site_df.loc["AAA", "Vs30_Ref"])
    assert pd.isna(site_df.loc["AAA", "Q_Vs30"])


def test_nzcvm_failure_raises_user_warning(run_site_table: Callable):
    """A failing NZCVM lookup is reported, with the original error kept as the cause."""
    with pytest.raises(UserWarning) as excinfo:
        run_site_table(
            inventory=[_inventory_row("AAA", -41.0, 174.0)],
            metadata=[],
            nzcvm_error=ValueError("no NZCVM data"),
        )
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_no_stations_returns_empty_tables(run_site_table: Callable):
    """An empty inventory gives empty site / station tables instead of an error."""
    site_df, station_df, calls = run_site_table(inventory=[], metadata=[])

    assert site_df.empty and station_df.empty
    assert "Vs30" in site_df.columns and "loc" in station_df.columns
    assert calls == {"nzcvm_stations": [], "vs30_points": 0}


def test_add_site_basins_priority_and_boundaries(monkeypatch: pytest.MonkeyPatch):
    """
    Later priority basins override earlier ones, non-priority basins never override,
    boundary points count as inside and points outside every basin get None.
    """
    squares = {
        "a.geojson": [(0, 0), (2, 0), (2, 2), (0, 2)],
        "b.geojson": [(1, 1), (3, 1), (3, 3), (1, 3)],
        "c.geojson": [(0, 0), (3, 0), (3, 3), (0, 3)],
    }

    class _Registry:
        def __init__(self, *_args: object) -> None:  # noqa: D107
            self.registry = {
                "basin": [
                    {"name": "BasinA_v1", "boundaries": ["a.geojson"]},
                    {"name": "BasinB_v1", "boundaries": ["b.geojson"]},
                    {"name": "BasinC_v1", "boundaries": ["c.geojson"]},
                ]
            }
            self.global_params = {"basins": ["BasinA_v1", "BasinB_v1", "BasinC_v1"]}

        def load_basin_boundary(self, boundary: str) -> np.ndarray:  # noqa: D102
            return np.array(squares[boundary], dtype=float)

    config = {"nzcvm_version": "test", "priority_basins": ["BasinB"]}
    monkeypatch.setattr(sites.registry, "CVMRegistry", _Registry)
    monkeypatch.setattr(
        sites.cfg, "Config", lambda: SimpleNamespace(get_value=config.get)
    )

    site_df = pd.DataFrame(
        {
            "lon": [0.5, 1.5, 2.5, 3.0, 5.0],
            "lat": [0.5, 1.5, 2.5, 0.5, 5.0],
        }
    )
    out = sites.add_site_basins(site_df, Path("nzcvm_data"))

    assert out["basin"].tolist() == ["BasinA", "BasinB", "BasinB", "BasinC", None]
    assert (out["nzcvm_version"] == "test").all()
