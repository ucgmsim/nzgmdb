from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import pandas as pd
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from nzgmdb.data_retrieval import sites


class _DummyFionaCollection:
    """
    Minimal context manager that mimics a Fiona Collection.

    Parameters
    ----------
    shapes : list
        Iterable of shape-like records to yield when iterated.

    Returns
    -------
    _DummyFionaCollection
        Context manager instance that can be iterated over.
    """

    def __init__(self, shapes: list[Any]) -> None:  # noqa: D107
        self._shapes = shapes

    def __enter__(self) -> Self:  # noqa: D105
        return self

    def __exit__(  # noqa: D105
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False

    def __iter__(self) -> Iterator[Any]:  # noqa: D105
        return iter(self._shapes)


def _make_test_geotiff(
    width: int = 10,
    height: int = 10,
    nodata: float = -9999.0,
) -> MemoryFile:
    """
    Create an in-memory single-band GeoTIFF for testing.

    Parameters
    ----------
    width : int, default=10
        Raster width in pixels.
    height : int, default=10
        Raster height in pixels.
    nodata : float, default=-9999.0
        NoData value written to the dataset (and to one pixel in the raster).

    Returns
    -------
    memfile : rasterio.io.MemoryFile
        In-memory GeoTIFF containing a simple gradient field with a NoData pixel.
    """
    data = np.arange(width * height, dtype=np.float32).reshape(height, width)
    transform = from_origin(0.0, 10.0, 1.0, 1.0)
    data[0, 0] = nodata

    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as ds:
        ds.write(data, 1)

    return memfile


def test_sample_points_from_geotiff_inside_outside_and_nodata() -> None:
    """
    Test that GeoTIFF sampling returns finite values for in-bounds points, and
    returns NaN for out-of-bounds or NoData pixels.

    Returns
    -------
    None
    """
    memfile = _make_test_geotiff()
    with memfile.open() as ds:
        points = np.array(
            [
                [9.5, 0.5],
                [5.5, 5.5],
                [20.0, 5.0],
                [-5.0, 5.0],
                [5.0, 20.0],
            ],
            dtype=float,
        )
        out = sites.sample_points_from_geotiff(ds.name, points).ravel()

    assert out.shape == (len(points),)
    assert np.isnan(out[0])
    assert np.isfinite(out[1])
    assert np.isnan(out[2])
    assert np.isnan(out[3])
    assert np.isnan(out[4])


def test_fill_gaps_with_nearest_fills_nans_and_preserves_finite() -> None:
    """
    Test that `fill_gaps_with_nearest` fills NaN entries while preserving
    existing finite values.

    Returns
    -------
    None
    """
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    values = np.array([10.0, 20.0, np.nan, 40.0], dtype=float)

    filled = sites.fill_gaps_with_nearest(coords, values, k=3)

    assert filled.shape == values.shape
    assert np.isfinite(filled[2])
    assert filled[0] == 10.0
    assert filled[1] == 20.0
    assert filled[3] == 40.0


@pytest.mark.parametrize(
    "points",
    [
        np.array(
            [
                [-36.8485, 174.7633],
                [-41.2865, 174.7762],
                [-43.5321, 172.6362],
                [-45.0312, 168.6626],
            ],
            dtype=float,
        ),
        np.array(
            [[-34.0, 172.5], [-47.5, 166.0], [-41.0, 179.9], [-41.0, 166.5]],
            dtype=float,
        ),
        np.array(
            [[-30.0, 174.0], [-50.0, 170.0], [-41.0, 160.0], [-41.0, -175.0]],
            dtype=float,
        ),
        np.array(
            [
                [48.8566, 2.3522],
                [51.5074, -0.1278],
                [52.5200, 13.4050],
                [41.9028, 12.4964],
            ],
            dtype=float,
        ),
        np.vstack(
            [
                np.array([[-36.8485, 174.7633], [-41.2865, 174.7762]], dtype=float),
                np.array([[48.8566, 2.3522], [51.5074, -0.1278]], dtype=float),
                np.array([[-43.5321, 172.6362], [-45.0312, 168.6626]], dtype=float),
            ]
        ),
    ],
)
def test_site_updates_vs30_and_z1_fields_only_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    points: np.ndarray,
) -> None:
    """
    Integration-style test of site table population logic using monkeypatches.

    Verifies expected output columns exist and that Vs30 and Z1.0 / Z2.5 values
    (and their reference/quality fields) are only set when data are available.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch external dependencies.
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory for test artifacts.
    points : numpy.ndarray, shape (N, 2)
        Input points as [lat, lon] used to construct a minimal metadata table.

    Returns
    -------
    None
    """

    class _Cfg:
        def get_value(self, key: str):
            if key == "channel_codes":
                return "HHZ"
            if key == "bbox":
                return [0.0, -90.0, 360.0, 90.0]
            if key == "nzcvm_version":
                return "test"
            return None

    monkeypatch.setattr(sites.cfg, "Config", _Cfg)

    class _DummyInv(list):
        pass

    class _DummyClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_stations(self, *args, **kwargs):
            return _DummyInv()

    monkeypatch.setattr(sites, "FDSN_Client", _DummyClient)

    monkeypatch.setattr(
        sites.fiona,
        "open",
        lambda *_a, **_k: _DummyFionaCollection([]),
    )

    # Do NOT monkeypatch NZGMDB_DATA.abspath (read-only property).
    # Instead, patch fetch() to return a temp file path for the tif and anything else requested.
    combined_tif = tmp_path / "combined_mvn_wgs84.tif"
    combined_tif.touch()

    def _fetch(name: str, *args, **kwargs):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)
        return str(p)

    monkeypatch.setattr(sites.NZGMDB_DATA, "fetch", _fetch)

    geo = pd.DataFrame(
        {
            "Name": [f"S{i}" for i in range(len(points))],
            "Lat": points[:, 0],
            "Long": points[:, 1],
            "Elevation": np.zeros(len(points)),
            "Vs30_median": [np.nan] * len(points),
            "Sigmaln_Vs30": [np.nan] * len(points),
            "T_median": [np.nan] * len(points),
            "sigmaln_T": [np.nan] * len(points),
            "Q_T": [None] * len(points),
            "D_T": [None] * len(points),
            "T_Ref": [None] * len(points),
            "Z1.0_median": [np.nan] * len(points),
            "sigmaln_Z1.0": [np.nan] * len(points),
            "Z1.0_Ref": [None] * len(points),
            "Z2.5_median": [np.nan] * len(points),
            "sigmaln_Z2.5": [np.nan] * len(points),
            "Z2.5_Ref": [None] * len(points),
            "NZS1170SiteClass": [None] * len(points),
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_a, **_k: geo)

    def _find_domain_from_shapes(df: pd.DataFrame, _shapes: list) -> pd.DataFrame:
        out = df.copy()
        out["domain_no"] = 1
        return out

    monkeypatch.setattr(
        sites.tect_domain, "find_domain_from_shapes", _find_domain_from_shapes
    )

    def _compute_station_thresholds(
        stations: pd.DataFrame, model_version: str = None
    ) -> pd.DataFrame:
        n = len(stations)
        return pd.DataFrame(
            {
                "Z1.0(km)": np.full(n, 0.5),
                "Z2.5(km)": np.full(n, 2.0),
                "sigma": np.full(n, 0.25),
            },
            index=stations.index,
        )

    monkeypatch.setattr(
        sites.threshold, "compute_station_thresholds", _compute_station_thresholds
    )

    def _sample_points_from_geotiff(
        file_path: str,
        latlon_points: np.ndarray,
        band: int = 1,
    ) -> np.ndarray:
        n = len(latlon_points)
        vals = np.linspace(100.0, 500.0, n).astype(float)
        if n >= 2:
            vals[0] = np.nan
        return vals.reshape(-1, 1)

    monkeypatch.setattr(
        sites, "sample_points_from_geotiff", _sample_points_from_geotiff
    )

    def _fill_gaps_with_nearest(
        coords: np.ndarray,
        values: np.ndarray,
        invalid_mask: np.ndarray = None,
        k: int = 8,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float).copy()
        values[np.isnan(values)] = 250.0
        return values

    monkeypatch.setattr(sites, "fill_gaps_with_nearest", _fill_gaps_with_nearest)

    site_df = sites.create_site_table_response()

    for col in [
        "sta",
        "lat",
        "lon",
        "Vs30",
        "Q_Vs30",
        "Vs30_Ref",
        "Z1.0",
        "Z2.5",
        "Z1.0_ref",
        "Z2.5_ref",
        "Q_Z1.0",
        "Q_Z2.5",
    ]:
        assert col in site_df.columns

    assert site_df["Z1.0"].notna().any()
    assert site_df["Z2.5"].notna().any()
    assert (site_df["Z1.0_ref"].dropna() == "NZCVM (2026)").all()
    assert (site_df["Z2.5_ref"].dropna() == "NZCVM (2026)").all()

    vs30_non_nan = site_df["Vs30"].notna()
    assert (site_df.loc[vs30_non_nan, "Vs30_Ref"] == "Foster et al. (2019)").all()
    assert (site_df.loc[vs30_non_nan, "Q_Vs30"] == "Q3").all()
    assert site_df.loc[~vs30_non_nan, "Vs30_Ref"].isna().all()
    assert site_df.loc[~vs30_non_nan, "Q_Vs30"].isna().all()
