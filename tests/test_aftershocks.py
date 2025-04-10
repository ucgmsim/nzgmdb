import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from pyproj import Geod
from shapely.geometry import MultiPoint, Point, Polygon

from nzgmdb.calculation import aftershocks


def square_from_center(lat: float, lon: float, size: float = 2.0):
    """
    Create a square polygon centered at a (lat, lon) point.

    Parameters
    ----------
    lat : float
        Latitude of the center point.
    lon : float
        Longitude of the center point.
    size : float
        Size of the square in km. Default is 2.0 km.

    Returns
    -------
    Polygon
        A shapely Polygon object representing the square.
    """
    # Half-dimensions
    half_size = size / 2.0

    # Move from center to get corners using forward geodesic calculation
    # Order: NW, NE, SE, SW
    # Latitude shifts: north/south (+/- half_size)
    # Longitude shifts: east/west (+/- half_size)

    # Create a geodetic calculator
    geod = Geod(ellps="WGS84")

    # Top-left
    lon1, lat1, _ = geod.fwd(lon, lat, 315, (2**0.5) * half_size * 1000)
    # Top-right
    lon2, lat2, _ = geod.fwd(lon, lat, 45, (2**0.5) * half_size * 1000)
    # Bottom-right
    lon3, lat3, _ = geod.fwd(lon, lat, 135, (2**0.5) * half_size * 1000)
    # Bottom-left
    lon4, lat4, _ = geod.fwd(lon, lat, 225, (2**0.5) * half_size * 1000)

    return Polygon([(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4)])


def test_abwd_crjb_with_mock_data():
    # Example mini catalog
    data = {
        "datetime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-10"]),
        "mag": [6.6, 4.5, 4.2],
    }
    catalogue_pd = pd.DataFrame(data)

    # Use the rectangle generator from earlier
    centers = [
        (34.0, -118.0),  # close to...
        (34.001, -118.001),  # <~0.15 km apart
        (34.2, -118.2),  # >20 km away
    ]
    rupture_area_poly = [square_from_center(lat, lon, 2.0) for lat, lon in centers]

    # Run clustering
    flagvector, cluster_labels = aftershocks.abwd_crjb(
        catalogue_pd,
        rupture_area_poly,
        crjb_cutoff=10.0,
    )

    # Assert types and shape
    assert isinstance(flagvector, np.ndarray)
    assert isinstance(cluster_labels, np.ndarray)
    assert len(flagvector) == len(catalogue_pd)
    assert len(cluster_labels) == len(catalogue_pd)

    # Check the correct cluster and flag values
    npt.assert_array_equal(cluster_labels, [1, 1, 0])
    npt.assert_array_equal(flagvector, [0, 1, 0])


@pytest.mark.parametrize(
    "centroid, expected_dist, tolerance",
    [
        ([[0.0, 0.0]], 0.0, 0.01),  # Inside polygon → distance = 0
        ([[2.0, 2.0]], 157.0, 1.0),  # Outside, ~157 km away
    ],
)
def test_calculate_crjb(centroid: np.ndarray, expected_dist: float, tolerance: float):
    poly = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    boundary = MultiPoint([Point(x, y) for x, y in poly.exterior.coords])
    centroid_array = np.array(centroid)

    dists = aftershocks.calculate_crjb(poly, boundary, centroid_array)

    assert np.isclose(
        dists, expected_dist, atol=tolerance
    ), f"Expected {expected_dist}, got {dists}"


def test_resample_polygon_1km():
    # Create a square polygon 1 degree per side
    poly = Polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])

    # 1 degree of arc is ~111.2 km, so perimeter ≈ 4 x 111.2 km = ~444.8 km
    # Expect 444 resampled points (plus 8)
    expected_points = int(poly.length * 111.2) + 8

    resampled = aftershocks.resample_polygon_1km([poly])

    assert len(resampled) == 1
    assert isinstance(resampled[0], MultiPoint)
    assert len(resampled[0].geoms) == expected_points


@pytest.mark.parametrize(
    "date_str, expected_decimal",
    [
        ("2020-01-01 00:00:00", 2020.0),
        ("2020-07-02 12:00:00", 2020.5),
        ("2020-12-31 23:59:59", 2020.999999),
        ("2021-03-01 00:00:00", 2021.163),  # approx March 1
    ],
)
def test_decimal_year(date_str: str, expected_decimal: float):
    # Create a one-row DataFrame
    df = pd.DataFrame({"datetime": [pd.to_datetime(date_str)]})

    # Run function
    result = aftershocks.decimal_year(df)[0]  # Only one row

    # Assert approximate equality
    assert np.isclose(
        result, expected_decimal, atol=1e-3
    ), f"For {date_str}, expected {expected_decimal}, got {result}"
