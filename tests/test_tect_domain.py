import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box, mapping

from nzgmdb.data_retrieval import tect_domain
from nzgmdb.management import config as cfg


def nztm_box(min_lon: float, min_lat: float, max_lon: float, max_lat: float):
    """
    Create a lon / lat box polygon in NZTM coordinates (matching the domain shapefiles).

    Parameters
    ----------
    min_lon : float
        Minimum longitude.
    min_lat : float
        Minimum latitude.
    max_lon : float
        Maximum longitude.
    max_lat : float
        Maximum latitude.

    Returns
    -------
    Polygon
        The box polygon in NZTM coordinates.
    """
    config = cfg.Config()
    wgs2nztm = Transformer.from_crs(
        config.get_value("ll_num"), config.get_value("nztm_num"), always_xy=True
    )
    x, y = wgs2nztm.transform(
        [min_lon, max_lon, max_lon, min_lon], [min_lat, min_lat, max_lat, max_lat]
    )
    return box(min(x), min(y), max(x), max(y))


def test_find_domain_from_shapes():
    """Polygon, MultiPolygon, overlap priority and Oceanic fallback."""
    single = nztm_box(172.0, -44.0, 173.0, -43.0)
    multi = nztm_box(175.0, -40.0, 176.0, -39.0).union(
        nztm_box(177.0, -39.0, 178.0, -38.0)
    )
    overlap = nztm_box(172.5, -43.5, 172.9, -43.1)
    shapes = [
        {
            "properties": {"Domain_No": "1a", "DomainType": "Typ1"},
            "geometry": mapping(single),
        },
        {
            "properties": {"Domain_No": 7, "DomainType": "Typ7"},
            "geometry": mapping(multi),
        },
        {
            "properties": {"Domain_No": "2", "DomainType": "Typ2"},
            "geometry": mapping(overlap),
        },
    ]
    assert mapping(multi)["type"] == "MultiPolygon"

    points = pd.DataFrame(
        {
            "lon": [172.2, 175.5, 177.5, 172.7, 168.0],
            "lat": [-43.8, -39.5, -38.5, -43.3, -46.0],
        }
    )
    result = tect_domain.find_domain_from_shapes(points, shapes)

    np.testing.assert_array_equal(
        result["domain_no"].to_numpy(dtype=object),
        np.array(["1a", 7, 7, "2", 0], dtype=object),
    )
    np.testing.assert_array_equal(
        result["domain_type"].to_numpy(dtype=object),
        ["Typ1", "Typ7", "Typ7", "Typ2", "Oceanic"],
    )
