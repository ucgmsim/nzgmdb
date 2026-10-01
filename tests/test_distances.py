import json
from pathlib import Path

import fiona
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod, Transformer
from shapely.geometry import Polygon

from cmt_solutions import cmt_data
from nzgmdb.calculation.distances import compute_distances_for_event
from nzgmdb.data_retrieval import tect_domain
from nzgmdb.management import config as cfg
from nzgmdb.management.data_registry import NZGMDB_DATA, REGISTRY

TEST_EVIDS = [
    "2016p858000",  # ff
    "2016p459109",  # cmt_unc
    "2016p427356",  # domain
    "2016p659242",  # cmt
]
geod = Geod(ellps="WGS84")


# ---------------------------
# Helpers
# ---------------------------
def generate_sites_around_event(lat, lon, evid, radii_km=(10, 100, 400), n_per_ring=4):
    stations = []
    sta_idx = 0

    for r in radii_km:
        bearings = np.linspace(0, 360, n_per_ring, endpoint=False)

        for b in bearings:
            lon2, lat2, _ = geod.fwd(lon, lat, b, r * 1000)  # meters

            stations.append(
                {
                    "sta": f"{evid}_sta_{sta_idx}",
                    "provider": "TEST",
                    "net": "NZ",
                    "lat": lat2,
                    "lon": lon2,
                    "elev": 0.0,
                }
            )
            sta_idx += 1

    return pd.DataFrame(stations)


def build_site_and_im_df(event_df):
    site_rows = []
    im_rows = []

    for _, event in event_df.iterrows():
        evid = event["evid"]
        sites = generate_sites_around_event(event["lat"], event["lon"], evid)

        sites["depth"] = sites["elev"] / -1000.0

        site_rows.append(sites)

        im_rows.extend([{"evid": evid, "sta": sta} for sta in sites["sta"]])

    site_df = pd.concat(site_rows, ignore_index=True)
    im_df = pd.DataFrame(im_rows)

    return site_df, im_df


# ---------------------------
# Main test
# ---------------------------
@pytest.mark.parametrize("evid", TEST_EVIDS)
def test_compute_distances_for_event(
    evid,
):
    """
    Test distances and geometry for multiple nodal plane determination paths.
    """
    benchmark_dir = Path(__file__).parent / "benchmark_distances"
    event_df_full = pd.read_csv(
        Path(__file__).parent / "distance_evid_info.csv",
    )

    # --- Load NZGMDB data ---
    # Get the CMT solutions data
    cmt_df = cmt_data.get_cmt_data()

    NZGMDB_DATA.fetch("TectonicDomains_Feb2021_8_NZTM.shp")
    NZGMDB_DATA.fetch("TectonicDomains_Feb2021_8_NZTM.dbf")
    NZGMDB_DATA.fetch("TectonicDomains_Feb2021_8_NZTM.shx")
    with fiona.open(
        Path(NZGMDB_DATA.abspath) / "TectonicDomains_Feb2021_8_NZTM.shp"
    ) as collection:
        shapes = list(collection)

    fallback_domain_values = tect_domain.find_domain_from_shapes(
        event_df_full.loc[:, ["lat", "lon"]],
        shapes,
    )

    # Add the fallback domain values to the event df
    event_df_full["domain_no_backup"] = fallback_domain_values.loc[
        :, "domain_no"
    ].values

    hik_objs = np.load(NZGMDB_DATA.fetch("hik_focmec.npy"), allow_pickle=True)[()]
    puy_objs = np.load(NZGMDB_DATA.fetch("puy_focmec.npy"), allow_pickle=True)[()]

    with open(NZGMDB_DATA.fetch("slab-faulting2.json"), "r", encoding="utf-8") as f:
        slab_faulting_geo = json.load(f)

    with open(NZGMDB_DATA.fetch("nzfocmecmod.json"), "r", encoding="utf-8") as f:
        nz_mech = json.load(f)

    # Get the focal domain
    domain_focal_df = pd.read_csv(
        NZGMDB_DATA.fetch("focal_mech_tectonic_domain_v1.csv"),
    )

    # Get the Taupo VZ polygon
    tect_domain_points = pd.read_csv(
        NZGMDB_DATA.fetch("tectonic_domain_polygon_points.csv"),
    )
    tvz_points = tect_domain_points[tect_domain_points.domain_no == 4][
        ["latitude", "longitude"]
    ]
    config = cfg.Config()
    ll_num = config.get_value("ll_num")
    nztm_num = config.get_value("nztm_num")
    wgs2nztm = Transformer.from_crs(ll_num, nztm_num)
    taupo_transform = np.dstack(
        np.array(wgs2nztm.transform(tvz_points.latitude, tvz_points.longitude))
    )[0]
    taupo_polygon = Polygon(taupo_transform)

    # Go through the registry keys and check if they are .srf files to use
    srf_files = {}
    for file in REGISTRY.keys():
        if file.endswith(".srf"):
            # If they are, add them to the srf files dictionary and fetch them
            NZGMDB_DATA.fetch(file)
            srf_files[Path(file).stem] = Path(NZGMDB_DATA.abspath) / file

    # --- Filter event ---
    event_row = event_df_full[event_df_full.evid == evid].iloc[0:1].iloc[0]

    # --- Build synthetic inputs ---
    event_df = pd.DataFrame([event_row])
    site_df, im_df = build_site_and_im_df(event_df)

    # --- Run function ---
    prop_df, extra_df, geom_df = compute_distances_for_event(
        event_row=event_row,
        im_df=im_df,
        site_df=site_df,
        cmt_df=cmt_df,
        domain_focal_df=domain_focal_df,
        taupo_polygon=taupo_polygon,
        srf_files=srf_files,
        hik_objs=hik_objs,
        puy_objs=puy_objs,
        nz_mech=nz_mech,
        slab_faulting_geo=slab_faulting_geo,
    )

    # --- Load benchmarks ---
    prop_benchmark = pd.read_csv(benchmark_dir / f"{evid}_propagation.csv")
    extra_benchmark = pd.read_csv(benchmark_dir / f"{evid}_extra.csv")
    geom_benchmark = pd.read_csv(benchmark_dir / f"{evid}_geometry.csv")

    # Sort to ensure stable comparison
    prop_df = prop_df.sort_values("sta").reset_index(drop=True)
    prop_benchmark = prop_benchmark.sort_values("sta").reset_index(drop=True)

    geom_df = geom_df.sort_values(["plane_id"]).reset_index(drop=True)
    geom_benchmark = geom_benchmark.sort_values(["plane_id"]).reset_index(drop=True)

    # --- Assertions ---
    pd.testing.assert_frame_equal(
        prop_df,
        prop_benchmark,
        atol=1e-3,
        check_dtype=False,
        check_index_type=False,
        check_names=False,
    )
    pd.testing.assert_frame_equal(
        extra_df,
        extra_benchmark,
        atol=1e-3,
        check_dtype=False,
        check_index_type=False,
        check_names=False,
    )
    pd.testing.assert_frame_equal(
        geom_df,
        geom_benchmark,
        atol=1e-3,
        check_dtype=False,
        check_index_type=False,
        check_names=False,
    )
