"""
    Contains the functions to add tectonic class to the data
"""

from pathlib import Path

import pandas as pd
import fiona


def add_tect_class(cmt_tectclass_ffp: Path, tect_shape_ffp: Path, geonet_cmt_ffp: Path, event_csv_ffp: Path):
    """
    Adds the tectonic class to the event data
    """

    cmt_tectclass_df = pd.read_csv(cmt_tectclass_ffp, low_memory=False)

    # Shape file for determining neotectonic domain
    shape = fiona.open(tect_shape_ffp)

    geonet_cmt_df = pd.read_csv(geonet_cmt_ffp, low_memory=False)
    df = pd.read_csv(event_csv_ffp, low_memory=False)

    # Additional files needed
    sub = "/home/joel/local/gmdb/tect_domain_folders/geospatial/Subduction_surfaces"
    NZ_SMDB_path = "/home/joel/local/gmdb/tect_domain_folders/Records/NZ_SMDB/Spectra_flatfiles/NZdatabase_flatfile_FAS_horizontal_GeoMean.csv"

    # Find the events thats are in the CMT data
    rows = df["evid"].isin(geonet_cmt_df["PublicID"])
    # Change the values of the source table mag, lat, lon, depth to the CMT's Mw, Latitude, Longitude, CD
    update_indices = geonet_cmt_df['PublicID'].isin(df.loc[rows, 'evid'])
    df.loc[rows, ['mag', 'lat', 'lon', 'depth']] = geonet_cmt_df.loc[
        update_indices, ['Mw', 'Latitude', 'Longitude', 'CD']].values
    # df.loc[rows, ["mag", "lat", "lon", "depth"]] = geonet_cmt_df.loc[rows, ["mag", "lat", "lon", "depth"]]
    # Set the for same events the mag_type, mag_method, loc_type and loc_grid to "Mw", "CMT", "CMT", "CMT"
