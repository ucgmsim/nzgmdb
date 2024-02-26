# Feb. 23, 2023: Updated MultiPath from pt[0] to pt.geoms[0]. Removed numba dependency from
# src_site_dist.py

from abc import ABC
import pandas as pd
import mag_scaling  # Extracted from the qcore package
from pyproj import Transformer  # conda install pyproj
import math
from math import sqrt, cos, sin, atan, atan2, acos, asin, pi
import numpy as np
from typing import List, Union, Dict
from qcore import formats, geo, srf
import os
import src_site_dist
import glob
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, LineString
from pandarallel import pandarallel  # conda install -c bjrn pandarallel
import obspy as op
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import UTCDateTime
import re

# Convert from WGS84 to NZGD2000
wgs2nztm = Transformer.from_crs(4326, 2193)
nztm2wgs = Transformer.from_crs(2193, 4326)

POINTS_PER_KILOMETER = (
    1 / 0.1
)  # 1km divided by distance between points (1km/0.1km gives 100m grid)


def fpcoor_2SDR(fnorm, slip):
    #     degrad = 180/np.pi
    if 1 - abs(fnorm[2]) < 1e-7:
        delt = 0
        phi = atan2(-slip[0], slip[1])
        clam = cos(phi) * slip[0] + sin(phi) * slip[1]
        slam = sin(phi) * slip[0] - cos(phi) * slip[1]
        lam = atan2(slam, clam)
    else:
        phi = atan2(-fnorm[0], fnorm[1])
        a = sqrt(fnorm[0] ** 2 + fnorm[1] ** 2)
        delt = atan2(a, -fnorm[2])
        clam = cos(phi) * slip[0] + sin(phi) * slip[1]
        slam = -slip[2] / sin(delt)
        lam = atan2(slam, clam)
        if delt > 0.5 * np.pi:
            delt = np.pi - delt
            phi = phi + np.pi
            lam = -lam
    strike = np.rad2deg(phi)
    if strike < 0:
        strike = strike + 360
    dip = np.rad2deg(delt)
    rake = np.rad2deg(lam)
    if rake <= -180:
        rake = rake + 360
    if rake > 180:
        rake = rake - 360
    return strike, dip, rake


def fpcoor_2FS(strike, dip, rake):
    #     degrad = 180/np.pi
    phi = np.deg2rad(strike)
    delt = np.deg2rad(dip)
    lam = np.deg2rad(rake)
    #     phi = strike/degrad
    #     delt = dip/degrad
    #     lam = rake/degrad

    fnorm = -sin(delt) * sin(phi), sin(delt) * cos(phi), -cos(delt)
    slip = (
        cos(lam) * cos(phi) + cos(delt) * sin(lam) * sin(phi),
        cos(lam) * sin(phi) - cos(delt) * sin(lam) * cos(phi),
        -sin(lam) * sin(delt),
    )

    return fnorm, slip


def mech_rot(in_data):
    norm1 = in_data[0:3]
    norm2 = in_data[3:6]
    slip1 = in_data[6:9]
    slip2 = in_data[9:12]

    degrad = 180 / np.pi
    rotemp = np.zeros(4)
    for itr in range(0, 4):
        n1 = np.empty(3)
        n2 = np.empty(3)
        if itr < 2:
            norm2_temp = norm2
            slip2_temp = slip2
        else:
            norm2_temp = slip2
            slip2_temp = norm2
        if (itr == 1) or (itr == 3):
            norm2_temp = -norm2_temp
            slip2_temp = -slip2_temp

        B1 = np.cross(norm1, slip1)
        B2 = np.cross(norm2_temp, slip2_temp)

        phi1 = np.dot(norm1, norm2_temp)
        phi2 = np.dot(slip1, slip2_temp)
        phi3 = np.dot(B1, B2)

        # In some cases, identical dot products produce values incrementally higher than 1
        if phi1 > 1:
            phi1 = 1
        if phi2 > 1:
            phi2 = 1
        if phi3 > 1:
            phi3 = 1
        if phi1 < -1:
            phi1 = -1
        if phi2 < -1:
            phi2 = -1
        if phi3 < -1:
            phi3 = -1

        phi = acos(phi1), acos(phi2), acos(phi3)

        # If the mechanisms are very close, rotation = 0
        if (phi[0] < 1e-4) and (phi[1] < 1e-4) and (phi[2] < 1e-4):
            rotemp[itr] = 0
        # If one vector is the same, it is the rotation axis
        elif phi[0] < 1e-4:
            rotemp[itr] = degrad * phi[1]
        elif phi[1] < 1e-4:
            rotemp[itr] = degrad * phi[2]
        elif phi[2] < 1e-4:
            rotemp[itr] = degrad * phi[0]
        # Find difference vectors - the rotation axis must be orthogonal to all three of
        # these vectors
        else:
            n = np.array(
                [
                    np.array(norm1) - np.array(norm2_temp),
                    np.array(slip1) - np.array(slip2_temp),
                    np.array(B1) - np.array(B2),
                ]
            )
            #             n = np.zeros([3,3])
            #             for i in range(0,3):
            #                 n[i,0] = norm1[i] - norm2_temp[i]
            #                 n[i,1] = slip1[i] - slip2_temp[i]
            #                 n[i,2] = B1[i] - B2[i]

            #             scale = np.zeros(3)
            #             for j in range(0,3):
            #                 scale[j] = sqrt(n[0][j] ** 2 + n[1][j] ** 2 + n[2][j] ** 2)
            #                 print(scale)
            #                 for i in range(0,3):
            #                     n[i,j] = n[i,j] / scale[j]
            scale = np.sqrt(np.sum(n**2, axis=0))
            n = n / scale
            qdot = (
                n[0][1] * n[0][2] + n[1][1] * n[1][2] + n[2][1] * n[2][2],
                n[0][0] * n[0][2] + n[1][0] * n[1][2] + n[2][0] * n[2][2],
                n[0][0] * n[0][1] + n[1][0] * n[1][1] + n[2][0] * n[2][1],
            )

            # Use the two largest difference vectors, as long as they aren't orthogonal
            iout = -1
            for i in range(0, 3):
                if qdot[i] > 0.9999:
                    iout = i
            if iout == -1:
                qmins = 10000
                qmins = scale.min()
                iout = np.where(scale == scale.min())[0][0]
            k = 1
            for j in range(0, 3):
                if j != iout:
                    if k == 1:
                        n1 = n[:, j]
                        k = 2
                    else:
                        n2 = n[:, j]
            # Find rotation axis by taking cross product
            R = np.cross(n1, n2)
            #             scaleR = np.sqrt(R[0] ** 2 + R[1] ** 2 + R[2] ** 2)
            scaleR = sqrt(np.sum(R**2))
            R = R / scaleR

            # Find rotation using axis furthest from rotation axis
            theta = np.array(
                [acos(np.dot(norm1, R)), acos(np.dot(slip1, R)), acos(np.dot(B1, R))]
            )
            qmindifs = abs(theta - np.pi / 2)
            iuse = np.argmin(
                qmindifs[0:2]
            )  # Pick the minimum from either the norm or slip axes
            rotemp[itr] = (cos(phi[iuse]) - cos(theta[iuse]) * cos(theta[iuse])) / (
                sin(theta[iuse]) * sin(theta[iuse])
            )
            if rotemp[itr] > 1:
                rotemp[itr] = 1
            if rotemp[itr] < -1:
                rotemp[itr] = -1
            rotemp[itr] = degrad * acos(rotemp[itr])

    # Find the minimum rotation for the 4 combos and change norm2 and slip2
    rota = 180
    irot = np.argmin(rotemp)
    rota = rotemp[irot]
    if irot < 2:
        norm2_out = norm2
        slip2_out = slip2
        plane_out = 1
    else:
        norm2_out = slip2
        slip2_out = norm2
        plane_out = 2
    if (irot == 1) or (irot == 3):
        norm2_out = -norm2_out
        slip2_out = -slip2_out

    return rota, norm2_out, slip2_out, plane_out


class Fault(ABC):
    subfault_spacing = 0.1
    type = 0
    _mag = None
    _moment = None
    name = None
    _latitude = None
    _longitude = None
    _strike = None
    _rake = None
    _dip = None
    _depth = None
    _width = None
    _length = None
    _shypo = None
    _dhypo = None

    @property
    def pid(self):
        return self.name

    @pid.setter
    def pid(self, value):
        self.name = value

    @property
    def mom(self):
        if self._moment is None:
            self._moment = mag2mom(self._mag)
        return self._moment

    @property
    def latitude(self):
        return self._latitude

    @latitude.setter
    def latitude(self, value):
        self._latitude = value

    @property
    def longitude(self):
        return self._longitude

    @longitude.setter
    def longitude(self, value):
        self._longitude = value

    @property
    def magnitude(self):
        return self._mag

    @magnitude.setter
    def magnitude(self, mag):
        if mag > 11:
            raise ValueError(
                f"Given mag {mag} is greater than theoretically possible 11"
            )
        self._mag = mag

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def to_dict(self):
        return {
            "type": self.type,
            "magnitude": self._mag,
            "moment": self.mom,
            "name": self.name,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "strike": self._strike,
            "rake": self._rake,
            "dip": self._dip,
            "depth": self._depth,
        }

    @property
    def rake(self):
        return self._rake

    @rake.setter
    def rake(self, value):
        value = ((value + 180) % 360) - 180
        self._set_rake(value)

    def _set_rake(self, value):
        self._rake = value

    @property
    def dip(self):
        return self._dip

    @dip.setter
    def dip(self, value):
        if 90 < value < 180:
            self.strike = self.strike + 180
            self.rake = -self.rake
            value = 180 - value
        elif value < 0 or value > 180:
            # The fault is now above ground
            raise ValueError(f"Invalid dip value: {value}")

        self._set_dip(value)

    def _set_dip(self, value):
        self._dip = value


def TVZ_path_calc(
    row,
    sta_df,
    taupo_polygon,
    tect_domain_points,
    wgs2nztm,
    r_epis,
    rrups_lon,
    rrups_lat,
    rrups_depth,
):
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon, LineString
    import numpy as np

    # Taupo VZ polygon acquired from https://www.geonet.org.nz/data/supplementary/earthquake_location_grope

    event_id = row.evid
    ev_lat = row.lat
    ev_lon = row.lon
    ev_depth = row.depth
    reloc = row.reloc
    ev_transform = wgs2nztm.transform(ev_lat, ev_lon)
    rrups_transform = wgs2nztm.transform(rrups_lat, rrups_lon)
    #         if ev_lon < 0:
    #             ev_lon = ev_lon+360
    network = sta_df.net
    station = sta_df.sta
    sta_lat = sta_df.lat
    sta_lon = sta_df.lon
    sta_elev = sta_df.elev
    sta_transform = wgs2nztm.transform(sta_lat, sta_lon)
    #         if sta_lon < 0:
    #             sta_lon = sta_lon+360
    tvz_lengths = []
    boundary_dists_rjb = []
    #     boundary_dists_rrup = []
    ii = 0
    for i, sta in sta_df.iterrows():
        sta_transform = wgs2nztm.transform(sta.lat, sta.lon)
        #         dist, az, b_az = op.geodetics.gps2dist_azimuth(ev_lat, ev_lon, sta.lat, sta.lon)
        #         dists.append(dist)
        #         azs.append(az)
        #         b_azs.append(b_az)
        #         r_epi = dist/1000
        #         r_hyp = (r_epi ** 2 + (ev_depth + sta.elev/1000) ** 2) ** 0.5
        #         r_epis.append(r_epi)
        #         r_hyps.append(r_hyp)
        line = LineString(
            [
                [rrups_transform[0][ii], rrups_transform[1][ii]],
                [sta_transform[0], sta_transform[1]],
            ]
        )
        #         line = LineString([[ev_transform[0],ev_transform[1]],[sta_transform[0],sta_transform[1]]])

        tvz_length = 0
        boundary_dist_rjb = None
        #         boundary_dist_rrup = None

        if line.intersection(taupo_polygon):
            polin = LineString(list(taupo_polygon.exterior.coords))
            pt = polin.intersection(line)
            if taupo_polygon.contains(Point(sta_transform)):
                boundary_dist_rjb = 0
            #                 boundary_dist_rrup = 0
            else:
                if pt.geom_type != "LineString":
                    if pt.geom_type == "MultiPoint":
                        pt = pt.geoms[0]
                    boundary_dist_rjb = (
                        np.sqrt(
                            (pt.xy[0][-1] - sta_transform[0]) ** 2
                            + (pt.xy[1][-1] - sta_transform[1]) ** 2
                        )
                        / 1000
                    )
                #                     boundary_dist_rrup = np.sqrt((pt.xy[0][-1] - sta_transform[0]) ** 2 + (pt.xy[1][-1] - sta_transform[1]) ** 2 + ((sta.depth - rrups_depth[i]) * 1000) ** 2) / 1000
                else:
                    boundary_dist_rjb = None
            #                     boundary_dist_rrup = None
            line_points = line.intersection(taupo_polygon)
            tvz_length = line_points.length / 1000 / r_epis[ii]

            if tvz_length > 1:
                tvz_length = 1
        #             TVZ_path_plot(df,taupo_polygon,line,nztm2wgs,r_epis[ii],sta)
        tvz_lengths.append(tvz_length)
        boundary_dists_rjb.append(boundary_dist_rjb)
        #         boundary_dists_rrup.append(boundary_dist_rrup)
        ii += 1

    # Output evid, net, sta, r_epi, r_hyp, az, b_az, reloc, rrup, and rjb. toa has
    # been omitted. This is better left in the phase arrival table.
    return tvz_lengths, boundary_dists_rjb


def rotate_back(x, y, xs, ys, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    from pyproj import CRS
    from pyproj import Transformer
    import math
    import numpy as np

    angle = np.radians(angle)
    # 	ox = orilon
    # 	oy = orilat
    # 	transformer_from_latlon = Transformer.from_crs(4326, 2193) # WSG84 to New Zealand NZDG2000 coordinate transform
    # 	transformer_to_latlon = Transformer.from_crs(2193, 4326)

    # 	ox, oy = transformer_from_latlon.transform(orilat,orilon)
    ox, oy = x, y
    #     px = ox+xs*1000
    #     py = oy-ys*1000
    px = xs
    py = ys
    # 	px, py = transformer_from_latlon.transform(lats,lons)

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    lats, lons = nztm2wgs.transform(qx, qy)
    # 	stations['z'] = (stations.elevation-dx)/dx
    return lats, lons


def geonet_rupture(file):
    data = []
    with open(file, "r") as fo:
        lines = fo.readlines()
        for line in lines:
            if "Surface geographical reference point" in line:
                olat, olon = re.split(r"\s{2,}", line.split(":")[1].strip())
                olat = float(olat.split("=")[1])
                olon = float(olon.split("=")[1])

    columns = [
        "segment",
        "northing",
        "easting",
        "depth",
        "slip",
        "strike",
        "dip",
        "rake",
        "width",
        "length",
        "rupture_time",
    ]
    df = pd.read_csv(file, comment="%", sep="\s+", header=None, names=columns)
    segments = np.unique(df.segment)

    #     olat, olon = -43.545, 172.69
    ox, oy = wgs2nztm.transform(olat, olon)

    srf_points = []
    srf_header: List[Dict[str, Union[int, float]]] = []
    plane_offset = 0
    for segment in segments:
        df_sub = df[df.segment == segment]
        strike, dip, rake = df_sub.iloc[0][["strike", "dip", "rake"]].values
        dip_dir = strike + 90
        if dip_dir >= 360:
            dip_dir = dip_dir - 360
        f_length = df_sub.iloc[0]["length"] / 1000
        f_width = df_sub.iloc[0]["width"] / 1000
        dtop = df_sub.depth.min() / 1000
        dbottom = df_sub.depth.max() / 1000
        height = dbottom - dtop

        qx, qy = ox + df_sub.iloc[0].easting, oy + df_sub.iloc[0].northing
        lat1, lon1 = nztm2wgs.transform(qx, qy)
        qx, qy = (
            ox + df_sub[df_sub.depth == dtop * 1000].easting.iloc[-1],
            oy + df_sub[df_sub.depth == dtop * 1000].northing.iloc[-1],
        )
        lat2, lon2 = nztm2wgs.transform(qx, qy)

        plane_point_distance = geo.ll_dist(lon1, lat1, lon2, lat2)

        nstrike = round(plane_point_distance * POINTS_PER_KILOMETER)
        strike_dist = plane_point_distance / nstrike

        end_strike = geo.ll_bearing(lon1, lat1, lon2, lat2)

        #     nstrike = len(df_sub[df_sub.depth == dtop])
        #     ndip = len(df_sub.depth.unique())
        #
        #     dip_dist = height / np.sin(np.deg2rad(dip))
        #     hdip_dist = np.cos(np.deg2rad(dip)) * width
        #     vdip_dist = np.sin(np.deg2rad(dip)) * width

        for j in range(nstrike):
            top_lat, top_lon = geo.ll_shift(lat1, lon1, strike_dist * j, end_strike)
            srf_points.append([top_lon, top_lat, dtop])

        width = abs(height / np.tan(np.deg2rad(dip)))
        dip_dist = height / np.sin(np.deg2rad(dip))

        ndip = int(round(dip_dist * POINTS_PER_KILOMETER))
        hdip_dist = width / ndip
        vdip_dist = height / ndip

        for j in range(1, ndip):
            hdist = j * hdip_dist
            vdist = j * vdip_dist + dtop
            for local_lon, local_lat, local_depth in srf_points[
                plane_offset : plane_offset + nstrike
            ]:
                new_lat, new_lon = geo.ll_shift(local_lat, local_lon, hdist, dip_dir)
                srf_points.append([new_lon, new_lat, vdist])

        plane_offset += nstrike * ndip
        srf_header.append(
            {
                "nstrike": nstrike,
                "ndip": ndip,
                "strike": strike,
                "dip": dip,
                "rake": rake,
                "length": plane_point_distance,
                "width": width,
            }
        )
    srf_points = np.array(srf_points)

    return srf_points, srf_header


def mw_to_a_nzcustomized(mw, tect_class):
    # Compute estimated rupture area, dependent on earthquake magnitude and tectonic
    # classification.
    if tect_class == "Interface":
        C = 4.15
    elif tect_class == "Slab":
        C = 4.0
    else:
        C = 4.2

    A = 10 ** (mw - C)

    return A


def lw_nzcustomized(area, tect_class):
    # Derive length and width from area depending on tectonic classification for eqs with
    # magnitudes greater than 7
    if tect_class == "Slab" or tect_class == "Interface":
        ratio = 1.2
    else:
        ratio = 1.3
    d = np.sqrt(area)
    l = d * ratio
    w = d / ratio

    return l, w


# Get event magnitude, location
# Get event strike, dip, rake


# for i, row in event_df[0:10].iterrows():
def path_props(
    df,
    df_merged,
    df_station,
    taupo_polygon,
    cmt_df,
    domain_focal_df,
    srf_files,
    srfs,
    geonet_cmt_df,
    event_out_file,
    props_out_file,
):
    # for index,df in event_sub.iterrows():
    row = df.copy()
    #     row = event_df[event_df.evid == '1506176'].iloc[0].copy()
    #     print(row.name)
    evid = row.evid
    lat = row.lat
    lon = row.lon
    depth = row.depth
    mag = row.mag
    reloc = row.reloc

    if np.isnan(mag):
        mag = row.mag_orig
    if mag != -9 and ~np.isnan(mag):
        try:
            # Calculate area from event tectonic type
            sta_df = df_merged[df_merged.evid.isin([evid])].merge(
                df_station[["net", "sta", "lat", "lon", "elev"]], on=["sta"]
            )
            indexNames = sta_df[
                (sta_df["net"] == "AU") & (sta_df["sta"] == "ARPS")
            ].index  # remove AU ARPS
            sta_df.drop(indexNames, inplace=True)
            sta_df["depth"] = sta_df["elev"] / -1000
            stations = sta_df[["lon", "lat", "depth"]].to_numpy()

            # If interface
            if row.tect_class == "Interface":
                magnitude_scaling_relation = (
                    mag_scaling.MagnitudeScalingRelations.SKARLATOUDIS2016
                )
            # If slab
            elif row.tect_class == "Slab" and mag >= 5.9 and mag <= 7.8:
                magnitude_scaling_relation = (
                    mag_scaling.MagnitudeScalingRelations.STRASSER2010SLAB
                )
            # If other
            else:
                magnitude_scaling_relation = (
                    mag_scaling.MagnitudeScalingRelations.LEONARD2014
                )

            srf_flag = np.isin(evid, srfs)
            georup_flag = np.isin(evid, georups)

            # Search for SRF files from which to draw parameters
            if srf_flag:
                srf_file = srf_files[np.where(np.isin(srfs, evid) == True)[0][0]]
                srf_points = srf.read_srf_points(srf_file)
                srf_header = srf.read_header(srf_file, idx=True)
                ztor = srf_points[0][2]
                dbottom = srf_points[-1][2]
                f_type = "ff"
                cmt = geonet_cmt_df[geonet_cmt_df.PublicID == evid].iloc[0]

                #         if len(srf_header) > 1:
                strike1 = cmt.strike1
                dip1 = cmt.dip1
                rake1 = cmt.rake1
                strike2 = cmt.strike2
                dip2 = cmt.dip2
                rake2 = cmt.rake2

                fault_strike = srf_header[0]["strike"]
                strike1_diff = abs(fault_strike - strike1)
                if strike1_diff > 180:
                    strike1_diff = 360 - strike1_diff
                strike2_diff = abs(fault_strike - strike2)
                if strike2_diff > 180:
                    strike2_diff = 360 - strike2_diff

                if strike1_diff < strike2_diff:
                    strike = strike1
                    rake = rake1
                    dip = dip1
                else:
                    strike = strike2
                    rake = rake2
                    dip = dip2
                length = np.sum([header["length"] for header in srf_header])
                dip_dist = np.mean([header["width"] for header in srf_header])
            # Search for GEONET rupture models from which to draw parameters
            elif georup_flag:
                f_type = "geonet_rm"
                georup_file = georup_files[
                    np.where(np.isin(georups, evid) == True)[0][0]
                ]
                srf_points, srf_header = geonet_rupture(georup_file)
                ztor = srf_points[0][2]
                dbottom = srf_points[-1][2]

                strike = np.mean([header["strike"] for header in srf_header])
                dip = np.mean([header["dip"] for header in srf_header])
                rake = np.mean([header["rake"] for header in srf_header])
                length = np.sum([header["length"] for header in srf_header])
                dip_dist = np.mean([header["width"] for header in srf_header])
            else:
                cmt = cmt_df[cmt_df.PublicID == evid]
                cmt_2 = geonet_cmt_df[geonet_cmt_df.PublicID == evid]
                if len(cmt) > 0:
                    cmt = cmt.iloc[0]
                    strike = cmt.strike1
                    dip = cmt.dip1
                    rake = cmt.rake1
                    f_type = "cmt"
                elif len(cmt_2) > 0:
                    cmt_2 = cmt_2.iloc[0]
                    f_type = "cmt_unc"
                    strike1 = cmt_2.strike1
                    dip1 = cmt_2.dip1
                    rake1 = cmt_2.rake1
                    norm, slip = fpcoor_2FS(strike1, dip1, rake1)
                    strike2 = cmt_2.strike2
                    dip2 = cmt_2.dip2
                    rake2 = cmt_2.rake2
                    if row.domain_no != 0:
                        domain = domain_focal_df[
                            domain_focal_df.Domain_No == row.domain_no
                        ].iloc[0]
                        do_strike = domain.strike
                        do_rake = domain.rake
                        do_dip = domain.dip
                    else:
                        do_strike = 220
                        do_dip = 45
                        do_rake = 90

                    do_norm, do_slip = fpcoor_2FS(do_strike, do_dip, do_rake)
                    rot_in = np.hstack((do_norm, norm, do_slip, slip))
                    rota, norm_out, slip_out, plane_out = mech_rot(rot_in)

                    if plane_out == 1:
                        strike = strike1
                        dip = dip1
                        rake = rake1
                    else:
                        strike = strike2
                        dip = dip2
                        rake = rake2

                #                     strike, dip, rake = fpcoor_2SDR(norm_out,slip_out)

                else:
                    f_type = "domain"
                    if row.domain_no == 0:
                        strike = 220
                        dip = 45
                        rake = 90
                    else:
                        domain = domain_focal_df[
                            domain_focal_df.Domain_No == row.domain_no
                        ].iloc[0]
                        strike = domain.strike
                        rake = domain.rake
                        dip = domain.dip
                fault = Fault
                fault.name = evid
                fault.latitude = lat
                fault.longitude = lon
                fault.depth = depth
                fault.magnitude = mag
                fault.moment = float(mag_scaling.mag2mom(mag))
                fault.strike = strike
                fault.dip = dip
                fault.rake = rake
                fault.magnitude_scaling_relation = magnitude_scaling_relation

                srf_points = []
                srf_header = []

                dip_dir = strike + 90
                if dip_dir >= 360:
                    dip_dir = dip_dir - 360

                # Area, length, and width using various magnitude - area scaling
                # relationships.
                area = mag_scaling.get_area(fault)
                dip_dist = mag_scaling.get_width(fault)
                length = mag_scaling.get_length(fault)

                # Area, length, and width from NZ Customized
                #                 area = mw_to_a_nzcustomized(mag, row.tect_class)
                #                 if mag < 7:
                #                     dip_dist = np.sqrt(area)
                #                     length = dip_dist
                #                 else:
                #                     length, dip_dist = lw_nzcustomized(area, row.tect_class)

                x, y = wgs2nztm.transform(fault.latitude, fault.longitude)
                z = fault.depth

                x1 = x - (length * 1000 / 2)
                x2 = x + (length * 1000 / 2)
                y1 = y
                y2 = y

                lat1, lon1 = rotate_back(x, y, x1, y1, strike)
                lat2, lon2 = rotate_back(x, y, x2, y2, strike)

                height = math.sin(math.radians(dip)) * dip_dist
                width = abs(height / math.tan(math.radians(dip)))
                dtop = z - (height / 2)  # The same as Ztor
                if dtop < 0:
                    dtop = 0
                ztor = dtop
                dbottom = dtop + height

                end_strike = geo.ll_bearing(lon1, lat1, lon2, lat2)

                nstrike = int(round(length * POINTS_PER_KILOMETER))
                if nstrike == 0:
                    rxs, rys = [], []

                    r_epis = geo.get_distances(
                        np.dstack([sta_df.lon.values, sta_df.lat.values])[0], lon, lat
                    )
                    r_hyps = np.sqrt(r_epis**2 + (depth - sta_df.depth.values) ** 2)
                    azs = np.array(
                        [
                            geo.ll_bearing(lon, lat, station[0], station[1])
                            for station in stations
                        ]
                    )
                    b_azs = np.array(
                        [
                            geo.ll_bearing(station[0], station[1], lon, lat)
                            for station in stations
                        ]
                    )
                    rrups_lon, rrups_lat, rrups_depth = (
                        np.repeat(row.lon, len(stations)),
                        np.repeat(row.lat, len(stations)),
                        np.repeat(row.depth, len(stations)),
                    )
                    tvz_lengths, boundary_dists_rjb = TVZ_path_calc(
                        row,
                        sta_df,
                        taupo_polygon,
                        tect_domain_points,
                        wgs2nztm,
                        r_epis,
                        rrups_lon,
                        rrups_lat,
                        rrups_depth,
                    )

                    prop_df = pd.DataFrame(
                        columns=[
                            "evid",
                            "net",
                            "sta",
                            "r_epi",
                            "r_hyp",
                            "r_jb",
                            "r_rup",
                            "r_x",
                            "r_y",
                            "r_tvz",
                            "r_xvf",
                            "az",
                            "b_az",
                            "reloc",
                        ]
                    )
                    prop_df[["evid", "net", "sta"]] = sta_df[["evid", "net", "sta"]]
                    prop_df["r_epi"] = r_epis
                    prop_df["r_hyp"] = r_hyps
                    prop_df["r_jb"] = r_epis
                    prop_df["r_rup"] = r_hyps
                    if len(rxs) > 0:
                        prop_df["r_x"] = rxs
                    if len(rys) > 0:
                        prop_df["r_y"] = rys
                    prop_df["r_tvz"] = tvz_lengths
                    prop_df["r_xvf"] = boundary_dists_rjb
                    #                     prop_df['r_xvf_rup'] = boundary_dists_rrup
                    prop_df["az"] = azs
                    prop_df["b_az"] = b_azs
                    prop_df["f_type"] = f_type
                    prop_df["reloc"] = reloc

                    row["strike"] = strike
                    row["dip"] = dip
                    row["rake"] = rake
                    row["f_length"] = length
                    row["f_width"] = dip_dist
                    row["f_type"] = f_type
                    row["z_tor"] = dtop
                    row["z_bor"] = dbottom

                    row_out = row.copy()
                    row_out = row_out.to_frame().T
                    row_out.to_csv(event_out_file, index=False, header=False, mode="a")
                    prop_df.to_csv(props_out_file, index=False, header=False, mode="a")
                    #                     continue
                    return
                #             return row, prop_df
                strike_dist = length / nstrike

                # From seistech_internal/scripts/empirical_db/calc_distances_flt.py
                for j in range(nstrike):
                    top_lat, top_lon = geo.ll_shift(
                        lat1, lon1, strike_dist * j, end_strike
                    )
                    srf_points.append([top_lon, top_lat, dtop])

                ndip = int(round(dip_dist * POINTS_PER_KILOMETER))
                hdip_dist = width / ndip
                vdip_dist = height / ndip

                plane_offset = 0
                for j in range(
                    1, ndip
                ):  # added plus 1 to ndip so that it has the bottom srf points
                    hdist = j * hdip_dist
                    vdist = j * vdip_dist + dtop
                    for local_lon, local_lat, local_depth in srf_points[
                        plane_offset : plane_offset + nstrike
                    ]:
                        new_lat, new_lon = geo.ll_shift(
                            local_lat, local_lon, hdist, dip_dir
                        )
                        srf_points.append([new_lon, new_lat, vdist])
                plane_offset += nstrike * ndip
                srf_header.append({"nstrike": nstrike, "ndip": ndip, "strike": strike})

                srf_points = np.array(srf_points)

            #     plt.plot([lon1,lon2],[lat1,lat2],'b')
            #     plt.plot(lon,lat,'bo')
            #     plt.plot(lon1,lat1,'ro')
            #     plt.plot(lon2,lat2,'go')
            #     plt.axis('equal')
            #     plt.show()

            (
                rrups,
                rjbs,
                rrups_lon,
                rrups_lat,
                rrups_depth,
            ) = src_site_dist.calc_rrup_rjb(srf_points, stations)
            if srf_header:
                rxs, rys = src_site_dist.calc_rx_ry(srf_points, srf_header, stations)
            else:
                rxs, rys = [], []

            # rotate line to be along strike
            # determine x1,y1,x2,y2 (cartesian coordinates)
            # convert to lat1,lon1,lat2,lon2 (geographic coordinates)
            # determine dtop and dbottom
            # generate srf header and srf points
            # determine stations from arrivals and ims (lat, lon, depth) and write to stations np array
            # calculate distances rjb, rrup, rx, ry

            # old tvz function to calculate r_hyp, r_epi
            #         dists, azs, b_azs, r_epis, r_hyps, tvz_lengths = TVZ_path_calc(row,sta_df,taupo_polygon,tect_domain_points,wgs2nztm)

            # New equations to calculate r_hyp and r_epi
            r_epis = geo.get_distances(
                np.dstack([sta_df.lon.values, sta_df.lat.values])[0], lon, lat
            )
            r_hyps = np.sqrt(r_epis**2 + (depth - sta_df.depth.values) ** 2)
            azs = np.array(
                [
                    geo.ll_bearing(lon, lat, station[0], station[1])
                    for station in stations
                ]
            )
            b_azs = np.array(
                [
                    geo.ll_bearing(station[0], station[1], lon, lat)
                    for station in stations
                ]
            )
            tvz_lengths, boundary_dists_rjb = TVZ_path_calc(
                row,
                sta_df,
                taupo_polygon,
                tect_domain_points,
                wgs2nztm,
                rjbs,
                rrups_lon,
                rrups_lat,
                rrups_depth,
            )

            prop_df = pd.DataFrame(
                columns=[
                    "evid",
                    "net",
                    "sta",
                    "r_epi",
                    "r_hyp",
                    "r_jb",
                    "r_rup",
                    "r_x",
                    "r_y",
                    "r_tvz",
                    "r_xvf",
                    "az",
                    "b_az",
                    "reloc",
                ]
            )
            prop_df[["evid", "net", "sta"]] = sta_df[["evid", "net", "sta"]]
            prop_df["r_epi"] = r_epis
            prop_df["r_hyp"] = r_hyps
            prop_df["r_jb"] = rjbs
            prop_df["r_rup"] = rrups
            if len(rxs) > 0:
                prop_df["r_x"] = rxs
            if len(rys) > 0:
                prop_df["r_y"] = rys
            prop_df["r_tvz"] = tvz_lengths
            prop_df["r_xvf"] = boundary_dists_rjb
            #             prop_df['r_xvf_rup'] = boundary_dists_rrup
            prop_df["az"] = azs
            prop_df["b_az"] = b_azs
            prop_df["f_type"] = f_type
            prop_df["reloc"] = reloc

            row["strike"] = strike
            row["dip"] = dip
            row["rake"] = rake
            row["f_length"] = length
            row["f_width"] = dip_dist
            row["f_type"] = f_type
            row["z_tor"] = ztor
            row["z_bor"] = dbottom

            row_out = row.copy()
            row_out = row_out.to_frame().T
            row_out.to_csv(event_out_file, index=False, header=False, mode="a")
            prop_df.to_csv(props_out_file, index=False, header=False, mode="a")
            return
        #             continue
        except Exception as e:
            print(e)
    else:
        sta_df = df_merged[df_merged.evid == evid].merge(
            df_station[["net", "sta", "lat", "lon", "elev"]], on=["sta"]
        )
        indexNames = sta_df[
            (sta_df["net"] == "AU") & (sta_df["sta"] == "ARPS")
        ].index  # remove AU ARPS
        sta_df.drop(indexNames, inplace=True)
        sta_df["depth"] = sta_df["elev"] / -1000
        stations = sta_df[["lon", "lat", "depth"]].to_numpy()

        r_epis = geo.get_distances(
            np.dstack([sta_df.lon.values, sta_df.lat.values])[0], lon, lat
        )
        r_hyps = np.sqrt(r_epis**2 + (depth - sta_df.depth.values) ** 2)
        rrups_lon, rrups_lat, rrups_depth = (
            np.repeat(row.lon, len(stations)),
            np.repeat(row.lat, len(stations)),
            np.repeat(row.depth, len(stations)),
        )
        azs = np.array(
            [geo.ll_bearing(lon, lat, station[0], station[1]) for station in stations]
        )
        b_azs = np.array(
            [geo.ll_bearing(station[0], station[1], lon, lat) for station in stations]
        )
        tvz_lengths, boundary_dists_rjb = TVZ_path_calc(
            row,
            sta_df,
            taupo_polygon,
            tect_domain_points,
            wgs2nztm,
            r_epis,
            rrups_lon,
            rrups_lat,
            rrups_depth,
        )

        prop_df = pd.DataFrame(
            columns=[
                "evid",
                "net",
                "sta",
                "r_epi",
                "r_hyp",
                "r_jb",
                "r_rup",
                "r_x",
                "r_y",
                "r_tvz",
                "r_xvf",
                "az",
                "b_az",
                "reloc",
            ]
        )
        prop_df[["evid", "net", "sta"]] = sta_df[["evid", "net", "sta"]]
        prop_df["r_epi"] = r_epis
        prop_df["r_hyp"] = r_hyps
        prop_df["r_jb"] = ""
        prop_df["r_rup"] = ""
        prop_df["r_x"] = ""
        prop_df["r_y"] = ""
        prop_df["r_tvz"] = tvz_lengths
        prop_df["r_xvf"] = boundary_dists_rjb
        #         prop_df['r_xvf_rup'] = boundary_dists_rrup
        prop_df["az"] = azs
        prop_df["b_az"] = b_azs
        prop_df["f_type"] = ""
        prop_df["reloc"] = reloc

        row["strike"] = ""
        row["dip"] = ""
        row["rake"] = ""
        row["f_length"] = ""
        row["f_width"] = ""
        row["f_type"] = ""
        row["z_tor"] = depth
        row["z_bor"] = depth

        row_out = row.copy()
        row_out = row_out.to_frame().T
        row_out.to_csv(event_out_file, index=False, header=False, mode="a")
        prop_df.to_csv(props_out_file, index=False, header=False, mode="a")


#     return row_out, prop_df

directory = "../"

geonet_cmt_df = pd.read_csv(
    directory + "focal/GeoNet_CMT_solutions.csv", low_memory=False
)
cmt_df = pd.read_csv(
    directory + "focal/GeoNet_CMT_solutions_20201129_PreferredNodalPlane_v1.csv",
    low_memory=False,
)
event_df = pd.read_csv(
    directory + "output/earthquake_source_table_tectdomain.csv", low_memory=False
)
event_df["evid"] = event_df.evid.astype("object")

### Filter data to be within lat/lon limitations.
event_df_copy = event_df.copy()
event_df_copy.loc[event_df_copy.lon < 0, "lon"] = (
    360 + event_df_copy.lon[event_df_copy.lon < 0]
)
event_df = event_df[
    ((event_df_copy.lon < 190) & (event_df_copy.lon > 155)) & (event_df_copy.lat < -15)
].reset_index(drop=True)
event_df["datetime"] = pd.to_datetime(event_df.datetime, utc=True)

domain_focal_df = pd.read_csv(
    directory + "focal/focal_mech_tectonic_domain_v1.csv", low_memory=False
)
srf_files = glob.glob(directory + "focal/SrfSourceModels/*.srf")
srfs = [os.path.basename(srf_file).split(".")[0] for srf_file in srf_files]

georup_files = glob.glob(
    directory + "focal/rupture-models/**/*_seismic_model.txt", recursive=True
)
georups = [os.path.basename(georup_file).split("_")[0] for georup_file in georup_files]

df_im = pd.read_csv(
    directory + "output/IM_catalogue/complete_ground_motion_im_catalogue_final.csv",
    low_memory=False,
)
df_im_sub = df_im[df_im.duplicated(subset=["sta", "evid"]) == False][["evid", "sta"]]
df_arr = pd.read_csv(directory + "output/phase_arrival_table.csv", low_memory=False)
df_arr_sub = df_arr[df_arr.duplicated(subset=["sta", "evid"]) == False][["evid", "sta"]]
df_merged = pd.concat([df_im_sub, df_arr_sub], axis=0, ignore_index=True)
df_merged = df_merged[df_merged.duplicated() == False]

tect_domain_points = pd.read_csv(
    directory + "TectonicDomains/tectonic_domain_polygon_points.csv", low_memory=False
)
tvz_points = tect_domain_points[tect_domain_points.domain_no == 4][
    ["latitude", "longitude"]
]
taupo_transform = np.dstack(
    np.array(wgs2nztm.transform(tvz_points.latitude, tvz_points.longitude))
)[0]
taupo_polygon = Polygon(taupo_transform)

# Load station information from FDSN in case station csv is not complete
client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client("IRIS")
inventory_NZ = client_NZ.get_stations()
inventory_IU = client_IU.get_stations(
    network="IU", station="SNZO,AFI,CTAO,RAO,FUNA,HNR,PMG"
)
inventory_AU = client_IU.get_stations(network="AU")
inventory = inventory_NZ + inventory_IU + inventory_AU
station_info = []
for network in inventory:
    for station in network:
        station_info.append(
            [
                network.code,
                station.code,
                station.latitude,
                station.longitude,
                station.elevation,
            ]
        )
# 		station_df = station_df.append({'net':network.code,'sta':station.code,'lat':station.latitude,
# 			'lon':station.longitude,'elev':station.elevation},True)

df_station = pd.DataFrame(station_info, columns=["net", "sta", "lat", "lon", "elev"])
df_station = df_station.drop_duplicates().reset_index(drop=True)

years = np.arange(2009, 2010)
# months = np.arange(1,13)
# years = np.unique(geonet.origintime.values.astype('datetime64[Y]').astype(int)+1970)
for year in years:
    # 	for month in months:
    process_year = year
    print("Processing for year " + str(process_year))
    event_out_file = (
        directory
        + "output/earthquake_source_table_complete_"
        + str(process_year)
        + ".csv"
    )
    props_out_file = (
        directory
        + "output/propagation_path_table_complete_"
        + str(process_year)
        + ".csv"
    )
    #     event_out_file = directory+'relocated_output/earthquake_source_table_complete_prop_'+str(process_year)+'.csv'
    #     props_out_file = directory+'relocated_output/propagation_path_table_complete_prop_'+str(process_year)+'.csv'
    #     process_month = month
    # 	process_year = [2003] ### Assign a list of years to download waveforms for
    event_sub_mask = event_df.datetime.dt.year == process_year
    #     event_sub_mask = event_df.datetime.dt.year == process_year
    # 		geonet_sub_mask = np.isin(geonet.origintime.values.astype('datetime64[Y]').astype(int)+1970,process_year)
    event_sub = event_df[event_sub_mask].reset_index(drop=True)

    if len(event_sub) > 0:
        event_list = event_sub.evid.unique()
        #         geonet_cmt_df_sub = geonet_cmt_df[geonet_cmt_df['PublicID'].isin(event_list)]
        #         cmt_df_sub = cmt_df[cmt_df['PublicID'].isin(event_list)]
        df_merged_sub = df_merged[
            df_merged["evid"].isin(event_list)
        ]  # Subsetting the data makes the program run significantly faster

        # When using geonet derived data
        events_out = pd.DataFrame(
            columns=[
                "evid",
                "datetime",
                "lat",
                "lon",
                "depth",
                "loc_type",
                "loc_grid",
                "mag",
                "mag_type",
                "mag_method",
                "mag_unc",
                "mag_orig",
                "mag_orig_type",
                "mag_orig_unc",
                "ndef",
                "nsta",
                "nmag",
                "t_res",
                "reloc",
                "tect_class",
                "tect_method",
                "domain_no",
                "domain_type",
                "strike",
                "dip",
                "rake",
                "f_length",
                "f_width",
                "f_type",
                "z_tor",
                "z_bor",
            ]
        )
        # When using data with MAXI locations, note that 'minimum' is 't_res'
        #         events_out = pd.DataFrame(columns=['evid', 'datetime', 'lat', 'lon', 'depth',
        #                'mag', 'mag_type', 'mag_method', 'mag_unc', 'mag_orig', 'mag_orig_type', 'mag_orig_unc', 'ndef', 'nsta', 'nmag',
        #                'reloc', 'minimum', 'finalgrid', 'x', 'y', 'z', 'x_c', 'y_c', 'z_c',
        #                'major', 'minor', 'z_err', 'theta', 'Q', 'loc_type', 'loc_grid', 'tect_class', 'tect_method', 'domain_no',
        #                'domain_type', 'strike', 'dip', 'rake', 'f_length', 'f_width', 'f_type',
        #                'z_tor', 'z_bor'])
        props_out = pd.DataFrame(
            columns=[
                "evid",
                "net",
                "sta",
                "r_epi",
                "r_hyp",
                "r_jb",
                "r_rup",
                "r_x",
                "r_y",
                "r_tvz",
                "r_xvf",
                "az",
                "b_az",
                "reloc",
                "f_type",
            ]
        )
        events_out.to_csv(event_out_file, index=False)
        props_out.to_csv(props_out_file, index=False)

        pandarallel.initialize(
            nb_workers=6, progress_bar=True
        )  # Customize the number of parallel workers
        event_sub.parallel_apply(
            lambda x: path_props(
                x,
                df_merged_sub,
                df_station,
                taupo_polygon,
                cmt_df,
                domain_focal_df,
                srf_files,
                srfs,
                geonet_cmt_df,
                event_out_file,
                props_out_file,
            ),
            axis=1,
        )
        print("")
        print("Completed processing for year " + str(process_year))
#

events_out_final = pd.concat(
    [
        pd.read_csv(f, low_memory=False)
        for f in glob.glob(directory + "output/earthquake_source_table_complete_*.csv")
    ]
)
props_out_final = pd.concat(
    [
        pd.read_csv(f, low_memory=False)
        for f in glob.glob(directory + "output/propagation_path_table_complete_*.csv")
    ]
)
# events_out_final = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'relocated_output/earthquake_source_table_complete_prop_*.csv')])
# props_out_final = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'relocated_output/propagation_path_table_complete_prop_*.csv')])
#
events_out_final.to_csv(
    directory + "output/earthquake_source_table_complete.csv", index=False
)
props_out_final.to_csv(
    directory + "output/propagation_path_table_complete.csv", index=False
)
