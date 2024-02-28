# Scans through waveforms linked to the rotd50 flat file and determines if they are likely
# to have clipping.

import pandas as pd
import obspy as op
from obspy.clients.fdsn import Client as FDSN_Client
from gmprocess.waveform_processing.clipping import clipping_check
import numpy as np
import glob
import os
from pandarallel import pandarallel

from obspy.geodetics.base import gps2dist_azimuth

from gmprocess.waveform_processing.clipping.clipping_ann import clipNet
from gmprocess.waveform_processing.clipping.max_amp import Max_Amp
from gmprocess.waveform_processing.clipping.histogram import Histogram
from gmprocess.waveform_processing.clipping.ping import Ping
from gmprocess.waveform_processing.processing_step import ProcessingStep

M_TO_KM = 1.0 / 1000


def return_clipped(row, threshold):
    gm = row.copy()
    try:
        sta = station_df[station_df.sta == gm.sta].iloc[0]
        event = eq_df[eq_df.evid == gm.evid].iloc[0]
        #     prop = prop_df[(prop_df.evid == gm.evid) & (prop_df.sta == gm.sta)].iloc[0]
        sta_lat = sta.lat
        sta_lon = sta.lon
        event_lat = event.lat
        event_lon = event.lon
        event_mag = event.mag
        #     dist = prop.r_hyp
        search = search_df[search_df.evid == gm.evid]
        if len(search) > 1:
            if sta.sta != "CPLB":
                search = search.iloc[1]
            else:
                search = search.iloc[0]
        else:
            search = search.iloc[0]
        mseed_file = glob.glob(search.mseed_dir + "/*" + gm.sta + "*.mseed")[0]
        st = op.read(mseed_file)

        dist = (
            gps2dist_azimuth(
                lat1=event_lat,
                lon1=event_lon,
                lat2=sta_lat,
                lon2=sta_lon,
            )[0]
            * M_TO_KM
        )

        event_mag = np.clip(event_mag, 3.0, 8.8)
        dist = np.clip(dist, 0.0, 645.0)

        clip_nnet = clipNet()

        max_amp_method = Max_Amp(st, max_amp_thresh=6e6)
        hist_method = Histogram(st)
        ping_method = Ping(st)

        inputs = [
            event_mag,
            dist,
            max_amp_method.is_clipped,
            hist_method.is_clipped,
            ping_method.is_clipped,
        ]
        prob_clip = clip_nnet.evaluate(inputs)[0][0]
        gm["clip_prob"] = prob_clip
        if prob_clip >= threshold:
            gm["clipped"] = True
        else:
            gm["clipped"] = False
    except:
        gm["clip_prob"] = None
        gm["clipped"] = None
    return gm


### Directory that you want to write CSVs to.
directory = "../output/"

# Threshold with which to consider data clipped
threshold = 0.2
root_dir = "/Volumes/SeaJade 2 Backup/NZ"
search_dirs = [
    "mseed_4-10_2022_preferred",
    "CPLB_mseed_4-10_preferred",
    "mseed_4-4.5_preferred",
    "mseed_4.5-5_preferred",
    "mseed_5-6_preferred",
    "mseed_6-10_preferred",
]

xml_files = []
evids = []
mseed_dirs = []

for search_dir in search_dirs:
    search_dir = root_dir + "/" + search_dir
    xml_files = xml_files + (glob.glob(search_dir + "/**/*.xml", recursive=True))

evids = [os.path.basename(xml_file).split(".")[0] for xml_file in xml_files]
mseed_dirs = [os.path.dirname(xml_file) + "/mseed/data" for xml_file in xml_files]

zipped = list(zip(evids, xml_files, mseed_dirs))
search_df = pd.DataFrame(zipped, columns=["evid", "xml_file", "mseed_dir"])

df = pd.read_csv(
    directory + "IM_catalogue/complete_ground_motion_im_catalogue_final.csv",
    low_memory=False,
)
df = df[df.component == "rotd50"].reset_index(drop=True)
df["datetime"] = pd.to_datetime(df["datetime"])
eq_df = pd.read_csv(
    directory + "earthquake_source_table_complete.csv", low_memory=False
)

client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client("IRIS")
inventory_NZ = client_NZ.get_stations()
inventory_IU = client_IU.get_stations(
    network="IU", station="SNZO,AFI,CTAO,RAO,FUNA,HNR,PMG"
)
inventory = inventory_NZ + inventory_IU
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
station_df = pd.DataFrame(station_info, columns=["net", "sta", "lat", "lon", "elev"])
station_df = station_df.drop_duplicates().reset_index(drop=True)

pandarallel.initialize(nb_workers=8, progress_bar=True)

### Enter years and months that you are interested in processing data for. Note that one
### year of data can take up to several days to process (in the case of 2016).
date_range = pd.date_range(start="2000-01", end="2023-01", freq="MS")
for idx, date in enumerate(date_range[0:-1]):
    year = date.year
    month = date.month
    df_sub_mask = (df.datetime >= date) & (df.datetime < date_range[idx + 1])
    df_sub = df[df_sub_mask].reset_index(drop=True)

    df_out = df_sub.parallel_apply(lambda x: return_clipped(x, threshold), axis=1)
    df_out = df_out[["evid", "sta", "clip_prob", "clipped"]]
    df_out.to_csv(
        directory + "clip_table_" + str(year) + "_" + str(month) + ".csv", index=False
    )

df_out = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in glob.glob(directory + "clip_table*.csv")]
)
df_out.to_csv(directory + "IM_catalogue/Tables/clip_table.csv", index=False)
