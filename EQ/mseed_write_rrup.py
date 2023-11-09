# Reads a list of catalogue of GeoNet events and downloads the mseed data compatible
# with instrument response to an mseed directory, shown as 'root' in the mseed_write
# function. This data is used as input for the ground motion classifier, which is in
# turn used to determine which events to comput intensity measures for.

# This version contains optimizations using list comprehension and other factors that allow
# it to run 3 times faster than ver. 1.

# This version incorporates magnitude-distance scaling when searching for stations to
# download waveform data from. It is also uses a ds595 calculation in order to determine
# the full waveform time window to download.

# Updated to use Dask instead of Ray or Pandarallel

import glob
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.core.event import read_events
from obspy import read_inventory
from obspy.signal import freqattributes,rotate
from obspy.geodetics import locations2degrees,degrees2kilometers,kilometers2degrees
from obspy.geodetics.base import gps2dist_azimuth
from obspy.core import UTCDateTime
from obspy.taup import TauPyModel
from multiprocessing import Pool,cpu_count
from scipy.interpolate import interp1d
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError
# from progress.bar import ChargingBar
import numpy as np
import pandas as pd
import os
import sys
import typing
import time
import numpy as np
import glob
# Dependency that needs to be executed from the same folder as this program.
import AfshariStewart_2016_Ds as asds

class Site:  # Class of site properties. initialize all attributes to None
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")  # station name
        self.Rrup = kwargs.get("rrup")  # closest distance coseismic rupture (km)
        self.Rjb = kwargs.get(
            "rjb"
        )  # closest horizontal distance coseismic rupture (km)
        self.Rx = kwargs.get(
            "rx", -1.0
        )  # distance measured perpendicular to fault strike from surface projection of
        #                       # updip edge of the fault rupture (+ve in downdip dir) (km)
        self.Ry = kwargs.get(
            "ry", -1.0
        )  # horizontal distance off the end of the rupture measured parallel
        self.Rtvz = kwargs.get(
            "rtvz"
        )  # source-to-site distance in the Taupo volcanic zone (TVZ) (km)
        self.vs30measured = kwargs.get(
            "vs30measured", False
        )  # yes =True (i.e. from Vs tests); no=False (i.e. estimated from geology)
        self.vs30 = kwargs.get("vs30")  # shear wave velocity at 30m depth (m/s)
        self.z1p0 = kwargs.get(
            "z1p0"
        )  # depth (km) to the 1.0km/s shear wave velocity horizon (optional, uses default relationship otherwise)
        self.z1p5 = kwargs.get("z1p5")  # (km)
        self.z2p5 = kwargs.get("z2p5")  # (km)
        self.siteclass = kwargs.get("siteclass")
        self.orientation = kwargs.get("orientation", "average")
        self.backarc = kwargs.get(
            "backarc", False
        )  # forearc/unknown = False, backarc = True
        self.fpeak = kwargs.get("fpeak", 0)


class Fault:  # Class of fault properties. initialize all attributes to None
    def __init__(self, **kwargs):
        self.dip = kwargs.get("dip")  # dip angle (degrees)
        self.faultstyle = kwargs.get(
            "faultstyle"
        )  # Faultstyle (options described in enum below)
        self.hdepth = kwargs.get("hdepth")  # hypocentre depth
        self.Mw = kwargs.get("Mw")  # moment tensor magnitude
        self.rake = kwargs.get("rake")  # rake angle (degrees)
        self.tect_type = kwargs.get(
            "tect_type"
        )  # tectonic type of the rupture (options described in the enum below)
        self.width = kwargs.get("width")  # down-dip width of the fault rupture plane
        self.zbot = kwargs.get("zbot")  # depth to the bottom of the seismogenic crust
        self.ztor = kwargs.get("ztor")  # depth to top of coseismic rupture (km)

def estimate_z1p0(vs30):
    return (
        np.exp(28.5 - 3.82 / 8.0 * np.log(vs30 ** 8 + 378.7 ** 8)) / 1000.0
    )  # CY08 estimate in KM

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def mseed_write(row,root,min_mag,max_mag,pad_time):

	vs30_default = 500

	channel_codes = 'HN?,BN?,EN?,HH?,BH?,EH?,SH?'

	siteprop = Site()
	faultprop = Fault()
	
	eventid = row.publicid
	print(eventid)
	try:
		# Check to see if the event is in the FDSN database, skip if it isn't!
		cat = client_NZ.get_events(eventid=eventid)
	except:
		return

	event = cat[0]
	
# 		if row.eventtype == 'outside of network interest':
# 			print('Event '+eventid+' outside of network interest')

	try:
		otime = event.preferred_origin().time
		event_lat = event.preferred_origin().latitude
		event_lon = event.preferred_origin().longitude
		event_depth = event.preferred_origin().depth/1000 # Depth in km
	
		folderA = otime.strftime('%Y')
		folderB = otime.strftime('%m_%b')
		folderC = otime.strftime('%Y-%m-%d_%H%M%S')
		directory = root+'/'+folderA+'/'+folderB+'/'+folderC
		seed_directory = root+'/'+folderA+'/'+folderB+'/'+folderC+'/mseed/data'
		filename = otime.strftime(str(eventid)+'.xml')
		fw = directory+'/'+filename
		if not os.path.exists(directory):
			os.makedirs(directory)
			print('Event',eventid,'created')
		else: # Skip events where the directory already exists
			print('Event',eventid,'exists...')
	# 		if p_bar:
	# 			progress(ii, total, status='')
	# 			ii += 1
			return
		event.write(fw,format='QUAKEML')
	
		preferred_magnitude = event.preferred_magnitude().mag
		# Add magnitude data to fault properties. Could add rupture type here in the
		# future, if possible and/or necessary.
		faultprop.Mw = preferred_magnitude
		if preferred_magnitude < mws.min():
			rrup = np.array(rrups.min())
		elif preferred_magnitude > mws.max():
			rrup = np.array(rrups.max())
		else:
			rrup = f_rrup(preferred_magnitude)
		maxradius = kilometers2degrees(rrup)
		inv_sub = inventory.select(latitude=event_lat,longitude=event_lon,maxradius=maxradius)
		inv_sub_sta = []
		for network in inv_sub:
			net = network.code
			for station in network:
				sta = station.code
				inv_sub_sta.append([net,sta])
		p_stations = []
		s_stations = []
		for pick in event.picks: 
			if pick.phase_hint[0].lower() == 'p':
				p_stations.append([pick.waveform_id.network_code,pick.waveform_id.station_code])
			if pick.phase_hint[0].lower() == 's':
				s_stations.append([pick.waveform_id.network_code,pick.waveform_id.station_code])
		# Build complete station list, which is based on the combined stations within 
		# maxradius and those selected for P and S arrivals.
		sta_df = pd.DataFrame(inv_sub_sta+p_stations+s_stations,columns=['net','sta']).drop_duplicates()
	
		# Check for existing files
		file_list = glob.glob(seed_directory+'/*.mseed')
		sta_written = [file.split('_')[-3] for file in file_list]
	
		for sta_index,sta_row in sta_df.iterrows():
			net = sta_row.net
			sta = sta_row.sta
			# Skip files that already exist
	# 			if np.isin(sta,sta_written):
	# 				continue
			inv = inventory.select(station=sta)
			if len(inv) == 0:
				continue
			# Add site properties, uses default vs30 value of 500 for now, could implement
			# site specific data in the future, if necessary.
			siteprop.vs30 = vs30_default
			siteprop.z1p0 = estimate_z1p0(siteprop.vs30)
			sta_lat = inv[0][0].latitude
			sta_lon = inv[0][0].longitude
			sta_el = inv[0][0].elevation/1000
			dist, az, b_az = gps2dist_azimuth(event_lat, event_lon, sta_lat, sta_lon)
			r_epi = dist/1000
			deg = kilometers2degrees(r_epi)
			r_hyp = (r_epi ** 2 + (event_depth + sta_el) ** 2) ** 0.5
			siteprop.Rrup = r_hyp
		
			# Estimate arrival times for P and S phases
			p_arrivals = model.get_travel_times(source_depth_in_km=event_depth,distance_in_degree=deg,phase_list=['ttp'])
			s_arrivals = model.get_travel_times(source_depth_in_km=event_depth,distance_in_degree=deg,phase_list=['tts'])
			ptime_est = otime + p_arrivals[0].time # Estimated earliest P arrival time from taup
			stime_est = otime + s_arrivals[0].time
		
			# Predict significant duration time from Afshari and Stewart (2016)
			ds, ds_std = asds.Afshari_Stewart_2016_Ds(siteprop, faultprop, 'Ds595')
				
			try:
				# Check network
				if net == 'IU':
					time.sleep(np.random.choice(range(1,7))) # IRIS seems to disconnect if there are too many rapid connections
					min_ttdiff = 15
					if (stime_est + ds * 1.2 - ptime_est) < min_ttdiff:
						st = client_IU.get_waveforms(net, sta, '*', channel_codes, ptime_est-15, ptime_est + min_ttdiff + pad_time, attach_response=True)
					else:
						st = client_IU.get_waveforms(net, sta, '*', channel_codes, ptime_est-15, stime_est + ds * 1.2 + pad_time, attach_response=True)
	# 				st = client_IU.get_waveforms(net, sta, '*', channel_codes, ptime_est-2, stime_est + ds * 1.2 + 15, attach_response=True)
					if not st[0].stats.response:
						continue
				elif net == 'NZ':
					no_response = False
					# Get one second of data and determine the minimum waveform length (needs to be 2048 samples). Commented out
					# because this is no longer true with the new version of the GMC.
					min_ttdiff = 15
					if (stime_est + ds * 1.2 - ptime_est) < min_ttdiff:
						st = client_NZ.get_waveforms(net, sta, '*', channel_codes, ptime_est-15, ptime_est + min_ttdiff + pad_time)
					else:
						st = client_NZ.get_waveforms(net, sta, '*', channel_codes, ptime_est-15, stime_est + ds * 1.2 + pad_time)
				# Get list of locations and channels
				loc_chans = np.unique(np.array([[tr.stats.location,tr.stats.channel[0:2]+'?'] for tr,tr in zip(st,st)]),axis=0)
				for loc_chan in loc_chans:
					loc, chan = loc_chan
					st_new = st.select(location=loc,channel=chan)
					chan_check = len(inv.select(location=loc,channel=chan,time=otime))
					filename = otime.strftime('%Y%m%d_%H%M%S_'+sta+'_'+loc+'_'+chan[0:2]+'.mseed')
					fw = seed_directory+'/'+filename
					if chan_check > 0:
						no_response = False
					else:
						no_response = True
					if no_response == True:
						continue
					if len(st_new) > 3:
						try:
							sample_rates = [tr.stats.sampling_rate for tr in st_new]
							max_samples = np.max(sample_rates)
							st_new = st_new.select(sampling_rate=max_samples)
							st_new.merge(fill_value = 'interpolate')
	# 								print(len(st_new))
	# 							print()
	# 							print('Merged data for ',st_new[0].stats.station,chan,loc,otime)
						except:
	# 							print()
	# 							print('Cannot merge data for ',st_new[0].stats.station,chan,loc,otime)
							continue
	# 						sta = st_new[0].stats.station
					if not os.path.exists(seed_directory):
						os.makedirs(seed_directory)
	# 					st_new.plot(outfile=seed_directory+'/'+sta+'_'+chan+'_'+str(ds)+'_'+str(r_hyp)+'.png')
					# Ensure traces all have the same length
					starttime_trim = np.max([tr.stats.starttime for tr in st_new])
					endtime_trim = np.min([tr.stats.endtime for tr in st_new])
					st_new.trim(starttime_trim,endtime_trim)
					st_new.write(fw,format='MSEED')				
			except FDSNNoDataException:
	# 				print('No data',sta)
				continue
			except ObsPyMSEEDFilesizeTooSmallError:
	# 				print('File size too small',sta)
				continue
			except Exception as e:
	# 				print()
	# 				print(e.__class__, "occurred for event "+str(eventid)+' and station '+str(sta))
				continue
	except:
		print('Something bad happened with event '+str(eventid))
# 		if p_bar:
# 			progress(ii, total, status='')
# 			ii += 1

### Parameters for the user to adjust ###

# GeoNet earthquake file
filename = 'earthquakes.csv'
# Minimum magnitude to filter on
min_mag = 4
# Maximum magnitude to filter on
max_mag = 10
# Channel codes with which to filter the station data
channel_codes = 'HN?,BN?,EN?,HH?,BH?,EH?,SH?'

# Root output file directory, uses min and max magnitudes to determine the root file
# name
root = '/Volumes/SeaJade 2 Backup/NZ/mseed_'+str(min_mag)+'-'+str(max_mag)+'_2022'

pad_time = 0 # Time to pad onto the end of the miniseed stream (in seconds)

# TauPy is used to estimate the arrival time for events at stations, here we use a general
# global model, iasp91
model = TauPyModel(model="iasp91")

# Generate cubic interpolation for magnitude distance relationship
mw_rrup = np.loadtxt('Mw_rrup.txt')
mws = mw_rrup[:,0]
rrups = mw_rrup[:,1]
f_rrup = interp1d(mws,rrups,kind='cubic')

# file_list = np.load('eq_paths.npy')
# inventory = read_inventory('/Volumes/SeaJade2/NZ/NZ_EQ_Catalog/nz_stations.xml')
client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')
inventory_NZ = client_NZ.get_stations(channel=channel_codes,level='response')
inventory_IU = client_IU.get_stations(network='IU',station='SNZO',channel=channel_codes,level='response')
inventory = inventory_NZ+inventory_IU

geonet = pd.read_csv(filename,low_memory=False)
geonet = geonet.sort_values('origintime')
geonet['origintime'] = pd.to_datetime(geonet['origintime'],format='%Y-%m-%dT%H:%M:%S.%fZ')
geonet = geonet.reset_index(drop=True)

# process_years = np.arange(2017,2019) ### Assign a list of years to download waveforms for
# for year in process_years:
# 	print('Processing '+str(year))
# 	print()
# 	geonet_sub_mask = np.isin(geonet.origintime.values.astype('datetime64[Y]').astype(int)+1970,year)

min_date = np.datetime64('2022-01-01').astype('datetime64[D]').astype(int)+1970
max_date = np.datetime64('2023-01-01').astype('datetime64[D]').astype(int)+1970
# date = np.datetime64('2004').astype('datetime64[Y]').astype(int)+1970
geonet_sub_mask = (geonet.origintime.values.astype('datetime64[D]').astype(int)+1970 >= min_date) & (geonet.origintime.values.astype('datetime64[D]').astype(int)+1970 < max_date)
geonet_sub = geonet[geonet_sub_mask].reset_index(drop=True)
geonet_sub = geonet_sub[(geonet_sub.magnitude >= min_mag) & (geonet_sub.magnitude < max_mag)].reset_index(drop=True) ### Start with M >= 4. Go back for smaller events later
# geonet_sub = geonet_sub[geonet_sub.eventtype != 'outside of network interest'].reset_index(drop=True) # Remove non-NZ events
# geonet_sub = geonet_sub[geonet_sub.magnitude >= 6].reset_index(drop=True)
# geonet_sub = geonet_sub[(geonet_sub.magnitude < 5) & (geonet_sub.magnitude >= 4.5)].reset_index(drop=True) ### Start with M >= 4. Go back for smaller events later

# kaikoura_id = '2016p858000'
# event = geonet[geonet.publicid == '3366146'] # Darfield
# event = geonet[geonet.publicid == '3468575'] # Christchurch
# event = geonet[geonet.publicid == '3124785'] # Dusky Sound
# event = geonet[geonet.publicid == '2016p858803'].iloc[0]
# event = geonet[geonet.publicid == '2013p543824'].iloc[0]
# mseed_write(event)

start_time = time.time()
import dask
delayed_results = []
for idx, row in geonet_sub.iterrows():
	result = dask.delayed(mseed_write)(row,root,min_mag,max_mag,pad_time)
	delayed_results.append(result)
dask.compute(*delayed_results)
end_time = time.time()-start_time
print('Took '+str(end_time)+' seconds to run')