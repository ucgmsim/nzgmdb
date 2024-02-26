# Added SNR, whether the data was filtered for a better SNR, and amplitude time to
# the magnitude information.

import pandas as pd
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import read_inventory
import obspy as op
from multiprocessing import Pool
import multiprocessing
import glob
from obspy.signal.invsim import paz_2_amplitude_value_of_freq_resp
from obspy.signal.invsim import simulate_seismometer as seis_sim
from obspy import Trace
import copy
from scipy.signal import iirfilter, sosfreqz
import os
import warnings
import math
# import ray
import dask
import time as timeit

def glitch_catcher(
    tr: Trace, 
    window_length: float, 
    threshold: float
):
    """
    Catch glitches - returns True if a glitch is found.

    Parameters
    ---
    tr:
        Trace to look for glitches in
    window_length:
        Window length to compute average differences over in seconds
    threshold:
        Threshold multiplier of average differences to declare a glitch
    """

    window_length = int(window_length * tr.stats.sampling_rate)
    # Using a window length to try and avoid removing earthquake signals...

    diffs = np.abs(tr.data[0:-1] - tr.data[1:])
    diff_moving = np.cumsum(diffs)
    diff_moving = diff_moving[window_length:] - diff_moving[0:-window_length:]
    diff_moving  = diff_moving / window_length
    # Extend diff_moving to the same length as diffs - not a great way to do this!
    diff_moving = np.concatenate(
        [np.array([diff_moving.mean()] * (len(diffs) - len(diff_moving))), 
         diff_moving])
    if np.any(diffs > diff_moving * threshold):
        print(f"Found large differences at {np.where(diffs > diff_moving * threshold)}")
        return True
    return False

def _rms(array):
    """
    Calculate RMS of array.

    :type array: numpy.ndarray
    :param array: Array to calculate the RMS for.

    :returns: RMS of array
    :rtype: float
    """
    return np.sqrt(np.mean(np.square(array)))

def _snr(tr, noise_window, signal_window):
    """
    Compute ratio of maximum signal amplitude to rms noise amplitude.

    :param tr: Trace to compute signal-to-noise ratio for
    :param noise_window: (start, end) of window to use for noise
    :param signal_window: (start, end) of window to use for signal

    :return: Signal-to-noise ratio, noise amplitude
    """
    noise_amp = _rms(
        tr.slice(starttime=noise_window[0], endtime=noise_window[1]).data)
    if np.isnan(noise_amp):
#         Logger.warning("Could not calculate noise with this data, setting "
#                        "to 1")
        noise_amp = 1.0
    try:
        signal_amp = tr.slice(
            starttime=signal_window[0], endtime=signal_window[1]).data.max()
    except ValueError as e:
#         Logger.error(e)
        return np.nan
    return signal_amp / noise_amp

def _max_p2t(data, delta, return_peak_trough=False):
    """
    Finds the maximum peak-to-trough amplitude and period.

    Originally designed to be used to calculate magnitudes (by
    taking half of the peak-to-trough amplitude as the peak amplitude).

    :type data: numpy.ndarray
    :param data: waveform trace to find the peak-to-trough in.
    :type delta: float
    :param delta: Sampling interval in seconds
    :type return_peak_trough: bool
    :param return_peak_trough:
        Optionally return the peak and trough

    :returns:
        tuple of (amplitude, period, time) with amplitude in the same
        scale as given in the input data, and period in seconds, and time in
        seconds from the start of the data window.
    :rtype: tuple
    """
    turning_points = []  # A list of tuples of (amplitude, sample)
    for i in range(1, len(data) - 1):
        if (data[i] < data[i - 1] and data[i] < data[i + 1]) or\
           (data[i] > data[i - 1] and data[i] > data[i + 1]):
            turning_points.append((data[i], i))
    if len(turning_points) >= 1:
        amplitudes = np.empty([len(turning_points) - 1],)
        half_periods = np.empty([len(turning_points) - 1],)
    else:
        print(
            'Turning points has length: ' + str(len(turning_points)) +
            ' data have length: ' + str(len(data)))
        return 0.0, 0.0, 0.0
    for i in range(1, len(turning_points)):
        half_periods[i - 1] = (delta * (turning_points[i][1] -
                                        turning_points[i - 1][1]))
        amplitudes[i - 1] = np.abs(turning_points[i][0] -
                                   turning_points[i - 1][0])
    amplitude = np.max(amplitudes)
    period = 2 * half_periods[np.argmax(amplitudes)]
    delay = delta * turning_points[np.argmax(amplitudes)][1]
    if not return_peak_trough:
        return amplitude, period, delay
    max_position = np.argmax(amplitudes)
    peak = max(
        t[0] for t in turning_points[max_position: max_position + 2])
    trough = min(
        t[0] for t in turning_points[max_position: max_position + 2])
    return amplitude, period, delay, peak, trough

def sim_wa(inventory, trace):

    trace.remove_response(inventory=inventory, output='VEL', water_level=1)
    
    PAZ_WA = {'poles': [-6.2832 + 4.7124j, -6.2832 - 4.7124j],
              'zeros': [0 + 0j], 'gain': 1, 'sensitivity': 2800}
    
    trace.data = seis_sim(trace.data, trace.stats.sampling_rate,
                               paz_remove=None, paz_simulate=PAZ_WA,
                               water_level=1)
    
    trace.data = trace.data * 1000
    
   
    return trace

def _sim_WA(trace, inventory, water_level, velocity=False):
    """
    Remove the instrument response from a trace and simulate a Wood-Anderson.

    Returns a de-meaned, de-trended, Wood Anderson simulated trace in
    its place.

    Works in-place on data and will destroy your original data, copy the
    trace before giving it to this function!

    :type trace: obspy.core.trace.Trace
    :param trace:
        A standard obspy trace, generally should be given without
        pre-filtering, if given with pre-filtering for use with
        amplitude determination for magnitudes you will need to
        worry about how you cope with the response of this filter
        yourself.
    :type inventory: obspy.core.inventory.Inventory
    :param inventory:
        Inventory containing response information for the stations in st.
    :type water_level: float
    :param water_level: Water level for the simulation.
    :type velocity: bool
    :param velocity:
        Whether to return a velocity trace or not - velocity is non-standard
        for Wood-Anderson instruments, but institutes that use seiscomp3 or
        Antelope require picks in velocity.

    :returns: Trace of Wood-Anderson simulated data
    :rtype: :class:`obspy.core.trace.Trace`
    """
	# Helpers for local magnitude estimation
	# Note Wood anderson sensitivity is 2080 as per Uhrhammer & Collins 1990
    PAZ_WA = {'poles': [-6.28318 + 4.71239j, -6.28318 - 4.71239j],
        'zeros': [0 + 0j], 'gain': 1.0, 'sensitivity': 2800}
#     PAZ_WA = {'poles': [-5.49779 + 5.60886j, -5.49779 - 5.60886j],
#         'zeros': [0 + 0j], 'gain': 1.0, 'sensitivity': 2080}
    assert isinstance(trace, Trace)
    paz_wa = copy.deepcopy(PAZ_WA)
    # Need to make a copy because we might edit it.
    if velocity:
        paz_wa['zeros'] = [0 + 0j, 0 + 0j]
    # De-trend data
#     trace.detrend('simple')
#     # Remove response to Velocity
    try:
#         trace.remove_response(
#             inventory=inventory, output="VEL", water_level=1)
		# Try doing this without the whole water level thing
        trace.remove_sensitivity(inventory=inventory)
        if trace.stats.channel[1] == 'N':
        	trace.integrate()
    except Exception:
        print(f"No response for {trace.id} at {trace.stats.starttime}")
        return None
    # Simulate Wood Anderson
    trace.data = seis_sim(trace.data, trace.stats.sampling_rate,
                          paz_remove=None, paz_simulate=paz_wa,
                          water_level=water_level)
    return trace

# @ray.remote
def convert_eq(eventid,client_NZ,client_IU,station_df,sta_corr,event_df_file,
			arrival_df_file,mag_df_file):
	"""
	Downloads waveforms for an event from the GEONET catalogue and computes the magnitude
	using the equations of Rhoades et al. (2020), also referred to as MLNZ20
	:type eventid: str
	:param eventid:
		The event identification number from the GEONET database. Usually is listed as
		the PublicID in the CSVs downloaded directly from their webservice.
	:type client_NZ: obspy.Client
	:param client_NZ: GEONET FDSN webservice client.
	:type client_IU: obspy.Client
	:param client_IU: IRIS FDSN webservice client (necessary for station SNZO).
	:type station_df: pandas.DataFrame
	:param station_df: DataFrame with station information
	:type sta_corr: pandas.DataFrame
	:param sta_corr: DataFrame with station magnitude corrections
	:type event_df_file: str
	:param event_df_file: Event CSV file name
	:type arrival_df_file: str
	:param arrival_df_file: Arrival CSV file name
	:type mag_df_file: str
	:param mag_df_file: Station magnitude CSV file name


	:returns: None, the program writes directly to files.
	:rtype: None
	"""

	highcut = 20
	lowcut = 1
	corners = 4
	velocity = False

	sorter = ['HH','BH','EH','SH','HN','BN']
	channel_filter = 'HH?,BH?,EH?,SH?,HN?,BN?'
	nz20_res = 0.278 # Uncertainty of the Rhoades et al. (2020) method.
	reloc = 'no' # Indicates if an earthquake has been relocated, default to 'no'.
	eventid = str(eventid)
	try:
		print('Processing event',eventid)
		cat = client_NZ.get_events(eventid=eventid)

		event_line = []
		mag_line = []
		phase_line = []
		prop_line = []

		for event in cat:
			ev_datetime = event.preferred_origin().time
			ev_lat = event.preferred_origin().latitude
			ev_lon = event.preferred_origin().longitude
			ev_depth = event.preferred_origin().depth/1000
			try:
				ev_loc_type = str(event.preferred_origin().method_id).split('/')[1]
			except:
				ev_loc_type = None
			try:
				ev_loc_grid = str(event.preferred_origin().earth_model_id).split('/')[1]
			except:
				ev_loc_grid = None
			try:
				ev_ndef = event.preferred_origin().quality.used_phase_count
			except:
				ev_ndef = None
			try:
				ev_nsta = event.preferred_origin().quality.used_station_count
			except:
				ev_nsta = None
			try:
				std = event.preferred_origin().quality.standard_error
			except:
				std = None
	
			# Find magnitude info, deprioritize 'M' measurements
			pref_mag_type = event.preferred_magnitude().magnitude_type
			if pref_mag_type.lower() == 'm':
				pref_mag_type = 'ML'
				mb_mag = [mag for mag in event.magnitudes if mag.magnitude_type.lower() == 'mb']
				if mb_mag:
					mb_mag = mb_mag[np.array([mb_mag.station_count for mb_mag in mb_mag]).argmax()]
				ml_loc_mag = [mag for mag in event.magnitudes if mag.magnitude_type.lower() == 'ml']
				if ml_loc_mag:
					ml_loc_mag = ml_loc_mag[np.array([ml_loc_mag.station_count for ml_loc_mag in ml_loc_mag]).argmax()]
				mlv_loc_mag = [mag for mag in event.magnitudes if mag.magnitude_type.lower() == 'mlv']
				if mlv_loc_mag:
					mlv_loc_mag = mlv_loc_mag[np.array([mlv_loc_mag.station_count for mlv_loc_mag in mlv_loc_mag]).argmax()]
				if mb_mag:
					# For events with few Mb measures, perform some checks.
					if mb_mag.station_count < 3:
						loc_mag = []
						if ml_loc_mag and mlv_loc_mag:
							ml_loc_mag = ml_loc_mag
							mlv_loc_mag = mlv_loc_mag
							if ml_loc_mag.station_count >= mlv_loc_mag.station_count:
								loc_mag = ml_loc_mag
							else:
								loc_mag = mlv_loc_mag
						elif ml_loc_mag:
							loc_mag = ml_loc_mag
						elif mlv_loc_mag:
							loc_mag = mlv_loc_mag
						if len(loc_mag) > 0:
							if loc_mag.station_count > mb_mag.station_count:
								loc_mag = loc_mag
								print(eventid+ 'mb count: '+str(mb_mag.station_count)+' ml count: '+str(loc_mag.station_count))
						else:
							loc_mag = mb_mag
							pref_mag_type = 'Mb'
					else:
						loc_mag = mb_mag
						pref_mag_type = 'Mb'
				else:
					if ml_loc_mag and mlv_loc_mag:
						ml_loc_mag = ml_loc_mag
						mlv_loc_mag = mlv_loc_mag
						# Always prefer vertical over horizontal magnitudes
	# 						loc_mag = mlv_loc_mag
						# Always take value with higher station count
						if ml_loc_mag.station_count >= mlv_loc_mag.station_count:
							loc_mag = ml_loc_mag
						else:
							loc_mag = mlv_loc_mag
					elif ml_loc_mag:
						loc_mag = ml_loc_mag
					elif mlv_loc_mag:
						loc_mag = mlv_loc_mag
	# 					else:
	# 						print('...No ML for',eventid,'trying MLv')
	# 						loc_mag = [mag for mag in event.magnitudes if mag.magnitude_type.lower() == 'mlv']
	# 						if loc_mag:
	# 							loc_mag = loc_mag[0]
					else:
						print('...No ML or MLv for',eventid,'trying M')
						loc_mag = [mag for mag in event.magnitudes if mag.magnitude_type.lower() == 'm'][0]
				pref_mag = loc_mag.mag
				pref_mag_method = 'uncorrected'
				pref_mag_unc = loc_mag.mag_errors.uncertainty
				pref_mag_sta_ids = loc_mag.station_magnitude_contributions
				pref_mag_nmag = len(loc_mag.station_magnitude_contributions)
			else:
				pref_mag = event.preferred_magnitude().mag
				pref_mag_method = 'uncorrected'
				pref_mag_unc = event.preferred_magnitude().mag_errors.uncertainty
				pref_mag_nmag = len(event.preferred_magnitude().station_magnitude_contributions)
				pref_mag_sta_ids = event.preferred_magnitude().station_magnitude_contributions


			arrivals = event.preferred_origin().arrivals
			picks = event.picks
			pick_data = [[pick.waveform_id.network_code,
				pick.waveform_id.station_code,
				pick.waveform_id.location_code,
				pick.waveform_id.channel_code,
				pick.time,
				pick.phase_hint] for pick in picks]
			amplitudes = event.amplitudes
			pick_data.extend([[amp.waveform_id.network_code,
				amp.waveform_id.station_code,
				amp.waveform_id.location_code,
				amp.waveform_id.channel_code,
				amp.time_window.reference,
				'none'] for amp in amplitudes if not any(pick for pick in  pick_data if amp.waveform_id.station_code in pick)])
			pick_data = pd.DataFrame(pick_data,columns=['net','sta','loc','chan','time','phase_hint'])

			# Gather station magnitudes
			i = 1
			sta_mag_line = []
			for p_idx, pick in pick_data.iterrows():
				net = pick.net
				sta = pick.sta
				loc = pick['loc']
				chan = pick.chan
			
				if net == 'NZ':
					client = client_NZ
				else:
					client = client_IU
						
				ns = [net,sta]
				row_exists = [row[1:3] for row in sta_mag_line if row[1:3] == ns]
				if row_exists:
					continue

				inventory_st = []
				search_channel = True
				with warnings.catch_warnings():
					warnings.filterwarnings("ignore", category=UserWarning)
					try:
						inventory_st = client.get_stations(network=net, station=sta, channel=channel_filter, level='response',starttime=pick.time,endtime=pick.time)
						for channel_code in sorter:
							for c in inventory_st[0][0]:
								if c.code[0:2] == channel_code:
									loc = c.location_code
									chan = channel_code + chan[-1]
									search_channel = False
									break
							if search_channel == False:
								break
					except:
						pass
				if len(inventory_st) == 0:
					sta_mags = [sta_mag for sta_mag in event.station_magnitudes if 
						((sta_mag.waveform_id.network_code == net) &
						(sta_mag.waveform_id.station_code == sta))]
					if sta_mags:
						for sta_mag in sta_mags:
							sta_mag_mag = sta_mag.mag
							sta_mag_type = sta_mag.station_magnitude_type
							amp = [amp for amp in event.amplitudes if amp.resource_id == sta_mag.amplitude_id]
							if amp:
								amp = amp[0]
								amp_amp = amp.generic_amplitude
								amp_unit = amp.unit
								if 'unit' in amp:
									amp_unit = amp.unit
								else:
									amp_unit = None
							else:
								amp_amp = None
								amp_unit = None
							magid = eventid+str('m')+str(i)
							i=i+1
							mag_line.append([magid, net, sta, sta_mag.waveform_id.location_code, sta_mag.waveform_id.channel_code, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit,None,None,None])
							sta_mag_line.append([magid, net, sta, sta_mag.waveform_id.location_code, sta_mag.waveform_id.channel_code, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit,None,None,None])
					continue

				if pref_mag_type.lower() == 'ml' or pref_mag_type.lower() == 'mlv' or pref_mag_type is None:

					sta_info = station_df[station_df.sta == sta]

					if len(sta_corr[sta_corr.sta == sta]) == 1:
						corr = sta_corr[sta_corr.sta == sta]['corr'].values[0]
					else:
						corr = 0
				
					if len(sta_info) > 0:
						dist, az, b_az = op.geodetics.gps2dist_azimuth(ev_lat, ev_lon, sta_info['lat'].values[0], sta_info['lon'].values[0])
						r_epi = dist/1000
						r_hyp = (r_epi ** 2 + (ev_depth + sta_info['elev'].values[0]/1000) ** 2) ** 0.5
					else: # If the station is not in the current inventory
						arrival = [arrival for arrival in arrivals if arrival.pick_id == pick.resource_id][0]
						r_epi = op.geodetics.degrees2kilometers(arrival.distance)
						r_hyp = ((r_epi) ** 2 + (ev_depth) ** 2) ** 0.5

					slow_vel = 3
					endtime = ev_datetime + r_hyp/slow_vel + 30
			
					if pick.phase_hint.lower()[0] == 'p':
						windowStart = pick.time - 35
						windowEnd = endtime
						noise_window = [windowStart,windowStart+30]
						signal_window = [pick.time-2,pick.time+30]				
					else:
						windowStart = pick.time - 45
						windowEnd = endtime
						noise_window = [windowStart,windowStart+30]
						signal_window = [pick.time-12,pick.time+20]				

					if len(chan) < 3:
						try:
							st = client.get_waveforms(net,sta,loc,chan+'?',windowStart,windowEnd)
							st = st.merge()
						except:
							continue
					else:
						try:
							st = client.get_waveforms(net,sta,loc,chan[0:2]+'?',windowStart,windowEnd)
							st = st.merge()
						except:
							continue

					for tr in st:
						filt_data = False
					
						sta_mag = []
						for sta_mag_search in event.station_magnitudes:
							if sta_mag_search.waveform_id.station_code == sta:
								if len(sta_mag_search.waveform_id.channel_code) > 2:
									if sta_mag_search.waveform_id.channel_code[2] == tr.stats.channel[2]:
										sta_mag = sta_mag_search
										break
								elif tr.stats.channel[2] != 'Z': # For 2012 + data, horizontal channels are combined
	# 									if sta_mag_search.waveform_id.channel_code == tr.stats.channel[0:2]:
									sta_mag = sta_mag_search
									break
	# 						sta_mag = [sta_mag for sta_mag in event.station_magnitudes if ((sta_mag.waveform_id.station_code == sta) & (sta_mag.waveform_id.channel_code[2] == tr.stats.channel[2]))]
						if sta_mag:
	# 							sta_mag = sta_mag[0]
							sta_mag_mag = sta_mag.mag
							sta_mag_type = sta_mag.station_magnitude_type
							amp = [amp for amp in event.amplitudes if amp.resource_id == sta_mag.amplitude_id]
						else:
							sta_mag_mag = None
							sta_mag_type = pref_mag_type # Set to event preferred magnitude type
							amp = None
						if amp:
							amp = amp[0]
							amp_amp = amp.generic_amplitude
							if 'unit' in amp:
								amp_unit = amp.unit
							else:
								amp_unit = None
						else:
							amp_amp = None
							amp_unit = None
						tr = tr.copy()
						tr = tr.split().detrend('demean').merge(fill_value=0)[0]

						tr_starttime = tr.stats.starttime
						tr_endtime = tr.stats.endtime
	# 						tr.taper(0.05)
						tr.trim(tr.stats.starttime-5,tr.stats.endtime,pad=True,fill_value=tr.data[0])
						tr.trim(tr.stats.starttime,tr.stats.endtime+5,pad=True,fill_value=tr.data[-1])
	# 					
						# Test if waveform has no measure data
						if tr.max() == 0:
							if sta_mag:
								print(f'Waveform for {tr.id} is empty')
								magid = eventid+str('m')+str(i)
								i = i+1
								mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit,None,None,None])
								sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit,None,None,None])
							continue
					
						# Test if Nyquist frequency is lower than highpass filter
	# 						if tr.stats.sampling_rate/2 < lowcut:
	# 							print(f'Sampling rate for {tr.id} too low')
	# 							magid = eventid+str('m')+str(i)
	# 							i = i+1
	# 							mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None])
	# 							sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None])
	# 							continue
					
	# 						tr.filter('highpass', freq=lowcut,
	# 							corners = corners)
						# Will zero phase filtering make a difference?
	# 						tr.filter('highpass', freq=lowcut,
	# 							corners = corners / 2, zerophase = True)

						tr = _sim_WA(tr, inventory_st, 0, velocity=False)
			
						# Check if there is no data, or if the trace is all 0s
						if tr == None or tr.max() == 0:
							if sta_mag:
								magid = eventid+str('m')+str(i)
								i = i+1
								mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit,None,None,None])
								sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit,None,None,None])
							continue

					
						trim_tr = tr.copy().trim(tr_starttime,tr_endtime)
	
						# Test SNR in noise window, if available, else use alternative SNR calc
						if len(trim_tr.slice(starttime=noise_window[0], endtime=noise_window[1]).data) > 0:
							snr = _snr(trim_tr,noise_window,signal_window)
						else:
							snr = abs(trim_tr.max() / np.sqrt(np.mean(np.square(trim_tr.data))))
						if snr < 3:
							# Not a high enough SNR
							print(f'SNR for {tr.id} not high enough, attempting to filter data')
  
							tr.filter('highpass', freq=lowcut,
								corners = corners, zerophase=True)
							trim_tr = tr.copy().trim(tr_starttime,tr_endtime)
							if len(trim_tr.slice(starttime=noise_window[0], endtime=noise_window[1]).data) > 0:
								snr = _snr(trim_tr,noise_window,signal_window)
							else:
								snr = abs(trim_tr.max() / np.sqrt(np.mean(np.square(trim_tr.data))))
							if snr >= 3:
								filt_data = True
							else:
								print(f'SNR for {tr.id} is still not high enough')
								continue
						
						tr.trim(tr_starttime,tr_endtime)		

						# Calculate the normalized noise amplitude
						amplitude, period, delay, peak, trough = _max_p2t(
							tr.data, tr.stats.delta, return_peak_trough=True)
						amp_time = tr_starttime+delay
	# 						print(amplitude,peak,trough)
										
						if peak >= abs(trough):
							amplitude = peak
						else:
							amplitude = abs(trough)
	# 						
	# 						# Calculate the absolute amplitude
						max_amplitude = abs(tr.max())
				
			
						# Generate poles and zeros for the filter we used earlier.
						# We need to get the gain for the digital SOS filter used by
						# obspy.
						if filt_data:
							sos = iirfilter(
								corners * 2, [lowcut / (0.5 * tr.stats.sampling_rate)],
								btype='highpass', ftype='butter', output='sos')
							_, gain = sosfreqz(sos, worN=[1 / period],
											   fs=tr.stats.sampling_rate)
							gain = np.abs(gain[0])  # Convert from complex to real.
							if gain < 1e-2:
								print(
									f"Pick made outside stable pass-band of filter "
									f"on {tr.id}, rejecting")
								continue
							amplitude /= gain
							peak /= gain
							trough /= gain
							max_amplitude /= gain
				#             amplitude *= 0.5
						amplitude *= 1000
						peak *= 1000
						trough *= 1000
						max_amplitude *= 1000
	# 						print(amplitude,amp_amp,amp_time,windowStart+delay,r_hyp,sta,chan)
							
						if r_epi:
							amp_unit = 'mm'
						
							R = r_hyp
							h = ev_depth

	# 							logA0R = 1.110 * np.log10(R / 100) + 0.00189 * (R - 100) + 3.0  # L. K. Hutton and David M. Boore, BSSA, 1987: The Ml for southern California
	# 							logA0R = 0.2869 - 1.272*1e-3*(R) -(1.493 * np.log10(R))	+ corr	# Ristau 2016
	# 						A = 10 ** (sta_mag_mag - logA0R)

							A = amplitude
							ML_i = np.log10(A) + np.log10(R) + 0.0029 * R # Robinson, 1987, add K for station correction

							if (h <= 40):
								H40 = 0
							else:
								H40 = h - 40

							NZlogA0R = 0.51 + ((-0.79E-3) * R) + (-1.67 * np.log10(R)) + ((2.38E-3) * H40) + corr # Corrected Ml for NZ according to Rhoades et al., 2021
							CML_i = np.log10(A) - NZlogA0R
	# 							ML_i = np.log10(A) - logA0R

							magid = eventid+str('m')+str(i)
							i = i+1
							mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, CML_i, 'NZ20', A, peak, trough, max_amplitude,amp_unit,snr,filt_data,amp_time])
							sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid,  sta_mag_mag, sta_mag_type, CML_i, 'NZ20', A, peak, trough, max_amplitude,amp_unit,snr,filt_data,amp_time])
						else: # If there is no epicentral distance measured
							magid = eventid+str('m')+str(i)
							i = i+1
							mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit,None,None,None])
							sta_mag_line.append([magid, net, sta, loc, tr.stats.channel, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit,None,None,None])
				else: # If not local magnitude
					sta_mags = [sta_mag for sta_mag in event.station_magnitudes if 
						((sta_mag.waveform_id.network_code == net) &
						(sta_mag.waveform_id.station_code == sta))]
					if sta_mags:
						for sta_mag in sta_mags:
							sta_mag_mag = sta_mag.mag
							sta_mag_type = sta_mag.station_magnitude_type
							amp = [amp for amp in event.amplitudes if amp.resource_id == sta_mag.amplitude_id]
							if amp:
								amp = amp[0]
								amp_amp = amp.generic_amplitude
								amp_unit = amp.unit
								if 'unit' in amp:
									amp_unit = amp.unit
								else:
									amp_unit = None
							else:
								amp_amp = None
								amp_unit = None
							magid = eventid+str('m')+str(i)
							i=i+1
							mag_line.append([magid, net, sta, sta_mag.waveform_id.location_code, sta_mag.waveform_id.channel_code, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit,None,None,None])
							sta_mag_line.append([magid, net, sta, sta_mag.waveform_id.location_code, sta_mag.waveform_id.channel_code, eventid, sta_mag_mag, sta_mag_type, None, 'uncorrected', amp_amp, None, None, None,amp_unit,None,None,None])

			sta_mag_df = pd.DataFrame(sta_mag_line, columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 'amp_peak','amp_trough','amp_max','amp_unit','SNR','filtered','amp_time'])

			if pref_mag_type.lower() == 'ml' or pref_mag_type.lower() == 'mlv':
				### Require at least two stations for a preferred cml solution
				if len(sta_mag_df[sta_mag_df.mag_corr.isnull() == False].sta.unique()) > 1:
					### SeisComp3 takes a trimmed mean, rejecting the lowest and highest 12.5% ML values
					mag_data = sta_mag_df[sta_mag_df.chan.str.endswith('Z',na=False)]
					if (mag_data.mag_corr.isnull() == False).sum() < 2:
						mag_data = sta_mag_df[sta_mag_df.chan.str.endswith(('E','N','1','2'),na=False)]
						if len(mag_data[mag_data.mag_corr.isnull() == False].sta.unique()) < 2:
							mag_data = sta_mag_df
							mag_type = pref_mag_type
							mag_method = pref_mag_method
				#             print(evid,event.mag,event.mag_type,event.mag_method,event.mag_unc,event.nmag)
						else:
							mag_type = 'cMl_H'
							mag_method = 'NZ20'
					else:
						mag_type = 'cMl'
						mag_method = 'NZ20'
					if mag_type[0:3] == 'cMl':
						Q1 = mag_data.mag_corr.quantile(0.25)
						Q3 = mag_data.mag_corr.quantile(0.75)
						IQR=Q3-Q1
						lowqe_bound=Q1 - 1.5 * IQR
						upper_bound=Q3 + 1.5 * IQR

						IQR_mags = mag_data[~((mag_data.mag_corr < lowqe_bound) | (mag_data.mag_corr > upper_bound))]
						# Run a check for anomalous station magnitudes (+ or - 2 from original value)
	# 						if len(IQR_mags.sta.unique()) <= 3:
						IQR_mags = IQR_mags[(IQR_mags.mag_corr <= event.preferred_magnitude().mag + 2) & (IQR_mags.mag_corr >= event.preferred_magnitude().mag - 2)]
		
						CML = IQR_mags.mag_corr.mean()
						new_std = IQR_mags.mag_corr.std()
						CML_unc = np.sqrt(nz20_res ** 2 + np.var(IQR_mags.mag_corr))

						nmag = len(IQR_mags[~IQR_mags.mag_corr.isnull()].sta.unique())

						event_line.append([eventid, ev_datetime, ev_lat, ev_lon, ev_depth, ev_loc_type, ev_loc_grid,
							CML, mag_type, mag_method, CML_unc, event.preferred_magnitude().mag, 
							event.preferred_magnitude().magnitude_type, event.preferred_magnitude().mag_errors.uncertainty, 
							ev_ndef, ev_nsta, nmag, std, reloc])								
				else:
					event_line.append([eventid, ev_datetime, ev_lat, ev_lon, ev_depth, ev_loc_type, ev_loc_grid,
						pref_mag, pref_mag_type, pref_mag_method, 
						pref_mag_unc, event.preferred_magnitude().mag, 
						event.preferred_magnitude().magnitude_type, event.preferred_magnitude().mag_errors.uncertainty, 
						ev_ndef, ev_nsta, pref_mag_nmag, std, reloc])			
			else:
	# 				pref_mag = event.preferred_magnitude().mag
	# 				pref_mag_method = 'uncorrected'
	# 				pref_mag_unc = event.preferred_magnitude().mag_errors.uncertainty
				event_line.append([eventid, ev_datetime, ev_lat, ev_lon, ev_depth, ev_loc_type, ev_loc_grid,
					pref_mag, pref_mag_type, pref_mag_method, 
					pref_mag_unc, event.preferred_magnitude().mag, 
					event.preferred_magnitude().magnitude_type, event.preferred_magnitude().mag_errors.uncertainty, 
					ev_ndef, ev_nsta, pref_mag_nmag, std, reloc])			

			# Get arrival data
			for i, arrival in enumerate(arrivals):
				arid = eventid+str('a')+str(i+1)
				phase = arrival.phase
				toa = arrival.takeoff_angle
				arr_t_res = arrival.time_residual
				pick = [pick for pick in event.picks if pick.resource_id == arrival.pick_id][0]
				arr_datetime = pick.time
				net = pick.waveform_id.network_code
				sta = pick.waveform_id.station_code
				loc = pick.waveform_id.location_code
				chan = pick.waveform_id.channel_code
				phase_line.append([arid, arr_datetime, net, sta, loc, chan, phase, arr_t_res, eventid])

		event_df = pd.DataFrame(event_line, columns=['evid', 'datetime', 'lat', 'lon', 'depth', 'loc_type', 
			'loc_grid', 'mag', 'mag_type', 'mag_method', 'mag_unc', 'mag_orig', 'mag_orig_type', 'mag_orig_unc', 'ndef', 'nsta', 'nmag', 't_res', 'reloc'])	
		arrival_df = pd.DataFrame(phase_line, columns=['arid', 'datetime', 'net', 'sta', 'loc', 'chan', 'phase', 't_res', 'evid'])
		mag_df = pd.DataFrame(mag_line, columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 'amp_peak', 'amp_trough', 'amp_max','amp_unit','SNR','filtered','amp_time'])

		event_df.to_csv(event_df_file,index=False,header=False,mode='a')
		arrival_df.to_csv(arrival_df_file,index=False,header=False,mode='a')
		mag_df.to_csv(mag_df_file,index=False,header=False,mode='a')
	except Exception as e:
		print('There may be something wrong with event '+eventid, e)
	return

### Directory that you want to write CSVs to.
directory = '../output/'

### File with station corrections
sta_corr = pd.read_csv('sta_corr.csv')

### File with GEONET events
filename = 'earthquakes.csv'
geonet = pd.read_csv(filename,low_memory=False)
geonet = geonet.sort_values('origintime')
# geonet['origintime'] = pd.to_datetime(geonet['origintime']).astype('datetime64[ns]')
geonet['origintime'] = geonet.origintime.apply(lambda x: UTCDateTime(x).datetime)
geonet = geonet.reset_index(drop=True)

client = FDSN_Client("GEONET")

### Get station information to create dataframe
client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')
inventory_NZ = client_NZ.get_stations()
inventory_IU = client_IU.get_stations(network='IU',station='SNZO,AFI,CTAO,RAO,FUNA,HNR,PMG')
inventory_AU = client_IU.get_stations(network='AU')
inventory = inventory_NZ+inventory_IU+inventory_AU

station_info = []
for network in inventory:
	for station in network:
		station_info.append([network.code, station.code, station.latitude, station.longitude, station.elevation])

station_df = pd.DataFrame(station_info,columns=['net','sta','lat','lon','elev'])
station_df = station_df.drop_duplicates().reset_index(drop=True)

### Create output file if it does not exist
if not os.path.exists(directory):
	os.makedirs(directory)

### Enter years and months that you are interested in processing data for. Note that one
### year of data can take up to several days to process (in the case of 2016).
date_range = pd.date_range(start='2015-05',end='2015-06',freq='MS')

for idx,date in enumerate(date_range[0:-1]):
	year = date.year
	month = date.month
	geonet_sub_mask = (geonet.origintime >= date) & (geonet.origintime < date_range[idx+1])
	geonet_sub = geonet[geonet_sub_mask].reset_index(drop=True)

	event_list = [row.publicid for index, row in geonet_sub.iterrows()] 

	event_df = pd.DataFrame(columns=['evid', 'datetime', 'lat', 'lon', 'depth', 'loc_type', 
		'loc_grid', 'mag', 'mag_type', 'mag_method', 'mag_unc', 'mag_orig', 'mag_orig_type', 'mag_orig_unc', 'ndef', 'nsta', 'nmag', 't_res', 'reloc'])	
	arrival_df = pd.DataFrame(columns=['arid', 'datetime', 'net', 'sta', 'loc', 'chan', 'phase', 't_res', 'evid'])
	mag_df = pd.DataFrame(columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 'amp_peak','amp_trough', 'amp_max','amp_unit','SNR','filtered','amp_time'])

	event_df_file = directory+'events_'+str(year)+'_'+str(month)+'.csv'
	arrival_df_file = directory+'arrivals_'+str(year)+'_'+str(month)+'.csv'
	mag_df_file = directory+'mags_'+str(year)+'_'+str(month)+'.csv'
	
	### Create initial output files
	event_df.to_csv(event_df_file,index=False)
	arrival_df.to_csv(arrival_df_file,index=False)
	mag_df.to_csv(mag_df_file,index=False)

	### Run the magnitude converter program
	start_time = timeit.time()
	import dask
	delayed_results = []
	for i in event_list:
		result = dask.delayed(convert_eq)(i,client_NZ,client_IU,station_df,sta_corr,event_df_file,
			arrival_df_file,mag_df_file)
		delayed_results.append(result)
	results = dask.compute(*delayed_results)
	end_time = timeit.time()-start_time
	print('Took '+str(end_time)+' seconds to run')

		
# 		start_time = timeit.time()
# 		result_ids = []
# 		ray.init()
# 		result_ids = [convert_eq.remote(i,client_NZ,client_IU,station_df,sta_corr,event_df_file,
# 			arrival_df_file,mag_df_file) for i in event_list]
# 		results = ray.get(result_ids)
# 		ray.shutdown()
# 		print(timeit.time() - start_time)	


# events_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'events_*.csv')])
# 
# # Check for missing events from entire GeoNet database
# geonet_sub = geonet[geonet.origintime.dt.year < 2021]	
# event_list_df = events_df.evid.copy().reset_index(drop=True)
# try_again = geonet_sub[geonet_sub.publicid.isin(events_df.evid.values.astype('str')) == False].publicid.tolist()
# 
# event_df = pd.DataFrame(columns=['evid', 'datetime', 'lat', 'lon', 'depth', 'loc_type', 
# 	'loc_grid', 'mag', 'mag_type', 'mag_method', 'mag_unc', 'ndef', 'nsta', 'nmag', 't_res', 'reloc'])	
# arrival_df = pd.DataFrame(columns=['arid', 'datetime', 'net', 'sta', 'loc', 'chan', 'phase', 't_res', 'evid'])
# 		mag_df = pd.DataFrame(columns=['magid', 'net', 'sta', 'loc', 'chan', 'evid', 'mag', 'mag_type', 'mag_corr', 'mag_corr_method', 'amp', 'amp_trough', 'amp_max'])
# prop_df = pd.DataFrame(columns=['evid','net','sta','r_epi','r_hyp','az','b_az','toa'])
# 
# event_df_file = directory+'events_retry.csv'
# arrival_df_file = directory+'arrivals_retry.csv'
# mag_df_file = directory+'mags_retry.csv'
# prop_df_file = directory+'props_retry.csv'
# 
# event_df.to_csv(event_df_file,index=False)
# arrival_df.to_csv(arrival_df_file,index=False)
# mag_df.to_csv(mag_df_file,index=False)
# prop_df.to_csv(prop_df_file,index=False)
# 
# cores = int(multiprocessing.cpu_count()-1)
# pool = Pool(cores)
# pool.map(convert_eq, try_again)
# pool.close()
# pool.join()
# 
events_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'events_*.csv')])
mags_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'mags_*.csv')])	
arrivals_df = pd.concat([pd.read_csv(f,low_memory=False) for f in glob.glob(directory+'arrivals_*.csv')])	
# station_sub = station_df[station_df['sta'].isin(arrivals_df['sta'].unique())]

events_df.to_csv(directory+'earthquake_source_table.csv',index=False)
mags_df.to_csv(directory+'station_magnitude_table.csv',index=False)
arrivals_df.to_csv(directory+'phase_arrival_table.csv',index=False)
# station_sub.to_csv(directory+'site_table.csv',index=False)