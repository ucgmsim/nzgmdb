# This program finds all ground motion IM data and compiles it to a single catalogue.
# It also searches for additional data not provided in the additional catalogues by
# reading QuakeML files in the mseed directories. This program is very dependent on
# directory structure. It also takes advantage of multiple processes because otherwise
# it would be very slow.
#
# 4 July 2023, Removed basedir from code; XML files are in the preferred directories now.
# Also removed gmid, that will be implemented in a later step.
#
# 31 October 2023, Cleaned up the script, removed reference to parameters that are no 
# longer part of the GMC (standard deviations, which were not useful)

import glob
import pandas as pd
import os
from datetime import datetime, timedelta
import obspy as op
from multiprocess import Pool,cpu_count
import itertools

def get_gm_data(file, preferred_dir):
	try:
		file_path = os.path.dirname(file).split('/')
		date, time = file_path[-1].split('_')
		dt = datetime.strptime(date+time,'%Y-%m-%d%H%M%S')

		folderA = str(dt.year)
		folderB = dt.strftime('%m_%b')
		folderC = dt.strftime('%Y-%m-%d_%H%M%S')

		# Sometimes the folder names do not match, this should account for that issue.
		directory = preferred_dir+folderA+'/'+folderB+'/'+folderC
		if not os.path.exists(directory):
			folderC = (dt - timedelta(seconds=1)).strftime('%Y-%m-%d_%H%M%S')
			directory = preferred_dir+folderA+'/'+folderB+'/'+folderC
		if not os.path.exists(directory):
			folderC = (dt + timedelta(seconds=1)).strftime('%Y-%m-%d_%H%M%S')
			directory = preferred_dir+folderA+'/'+folderB+'/'+folderC			

		# Old xml name format
		# 	xml_name = dt.strftime('%Y%m%d_%H%M%S.xml')
		# 	xml_file = base_dir+folderA+'/'+folderB+'/'+folderC+'/'+xml_name
		# 	event = op.read_events(xml_file)[0]
		# 	evid = str(event.resource_id).split('/')[-1]

		# New xml name format
		xml_path = preferred_dir+folderA+'/'+folderB+'/'+folderC+'/*.xml'
		xml_name = os.path.basename(glob.glob(xml_path)[0])
		evid = xml_name.split('.')[0]

		df = pd.read_csv(file).reset_index(drop=True)
		df['evid'] = evid
		df['datetime'] = pd.to_datetime(dt).to_datetime64()

		folderC = dt.strftime('%Y-%m-%d_%H%M%S')

		station_list = df.station.unique()

		for index, row in df.iterrows():
			sta = row.station
			file_name = glob.glob(preferred_dir+folderA+'/'+folderB+'/'+folderC+'/mseed/data/'+dt.strftime('%Y%m%d_%H%M%S_')+sta+'_*'+'_*.mseed')[0]
			print(file_name)
			loc = file_name.split('_')[-2]
			chan = file_name.split('_')[-1].split('.')[0]
			df.loc[index,['loc','chan']] = [loc,chan]
		return df
	except:
		print('File missing: '+file)

### Minimum and maximum magnitude used for searching for relevant files, year for data after
### 2021
lower_mag = '4'
upper_mag = '10'
year = '2022'

### NOTE: Data is very path dependent, if running from your own machine, you will likely
### need to change file paths information in the below lines (until preferred_dir)

### Ground motion intensity measure root
if year:
	search_dir_year = year+'_'
else:
	search_dir_year = ''
search_dir = '/Volumes/SeaJade 2 Backup/NZ/'+search_dir_year+'GM_IM_'+lower_mag+'-'+upper_mag+'_fas_2/*/*/*/'
# search_dir = '/Volumes/SeaJade 2 Backup/NZ/CPLB_'+search_dir_year+'GM_IM_'+lower_mag+'-'+upper_mag+'_fas_2/*/*/*/'

### GMC (ground motion classification) file
if year:
	mseed_year = '_'+year
else:
	mseed_year = ''
gmc_file = '/Volumes/SeaJade 2 Backup/NZ/gmc_record/20231027_predictions/predictions_'+lower_mag+'-'+upper_mag+mseed_year+'.csv'
# gmc_file = '/Volumes/SeaJade 2 Backup/NZ/gmc_record/20231027_predictions/predictions_CPLB_'+lower_mag+'-'+upper_mag+mseed_year+'.csv'
file_list = glob.glob(search_dir+'gm_all.csv')
directory = '../output/'

### Directory with preferred mseeds used to find the matching files
preferred_dir = '/Volumes/SeaJade 2 Backup/NZ/mseed_'+lower_mag+'-'+upper_mag+mseed_year+'_preferred/'
# preferred_dir = '/Volumes/SeaJade 2 Backup/NZ/CPLB_mseed_'+lower_mag+'-'+upper_mag+mseed_year+'_preferred/'

### Initiate and run the program over multiple cores for minimum compilation time
cores = int(cpu_count()-1)
pool = Pool(cores)
df_all = pd.concat(pool.starmap(get_gm_data, zip(file_list,itertools.repeat(preferred_dir))),ignore_index=True)
pool.close()
pool.join()

### Prepare the GM IM results for writing to .CSV file
df_all = df_all.sort_values(['datetime','station','component'])
df_all = df_all.rename(columns = {'station':'sta'})
df_all['loc'] = df_all['loc'].astype('int')

# Add ground motion quality results and write to a separate 'final' file
# gmc_results = pd.read_csv(
# 	'/Volumes/SeaJade2/NZ/Reports/For Robin/GMC_cat/meta_gmc_results.csv',low_memory=False)
gmc_results = pd.read_csv(gmc_file,low_memory=False)
gmc_results['chan'] = gmc_results['record_id'].str.split('_').str[-2]
gmc_results['loc'] = gmc_results['record_id'].str.split('_').str[-3].astype('int')
gmc_results = gmc_results.rename(columns={'event_id':'evid','station':'sta'})
gmc_results['evid'] = gmc_results.evid.str.split('.').str[0]

new_df = gmc_results.drop_duplicates(subset='record').reset_index(drop=True)

new_df[['score_mean_X', 'fmin_mean_X', 'multi_mean_X']] = gmc_results[gmc_results.component == 'X'][['score_mean', 'fmin_mean', 'multi_mean', ]].values
new_df[['score_mean_Y', 'fmin_mean_Y', 'multi_mean_Y']] = gmc_results[gmc_results.component == 'Y'][['score_mean', 'fmin_mean', 'multi_mean', ]].values
new_df[['score_mean_Z', 'fmin_mean_Z', 'multi_mean_Z']] = gmc_results[gmc_results.component == 'Z'][['score_mean', 'fmin_mean', 'multi_mean', ]].values

gm_final = pd.merge(df_all,new_df[['evid','sta','chan','loc',
	'score_mean_X','fmin_mean_X', 'multi_mean_X',
    'score_mean_Y','fmin_mean_Y', 'multi_mean_Y',
    'score_mean_Z','fmin_mean_Z', 'multi_mean_Z',
    ]],on=['evid','sta','chan','loc'],how='left')

if not os.path.exists(directory+'/IM_catalogue'):
    os.makedirs(directory+'/IM_catalogue')
gm_final.to_csv(directory+'IM_catalogue/ground_motion_im_catalogue_'+search_dir_year+lower_mag+'-'+upper_mag+'_final.csv',index=False)
# gm_final.to_csv(directory+'IM_catalogue/ground_motion_im_catalogue_CPLB_'+search_dir_year+lower_mag+'-'+upper_mag+'_final.csv',index=False)