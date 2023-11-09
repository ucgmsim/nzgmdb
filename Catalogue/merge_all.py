import pandas as pd
import os
import numpy as np

# Searches for unique events based on calculated GM IMs
directory = '../output/'

event_cat = pd.read_csv(directory+'earthquake_source_table_complete.csv',low_memory=False)
event_cat['datetime'] = pd.to_datetime(event_cat['datetime'],format='mixed')

# Remove events from far outside of NZ
event_cat.loc[event_cat.lon < 0, 'lon'] = 360 + event_cat.lon[event_cat.lon < 0]
event_cat = event_cat[(event_cat.lon < 190) & (event_cat.lon >155)]
event_cat.loc[event_cat.lon > 180, 'lon'] = event_cat.lon[event_cat.lon > 180] - 360
event_cat = event_cat[(event_cat.lat < -15)]
event_cat.loc[event_cat.strike == 360,'strike'] = 0
event_cat.loc[event_cat.rake > 180, 'rake'] = event_cat.rake[event_cat.rake > 180] - 360
event_cat.reset_index(drop=True,inplace=True)

mag_df = pd.read_csv(directory+'station_magnitude_table.csv',low_memory=False)

arr_df = pd.read_csv(directory+'phase_arrival_table.csv',low_memory=False)
arr_df['datetime'] = pd.to_datetime(arr_df['datetime'],utc=True)

prop_df = pd.read_csv(directory+'propagation_path_table_complete.csv',low_memory=False)
# prop_df['rrup'] = prop_df.r_hyp
# prop_df['rjb'] = prop_df.r_epi

sta_df = pd.read_csv(directory+'site_table_basin.csv',low_memory=False)
sta_df = sta_df.drop_duplicates() # Just in case any data is duplicated

gm_im_df = pd.read_csv(directory+'IM_catalogue/complete_ground_motion_im_catalogue_final.csv',low_memory=False)
unique_events = gm_im_df.evid.unique()

# Subsets unique events
event_cat_sub = event_cat[event_cat['evid'].isin(unique_events)].reset_index(drop=True)
event_cat_sub.drop_duplicates(subset='evid',inplace=True)
mag_df_sub = mag_df[mag_df['evid'].isin(unique_events)]
arr_df_sub = arr_df[arr_df['evid'].isin(unique_events)]
prop_df_sub = prop_df[prop_df['evid'].isin(unique_events)]
prop_df_sub = prop_df_sub[prop_df_sub[['evid','net','sta']].duplicated() == False]
# The below searches for AU ARPS values and removes them. There are two stations with the same site name.
indexNames = prop_df_sub[ (prop_df_sub['net'] == 'AU') & (prop_df_sub['sta'] == 'ARPS') ].index
prop_df_sub.drop(indexNames , inplace=True)
# prop_df_sub = prop_df_sub[prop_df_sub[['evid','sta']].duplicated() == True] # In case there is a duplicated station, this happens with ARPS [AU and NZ]
unique_stas = np.unique(np.append(gm_im_df['sta'].unique(),arr_df_sub['sta'].unique()))
station_sub = sta_df[sta_df['sta'].isin(unique_stas)]
directory = '../output/IM_catalogue/Tables/'

### Work on adding new parameters to the gm_im table
# select_gms = (gm_select_df['evid']+gm_select_df['sta']).values
# im_ids = (gm_im_df['evid']+gm_im_df['sta'])
# gm_im_df = gm_im_df[im_ids.isin(select_gms)].reset_index(drop=True)

gm_im_df_flat = gm_im_df.copy()

# gm_im_df_flat = gm_im_df_flat.set_index(gm_im_df_flat['evid']+gm_im_df_flat['sta']).merge(gm_select_df.set_index(\
# 	gm_select_df['evid']+gm_select_df['sta'])[['fmax_mean_X', 'fmax_mean_Y', 'fmax_mean_Z']],left_index=True,
# 	right_index=True,how='left').reset_index(drop=True)

gm_im_df_flat = gm_im_df_flat.merge(event_cat_sub[['evid','lat','lon','depth','mag','mag_type','tect_class',
	'reloc','domain_no','domain_type','strike','dip','rake','f_length','f_width','f_type',
	'z_tor','z_bor']],on='evid',how='left')
gm_im_df_flat.rename(columns={'lat':'ev_lat','lon':'ev_lon','depth':'ev_depth'},inplace=True)

gm_im_df_flat = gm_im_df_flat.merge(sta_df[['sta','lat','lon','Vs30','Vs30_std','Q_Vs30','T0','T0_std','Q_T0',
	'Z1.0','Z1.0_std','Q_Z1.0','Z2.5','Z2.5_std','Q_Z2.5','site_domain_no']],on='sta',how='left')
gm_im_df_flat.rename(columns={'lat':'sta_lat','lon':'sta_lon'},inplace=True)

gm_im_df_flat = gm_im_df_flat.set_index(gm_im_df_flat['evid']+gm_im_df_flat['sta']).join(prop_df_sub.set_index(\
	prop_df_sub['evid']+prop_df_sub['sta'])[['r_epi','r_hyp','r_jb','r_rup','r_x','r_y',
	'r_tvz','r_xvf']],how='left').reset_index(drop=True)

# gm_im_df_flat = gm_im_df_flat.set_index(gm_im_df_flat['evid']+gm_im_df_flat['sta']).join(clip_df.set_index(\
# 	clip_df['evid']+clip_df['sta'])[['clip_prob','clipped']],how='left').reset_index(drop=True)
	
gm_im_df_flat.sort_values(['datetime','sta','component'],inplace=True)

psa_columns = gm_im_df_flat.columns[gm_im_df_flat.columns.str.contains('pSA')].tolist()
fas_columns = gm_im_df_flat.columns[gm_im_df_flat.columns.str.contains('FAS')].tolist()
columns = ['gmid', 'datetime', 'evid', 'sta', 'loc', 'chan', 'component', 'ev_lat',
       'ev_lon', 'ev_depth', 'mag', 'mag_type', 'tect_class', 'reloc',
       'domain_no', 'domain_type', 'strike', 'dip', 'rake', 'f_length',
       'f_width', 'f_type', 'z_tor', 'z_bor', 'sta_lat', 'sta_lon',
       'r_epi', 'r_hyp', 'r_jb', 'r_rup', 'r_x', 'r_y', 'r_tvz', 'r_xvf', 'Vs30',
       'Vs30_std', 'Q_Vs30', 'T0', 'T0_std', 'Q_T0', 'Z1.0',
       'Z1.0_std', 'Q_Z1.0', 'Z2.5', 'Z2.5_std', 'Q_Z2.5', 'site_domain_no','PGA',
       'PGV', 'CAV', 'AI', 'Ds575', 'Ds595', 'MMI',
       'score_mean_X','fmin_mean_X','fmax_mean_X','multi_mean_X',
       'score_mean_Y','fmin_mean_Y','fmax_mean_Y','multi_mean_Y',
       'score_mean_Z','fmin_mean_Z','fmax_mean_Z','multi_mean_Z',
       'clip_prob','clipped']+psa_columns+fas_columns
# gm_im_df_flat = gm_im_df_flat[['gmid', 'evid', 'datetime', 'sta', 'loc', 'chan', 'component',
#     	'ev_lat', 'ev_lon', 'ev_depth', 'mag', 'mag_type', 'tect_class', 'domain_no', 
#     	'domain_type', 'strike', 'dip', 'rake', 'f_length', 'f_width', 'f_type', 'z_tor', 
#     	'z_bor','reloc','sta_lat', 'sta_lon', 'r_epi', 'r_hyp', 'r_jb', 'r_rup', 'r_x', 
#     	'r_y', 'r_tvz', 'Vs30_preferred', 'Vs30_preferred_model', 'T0', 'Z1.0', 'Z2.5', 
#     	'Z_preferred_model', 'Q_Vs30', 'Q_Z1.0', 'Q_Z2.5', 'Q_T0', 'site_domain_no', 
#     	'PGA', 'PGV', 'CAV', 'AI', 'Ds575', 'Ds595', 'MMI', 
#     	'pSA_0.01', 'pSA_0.02', 'pSA_0.03', 'pSA_0.04', 'pSA_0.05', 'pSA_0.075', 'pSA_0.1', 
#     	'pSA_0.12', 'pSA_0.15', 'pSA_0.17', 'pSA_0.2', 'pSA_0.25', 'pSA_0.3', 'pSA_0.4', 
#     	'pSA_0.5', 'pSA_0.6', 'pSA_0.7', 'pSA_0.75', 'pSA_0.8', 'pSA_0.9', 'pSA_1.0', 
#     	'pSA_1.25', 'pSA_1.5', 'pSA_2.0', 'pSA_2.5', 'pSA_3.0', 'pSA_4.0', 'pSA_5.0', 
#     	'pSA_6.0', 'pSA_7.5', 'pSA_10.0', 'score_X', 'f_min_X', 'score_Y', 'f_min_Y', 
#     	'score_Z', 'f_min_Z']]
gm_im_df_flat = gm_im_df_flat[columns]
# Find null Vs30 values and infill with Foster hybrid data    
# gm_im_df_flat.loc[gm_im_df_flat.Vs30.isnull(),'Vs30'] = gm_im_df_flat[gm_im_df_flat.Vs30.isnull()].join(station_sub[['sta','Vs30_foster_hybrid']].set_index('sta'),on='sta',how='left').Vs30_foster_hybrid.values

# Separate GM IM catalogues into separate tables        
df_000 = gm_im_df[gm_im_df.component == '000']
df_090 = gm_im_df[gm_im_df.component == '090']
df_ver = gm_im_df[gm_im_df.component == 'ver']
df_rotd50 = gm_im_df[gm_im_df.component == 'rotd50']
df_rotd100 = gm_im_df[gm_im_df.component == 'rotd100']

df_000 = df_000.drop(['score_mean_X','fmin_mean_X','fmax_mean_X',
    'multi_mean_X','score_mean_Z','fmin_mean_Z','fmax_mean_Z',
    'multi_mean_Z'],axis=1)
df_090 = df_090.drop(['score_mean_Y','fmin_mean_Y','fmax_mean_Y',
    'multi_mean_Y','score_mean_Z','fmin_mean_Z','fmax_mean_Z',
    'multi_mean_Z'],axis=1)
df_ver = df_ver.drop(['score_mean_X','fmin_mean_X','fmax_mean_X',
    'multi_mean_X','score_mean_Y','fmin_mean_Y','fmax_mean_Y',
    'multi_mean_Y'],axis=1)
df_rotd50 = df_rotd50.drop(['score_mean_Z','fmin_mean_Z','fmax_mean_Z',
    'multi_mean_Z'],axis=1)
df_rotd100 = df_rotd100.drop(['score_mean_Z','fmin_mean_Z','fmax_mean_Z',
    'multi_mean_Z'],axis=1)

df_000_flat = gm_im_df_flat[gm_im_df_flat.component == '000']
df_090_flat = gm_im_df_flat[gm_im_df_flat.component == '090']
df_ver_flat = gm_im_df_flat[gm_im_df_flat.component == 'ver']
df_rotd50_flat = gm_im_df_flat[gm_im_df_flat.component == 'rotd50']
df_rotd100_flat = gm_im_df_flat[gm_im_df_flat.component == 'rotd100']

df_000_flat = df_000_flat.drop(['score_mean_X','fmin_mean_X','fmax_mean_X',
    'multi_mean_X','score_mean_Z','fmin_mean_Z','fmax_mean_Z',
    'multi_mean_Z',],axis=1)
df_090_flat = df_090_flat.drop(['score_mean_Y','fmin_mean_Y','fmax_mean_Y',
    'multi_mean_Y','score_mean_Z','fmin_mean_Z','fmax_mean_Z',
    'multi_mean_Z'],axis=1)
df_ver_flat = df_ver_flat.drop(['score_mean_X','fmin_mean_X','fmax_mean_X',
    'multi_mean_X','score_mean_Y','fmin_mean_Y','fmax_mean_Y',
    'multi_mean_Y'],axis=1)
df_rotd50_flat = df_rotd50_flat.drop(['score_mean_Z','fmin_mean_Z','fmax_mean_Z',
    'multi_mean_Z'],axis=1)
df_rotd100_flat = df_rotd100_flat.drop(['score_mean_Z','fmin_mean_Z','fmax_mean_Z',
    'multi_mean_Z'],axis=1)

###

if not os.path.exists(directory):
	os.makedirs(directory)

# Writes subset data to new csv files
event_cat_sub.to_csv(directory+'earthquake_source_table.csv',index=False)
mag_df_sub.to_csv(directory+'station_magnitude_table.csv',index=False)
arr_df_sub.to_csv(directory+'phase_arrival_table.csv',index=False)
prop_df_sub.to_csv(directory+'propagation_path_table.csv',index=False)
station_sub.to_csv(directory+'site_table.csv',index=False)
# gm_im_df.to_csv(directory+'ground_motion_im_catalogue_final_expanded.csv',index=False)
df_000.to_csv(directory+'ground_motion_im_table_000.csv',index=False)
df_090.to_csv(directory+'ground_motion_im_table_090.csv',index=False)
df_ver.to_csv(directory+'ground_motion_im_table_ver.csv',index=False)
df_rotd50.to_csv(directory+'ground_motion_im_table_rotd50.csv',index=False)
df_rotd100.to_csv(directory+'ground_motion_im_table_rotd100.csv',index=False)
df_000_flat.to_csv(directory+'ground_motion_im_table_000_flat.csv',index=False)
df_090_flat.to_csv(directory+'ground_motion_im_table_090_flat.csv',index=False)
df_ver_flat.to_csv(directory+'ground_motion_im_table_ver_flat.csv',index=False)
df_rotd50_flat.to_csv(directory+'ground_motion_im_table_rotd50_flat.csv',index=False)
df_rotd100_flat.to_csv(directory+'ground_motion_im_table_rotd100_flat.csv',index=False)


# Function to find all duplicated events in the relocated catalogue and search for the 
# nearest datetime in the event catalogue. Write the ID to a list. Relabel duplicated IDs 
# next!

def relocation_dups(event_df,event_cat):

	event_df['evid'] = event_df.evid.astype(str)
	event_df['datetime'] = pd.to_datetime(event_df.datetime)

	event_cat['datetime'] = pd.to_datetime(event_cat.datetime).astype('datetime64[ns]')

	all_dups = event_df[event_df.evid.duplicated(keep=False)]
	all_dups_datetimes = all_dups.datetime.values

	non_dup_ids = []
	non_dup_indices = []
	for index,row in all_dups.iterrows():
		date = row.datetime
		i = np.argmin(np.abs(event_cat.datetime - date))
		non_dup_id = event_cat.iloc[i].evid
		non_dup_mag = event_cat.iloc[i].mag
	# 	non_dup_id = event_cat.iloc[np.where(event_cat_datetimes == min(event_cat_datetimes, key=lambda d: abs(d - date)))[0][0]].evid
		print(index,row.evid,non_dup_id,row.mag,non_dup_mag)
		non_dup_ids.append(non_dup_id)
		non_dup_indices.append(index)

	event_df.loc[non_dup_indices,['evid']] = non_dup_ids

	event_df_dups = event_df[event_df.evid.duplicated()]

	drop_dups = []
	for index,row in event_df_dups.iterrows():
		evid = row.evid
		event_df_dup_pair = event_df[event_df.evid == evid]
		cat_time = event_cat[event_cat.evid == evid].datetime.iloc[0]
		i = np.argmax(np.abs(event_df_dup_pair.datetime - cat_time))
		dup_drop = event_df_dup_pair.iloc[i].name
		print(dup_drop)
		drop_dups.append(dup_drop)

	event_df = event_df.drop(index=drop_dups)
	
	return event_df

# 	event_df.to_csv('martins_new.csv',index=False)

# event_df = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/Scripts/EQ/martins_test.csv',low_memory=False)
# event_cat = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/earthquake_source_table.csv',low_memory=False)
# 
# relocation_dups(event_df, event_cat)