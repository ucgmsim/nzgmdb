# This program merges all of the ground-motion intensity-measure files along with the
# maximum frequency data and filters according to parameters specified in our paper.

# Note, also require openpyxl to read the xlsx file
import pandas as pd
import os
import glob

directory = '../output/IM_catalogue'

files = glob.glob(directory+'/ground_motion*final.csv') # Check to make sure this doesn't merge in an already merged file

gm_im_dfs = pd.concat([pd.read_csv(file, low_memory=False) for file in files])
gm_im_dfs.sort_values('datetime',inplace=True)

fmax_df = pd.read_excel(directory+'/0_fmax_v2.xlsx')
clip_df = pd.read_csv(directory+'/Tables/clip_table.csv')

# Excise scores that made it through the process...
gm_im_dfs = gm_im_dfs[gm_im_dfs.chan.isin(['HN','BN'])]
gm_im_dfs = gm_im_dfs[(gm_im_dfs.score_mean_X >= 0.5) & (gm_im_dfs.score_mean_Y >= 0.5) & (gm_im_dfs.score_mean_Z >= 0.5)]
# Filter by multi mean
gm_im_dfs = gm_im_dfs[(gm_im_dfs.multi_mean_X <= 0.2) & (gm_im_dfs.multi_mean_Y <= 0.2) & (gm_im_dfs.multi_mean_Z <= 0.2)]
# Filter by fmin mean
gm_im_dfs = gm_im_dfs[(gm_im_dfs.fmin_mean_X <= 2) & (gm_im_dfs.fmin_mean_Y <= 2) & (gm_im_dfs.fmin_mean_Z <= 2)]

# Filter for Ds595
sub = gm_im_dfs[(gm_im_dfs.component == 'ver') | (gm_im_dfs.component == '000') | (gm_im_dfs.component == '090')].reset_index(drop=True)
sub_group = sub.groupby(['evid','sta']).sum()
Ds595_filter = sub_group[sub_group.Ds595 < 3].reset_index()[['evid','sta']]
for idx,row in Ds595_filter.iterrows():
	gm_im_dfs = gm_im_dfs[~((gm_im_dfs.evid == row.evid) & (gm_im_dfs.sta == row.sta))]
	
gm_im_dfs.reset_index(drop=True,inplace=True)

# Merge fmax data and filter by fmax mean
gm_im_dfs = gm_im_dfs.set_index(gm_im_dfs.evid+'_'+gm_im_dfs.sta+'_'+gm_im_dfs.chan).merge(fmax_df.set_index('ev_sta'),
	left_index=True,right_index=True,how='left').reset_index(drop=True)
gm_im_dfs[['fmax_mean_X','fmax_mean_Y','fmax_mean_Z']] = gm_im_dfs[['fmax_090','fmax_000','fmax_ver']].values
gm_im_dfs.drop(columns=['fmax_090','fmax_000','fmax_ver'],inplace=True)
gm_im_dfs = gm_im_dfs[(gm_im_dfs.fmax_mean_X > 4.1) & (gm_im_dfs.fmax_mean_Y > 4.1) & (gm_im_dfs.fmax_mean_Z > 4.1)].reset_index(drop=True)

# Merge clip data and filter by clip probability
gm_im_dfs = gm_im_dfs.set_index(gm_im_dfs.evid+'_'+gm_im_dfs.sta).merge(clip_df.set_index(
	clip_df.evid+'_'+clip_df.sta)[['clip_prob','clipped']], left_index=True,right_index=True,
	how='left').reset_index(drop=True)
gm_im_dfs = gm_im_dfs[gm_im_dfs.clip_prob <= 0.2].reset_index(drop=True)

gm_im_dfs.sort_values(['datetime','sta','component'],inplace=True)

unique_events = gm_im_dfs.evid.unique()

# Assign ground motion IDs
for event in unique_events:
	print(event)
	gmids = (event+'gm'+(gm_im_dfs[gm_im_dfs.evid.values == event].reset_index().index.to_series() + 1).astype('str')).values
	gm_im_dfs.loc[gm_im_dfs.evid.values == event,'gmid'] = gmids

gm_im_dfs.reset_index(drop=True)

psa_columns = gm_im_dfs.columns[gm_im_dfs.columns.str.contains('pSA')].tolist()
fas_columns = gm_im_dfs.columns[gm_im_dfs.columns.str.contains('FAS')].tolist()
columns = ['gmid', 'evid', 'datetime', 'sta', 'loc', 'chan', 'component', 'PGA',
       'PGV', 'CAV', 'AI', 'Ds575', 'Ds595', 'MMI',
       'score_mean_X','fmin_mean_X','fmax_mean_X','multi_mean_X',
       'score_mean_Y','fmin_mean_Y','fmax_mean_Y','multi_mean_Y',
       'score_mean_Z','fmin_mean_Z','fmax_mean_Z','multi_mean_Z',
       'clip_prob','clipped']+psa_columns+fas_columns
gm_im_dfs = gm_im_dfs[columns]


gm_im_dfs.to_csv(directory+'/complete_ground_motion_im_catalogue_final.csv',index=False)