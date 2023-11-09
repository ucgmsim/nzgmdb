import pandas as pd
from obspy.clients.fdsn import Client as FDSN_Client
import numpy as np
import fiona
from pyproj import Transformer

# def TVZ_calc(cat):
#     from shapely.geometry import Point
#     from shapely.geometry.polygon import Polygon
#     import numpy as np
#     from pyproj import Transformer			# conda install pyproj
# 
#     wgs2nztm = Transformer.from_crs(4326, 2193)
# 
#     # Taupo VZ polygon acquired from https://www.geonet.org.nz/data/supplementary/earthquake_location_grope
#     tect_domain_points = pd.read_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/tectonic domains/tectonic_domain_polygon_points.csv',low_memory=False)
#     tvz_points = tect_domain_points[tect_domain_points.domain_no == 4][['latitude','longitude']]
#     taupo_transform = np.dstack(np.array(wgs2nztm.transform(tvz_points.latitude,tvz_points.longitude)))[0]
#     taupo_polygon = Polygon(taupo_transform)
# 
#     for index,row in cat.iterrows():
#         sta_id = row.sta
#         lat = row.lat
#         lon = row.lon
#         if lon < 0:
#             lon = lon+360
# 
#         point = Point(lat,lon)
# 
#         if taupo_polygon.contains(point):
#             cat.loc[index,'TVZ'] = 1
#         else:
#             cat.loc[index,'TVZ'] = 0
# 
#     return cat

def get_domains(df,shapes,transformer):
	from shapely.geometry import Point	# conda install shapely
	from shapely.geometry.polygon import Polygon
	import numpy as np
	
	x,y = transformer.transform(df.lon,df.lat)
	points = [x,y]

	point = Point(points)

	no, name, type = [],[],[]
	for layer in shapes:
		domain_no = layer['properties']['Domain_No']
		domain_name = layer['properties']['DomainName']
		domain_type = layer['properties']['DomainType']
		geometry_type = layer['geometry']['type']
		geometry_coords = layer['geometry']['coordinates']
		if geometry_type == 'MultiPolygon':
			for coords in geometry_coords:		
				polygon = Polygon(coords[0])
				if polygon.contains(point):
					no, name, type = domain_no, domain_name, domain_type
		else:
			polygon = Polygon(geometry_coords[0])
			if polygon.contains(point):
				no, name, type = domain_no, domain_name, domain_type
	if not no:
		no, name, type = 0, 'Oceanic', None
	return pd.Series([no, name, type])

	
shape = fiona.open("../TectonicDomains/TectonicDomains_Feb2021_8_NZTM.shp")

# Writes layers of the shape file to a shapes array; prevents read errors if accessing
# the shape file directly
shapes = []
for layer in shape:
	domain_no = layer['properties']['Domain_No']
	domain_name = layer['properties']['DomainName']
# 	print(domain_no,domain_name)
	shapes.append(layer)

wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)

client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')

client_NZ = FDSN_Client("GEONET")
client_IU = FDSN_Client('IRIS')
inventory_NZ = client_NZ.get_stations()
inventory_IU = client_IU.get_stations(network='IU',station='SNZO,AFI,CTAO,RAO,FUNA,HNR,PMG')
inventory = inventory_NZ+inventory_IU
station_info = []
for network in inventory:
	for station in network:
		station_info.append([network.code, station.code, station.latitude, station.longitude, station.elevation])
sta_df = pd.DataFrame(station_info,columns=['net','sta','lat','lon','elev'])
sta_df = sta_df.drop_duplicates().reset_index(drop=True)

# sc_df = pd.read_excel('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/site_info/Geonet  Metadata  Summary_v1.4.xlsx')
sc_df = pd.read_csv('../Geonet  Metadata  Summary_v1.4.csv')
sc_df.rename(columns={'Name':'sta', 'Lat':'lat', 'Long':'lon', 'NZS1170SiteClass':'site_class', 'Vs30_median':'Vs30',
       'Sigmaln_Vs30':'Vs30_std', 'T_median':'Tsite', 'sigmaln_T':'Tsite_std', 'Q_T':'Q_Tsite',
       'D_T':'D_Tsite', 'T_Ref':'Tsite_ref', 'Z1.0_median':'Z1.0', 'sigmaln_Z1.0':'Z1.0_std', 'Z1.0_Ref':'Z1.0_ref',
       'Z2.5_median':'Z2.5', 'sigmaln_Z2.5':'Z2.5_std', 'Z2.5_Ref':'Z2.5_ref'},inplace=True)
merged_df = sc_df.set_index(sc_df.sta).join(sta_df.set_index(sta_df.sta)[['net','elev']],how='left').reset_index(drop=True)
merged_df[['site_domain_no', 'site_domain_name', 'site_domain_type']] = merged_df.apply(lambda x: get_domains(x,shapes,wgs2nztm),axis=1)       
 
merged_final_df = merged_df[['net', 'sta', 'lat', 'lon', 'elev', 'site_class', 'Vs30', 'Vs30_std', 
       'Q_Vs30', 'Vs30_Ref', 'Tsite', 'Tsite_std', 'Q_Tsite', 'D_Tsite', 'Tsite_ref',
       'Z1.0', 'Z1.0_std', 'Q_Z1.0', 'Z1.0_ref', 'Z2.5', 'Z2.5_std', 'Q_Z2.5',
       'Z2.5_ref', 'site_domain_no']].copy()
merged_final_df['Z2.5'] = merged_final_df['Z2.5'] / 1000

drop_index = merged_final_df[(merged_final_df.net == 'AU') & (merged_final_df.sta == 'ARPS')].index

merged_final_df.drop(index=drop_index,inplace=True)

merged_final_df.to_csv('/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/converted_output/site_table_response.csv',index=False)      