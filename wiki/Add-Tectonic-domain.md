Adds the tectonic type to the earthquake source table

# Prerequisites
earthquake source table generated so Parse Genet needs to have been run

# Process

## Updating event info based on CMT / NZSMDB data
Fetches the CMT and NZSMDB data from the github repos directly so they are always up to date at the time that you run the pipeline.
Updates each entry in the earthquake source table if the same evid is found in the CMT data. Updates the ["mag", "lat", "lon", "depth"], notifies in the mag-method to be CMT
Then merges the NZSMDB data such as CuspID and TectClass where the evid matches

## Regions
Creates 3 region areas based on if the lat lon points are offshore to the fault, on the fault or downdip to the fault. With the Kermadec and Hikurangi and Puysegur datasets. d_s and d_d values of 10, 47, 11, 30 respectively
Regions are based on this 
`df_a = df[(df.depth < d_s)]
df_b = df[(df.depth >= d_s) & (df.depth <= d_d)]
df_c = df[(df.depth > d_d)]`

Then the regions are used with a horizontal and vertical threshold from each lat lon to each point on the region and depth to determine the tect class.

`Region A (vertical prism offshore of seismogenic zone of fault plane):
depth<60km: 'Outer rise'
depth>=60km: 'Slab'
 
Region B (vertical prism containing seismogenic zone of fault plane):
depth<min(slab surface, 20km): 'Crustal'
min(slab surface, 20km)>depth>60km: 'Interface'
depth>60km: 'Slab'
 
Region C (vertical prism downdip of the seismogenic zone of the fault plane):
depth<30km: 'Crustal'
30km<depth<slab surface: 'Undetermined'
depth>slab surface: 'Slab'
 
Elsewhere (Farfield):
depth<30km: 'Crustal'
depth>30km: 'Undetermined'`

## Domain
Domain is found using the same method at the site table basin with the "TectonicDomains_Feb2021_8_NZTM.shp" file and the domain number and the domain type is also added, if not found then defaults to 0, Oceanic

Code can be found at (https://github.com/ucgmsim/nzgmdb/blob/d020c6e32a76c156c1c58ded49ca7f4c76ee0f5d/nzgmdb/data_retrieval/tect_domain.py#L1)

# Output
The Earthquake source table with updated columns and extra columns such as domain and tectclass definitions from CMT or NZSMDB per event id.
