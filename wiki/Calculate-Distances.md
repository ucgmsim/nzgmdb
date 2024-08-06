Determines correct nodal plane, calculates rrup values for propagation table

# Prerequisites

Merged IM data

# Process

Determines the correct nodal plane to calculate rrup values for the propagation table, does the following for every event:

Checks if the event id is within the set of srf files that define specific earthquakes such as Christchurch Feb 2011 and Darfield as well as Kaikoura 2016.
If the event id is within this set then it loads the srf files and gets the nodal plane information as well as the srf points from the srf directly and then moves on to the rrup calculations.

Otherwise it checks the modified CMT solutions file "GeoNet_CMT_solutions_20201129_PreferredNodalPlane_v1.csv" as these have been
previously determined for the 1st nodal plane being correct and so strike dip and rake are extracted from this file.

If the event is in neither of these then it checks the general CMT solutions file which is fetched directly from Geonet's GitHub repo.
A function called mech_rot is then applied which checks which nodal plane is as close to the region specific strike as possible and then selects that nodal plane for strike dip and rake.

The code for this function can be found here (https://github.com/ucgmsim/nzgmdb/blob/2fa80fa0917989c1103ed0a1e4821be7bb8f0e73/nzgmdb/calculation/distances.py#L61)

If still the event is in none of these then the general domain strike dip and rake is used which is determined by this file "focal_mech_tectonic_domain_v1.csv"

After the strike dip and rake are determined, if the length and width are not defined then the following is used to get the Length and Width variables
```
mag_scale_slab_min = 5.9
mag_scale_slab_max = 7.8
 Get the width and length using magnitude scaling
    if tect_class == "Interface":
        # Use SKARLATOUDIS2016
        dip_dist = np.sqrt(mag_scaling.mw_to_a_skarlatoudis(mag))
        length = dip_dist
    elif tect_class == "Slab" and mag_scale_slab_min <= mag <= mag_scale_slab_max:
        # Use STRASSER2010SLAB
        length = strasser_2010.mw_to_l_strasser_2010_slab(mag)
        dip_dist = strasser_2010.mw_to_w_strasser_2010_slab(mag)
    else:
        # Use LEONARD2014
        length = mag_scaling.mw_to_l_leonard(mag, rake)
        dip_dist = mag_scaling.mw_to_w_leonard(mag, rake)
```

From this if the srf points are not already provided from the srf files then a grid of points are generated using the length and width and the strike and dip values.

Once srf points are guaranteed then the rrup values are calculated for each station and the event and stored in a dataframe.
This also includes rjb and then rx and ry are calculated using the closest points on the srf to the station.

A few other distance metrics are calculated such as r_epis, r_hyps, azs, b_azs and then also tvz_length and boundary_dists_rjb which are dealing with the volcanic zone region.
# Output

All of this data is added to a propagation dataframe which is then saved as a csv file which contains every event and site pair.
Extra event data such as strike, dip, rake, f_length, f_width, f_type, z_tor and z_bor are added to the earthquake source table based on results done during this calculation.
