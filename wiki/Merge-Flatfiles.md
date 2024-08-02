Merges all flatfiles to ensure to remove filtered entries and split IM results per component into different flatfiles

# Prerequisites

Distances calculated

# Process

Reads all the flatfiles to ensure that only events that are fully passed all the filtering steps applied to the ground_motion_im_catalogue.

Splits the ground_motion_im_catalogue table into 5 different tables for each component (000, 090, ver, rotd50, rotd100).
Also creates a duplicated version of these that contain a flatfile version with merged data from the event, site and propagation flatfiles.

# Output

Saves a file called "missing_cites.csv" which contains sites that have IM data but the is no record of them in the site_table.

Below is a list of all the flatfile csvs which are saved:
- earthquake_source_table
- station_magnitude_table
- phase_arrival_table
- site_table
- ground_motion_im_table_000
- ground_motion_im_table_090
- ground_motion_im_table_ver
- ground_motion_im_table_rotd50
- ground_motion_im_table_rotd100
- ground_motion_im_table_000_flat
- ground_motion_im_table_090_flat
- ground_motion_im_table_ver_flat
- ground_motion_im_table_rotd50_flat
- ground_motion_im_table_rotd100_flat