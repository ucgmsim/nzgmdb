Generates the P and S Wave arrival times for records using a custom P wave picker and the inclusion of Geonet pick data

# Prerequisites
- Parse Geonet (mseed files generated)

# Process
For each mseed (mutiprocessed) tries to get the p-wave using a custom picker and the geonet phase arrival information.

## Custom P-Wave Picker
Using a custom P-Wave Picker which is based on the fixed-base viscously damped single-degree-of-freedom (SDF) oscillator model on each component 000, 090, ver.
First checks if the vertical component succeeds and if so uses that p_wave pick for the record, otherwise finds the max p-wave out of all 3 components (only 000 and 090 are left however) and then takes away 0.5 seconds.
Saves the datetime of the p-wave chosen from the starttime found in the mseed and then adds the time found for the p-wave. Time residual is not able to be computed for this and so is just set to 0.

Code can be found here (https://github.com/ucgmsim/nzgmdb/blob/d020c6e32a76c156c1c58ded49ca7f4c76ee0f5d/nzgmdb/phase_arrival/gen_phase_arrival_table.py#L46)

## Geonet P and S Wave Picker
Uses the FDSN client event picks to fetch the P and S wave data, when selecting ensures that the pick matches the network, station, location and first 2 letters of the channel.
This data is then paired with the arrival data to figure out if the time of the wave is a P or S wave as well as the time residual information.

## Merging data
Both of these sets are then merged together to form the final output phase arrival table.
The Custom P-Wave Picker has the top priority for P-Wave selection, where there are records that have no P-Wave from the custom picker then the Geonet value is added in its place.
S Waves from Geonet are just all added in as they are the only S Waves produced from either of the pickers.

# Output
Contains the following colums
* evid
* datetime (datetime of the P or S wave arrival)
* net
* sta
* loc
* chan
* phase (either P or S for P-Wave or S-Wave)
* t_res (time residual between the geonet phase picker and the arrival time observed)