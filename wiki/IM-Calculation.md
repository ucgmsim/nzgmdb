Performs Intensity Measure Calculations such as pSA etc.

# Prerequisites

- Processed records (.000 .090 and .ver files in the processed directory)

# Process

Takes the processed records and calculates the Intensity Measures for each record and component.
Uses the following config settings for computation:

### IMS Computed:
- PGA
- PGV
- CAV
- AI
- Ds575
- Ds595
- MMI
- pSA
- FAS

### pSA periods: 

111 total periods.

0.010, 0.020, 0.022, 0.025, 0.029, 0.030, 0.032, 0.035, 0.036, 0.040, 0.042, 0.044, 0.045, 0.046, 0.048, 0.050, 0.055, 0.060, 0.065, 0.067, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100, 0.110, 0.120, 0.130, 0.133, 0.140, 0.150, 0.160, 0.170, 0.180, 0.190, 0.200, 0.220, 0.240, 0.250, 0.260, 0.280, 0.290, 0.300, 0.320, 0.340, 0.350, 0.360, 0.380, 0.400, 0.420, 0.440, 0.450, 0.460, 0.480, 0.500, 0.550, 0.600, 0.650, 0.667, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 1.000, 1.100, 1.200, 1.300, 1.400, 1.500, 1.600, 1.700, 1.800, 1.900, 2.000, 2.200, 2.400, 2.500, 2.600, 2.800, 3.000, 3.200, 3.400, 3.500, 3.600, 3.800, 4.000, 4.200, 4.400, 4.600, 4.800, 5.000, 5.500, 6.000, 6.500, 7.000, 7.500, 8.000, 8.500, 9.000, 9.500, 10.000, 11.000, 12.000, 13.000, 14.000, 15.000, 20.000

### FAS periods:

Log distribution between 0.01318257 and 100 with 389 numbers.

### Components:
- 000
- 090
- ver
- rotd50
- rotd100

# Output

Outputs a csv file per record with all the IM data for each of the components.
This file is placed in the IM directory named as evid_station_channel_location_IM.csv

A skipped records file is also created and saved in the flatfiles directory holding the record_id and the reason it failed,
at the moment this is only if the record is missing a component.