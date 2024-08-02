Performs Intensity Measure Calculations such as pSA etc.

# Prerequisites

Processed records (.000 .090 and .ver files in the processed directory)

# Process

Takes the processed records and calculates the Intensity Measures for each record and component.
Uses the following config settings for computation:

IMS Computed:
- PGA
- PGV
- CAV
- AI
- Ds575
- Ds595
- MMI
- pSA
- FAS

pSA periods: 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0

Components:
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