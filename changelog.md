# NZGMDB Changelog

## Version 4.4 - September 25 **2000-01-01 to 2024-12-31**
* Adding Broadband data (HH) to the database
* New Quality Filter to remove Broadband data during certain time periods due to sensitivity issues
* New Quality Filter to compare against an empirical GMPE (Atkinson 2022) to remove significant outliers
* Add ability to generate a report to compare NZGMDB versions
* Waveform Extraction seperated from Parse GeoNet step to help with multi-event detection
* Changed end time of waveform extraction to be p_time_est + ds595 * 3 stddev
* Handles multi-trace issues
* Adds installation dates for sites
* New Quality Filter to check for Jerk in the waveform from ClipNet

## Version 4.3 - July 25 **2000-01-01 to 2024-12-31**
* Sensitivity Fix (previously always taking first value not for actual datetime expected)
* Add back Reyners Relocations for Domain (Non FF or CMT solution) Earthquakes
* Seperation of fmin and fmax to horizontal and vertical components for filtering
* 7 new Finite Fault Models added
* Big speed increase with custom n_procs per process

---

## Version 4.2 - Feb 25 **2000-01-01 to 2024-12-31**
* PhaseNet for p-wave picking, save prob series as input for GMC
* GMC filter change to not include Vertical component
* Introduced the concept of a full database vs a quality database
* Save plane info from CCLD to source table, include avg strike, dip rake for srf's based on slip weighted average
* Remove records from the quality db if they have no station info, for full db add lon lat values
* Aftershocks
* New IM distribution over components
* Increased date range to full 24

---

## Version 4.1 - Nov 24 **2000-01-01 to 2024-6-30**
* Ds window increased from 1.2 to 2
* Fixed a run problem that skipped some events in 2016
* Fix duplicated SNZO site
* Increased date range to mid 24

---

## Version 4.0 - Oct 24 **2000-01-01 to 2022-12-31**
* Automated pipeline from v3.5
* Change to Openquake Afshari and Stewart model for Ds Window
* new file structure and waveform format
* CCLD for distances
* is_ground_level added and used as a filter (loc_elev and is_ground_level cols added)
* Filter down duplicate records with HN / BN for the same evid_sta combo (we prioritize HN over BN)
* Use fmin / 1.25 for HPF
* Include all GMC fmin values for each component in the flatfiles

---