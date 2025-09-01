
### 🔹 Waveform Window Calculation

The waveform download window is determined using seismic travel time models and configuration parameters:

#### **Travel Time Estimation**
- **P-wave arrival:** Calculated using TauPy iasp91 model
- **S-wave arrival:** Calculated using TauPy iasp91 model

#### **Duration (Ds) Calculation**
- **Model:** Afshari and Stewart (2016) from OpenQuake
- **Parameters:**
  - Vs30 from site table (default: `vs30: 500` m/s if unavailable)
  - Default rake value: 90°
  - Magnitude from preferred magnitude
  - Z1.0 estimated using Chiou-Young 2008 model from Vs30

#### **Window Definition**
- **Start time:** P-wave arrival minus `min_time_difference: 15` seconds
- **End time:** S-wave arrival + Ds × `ds_multiplier: 2`, also ensures a minimum duration
- **Minimum duration:** Controlled by `min_time_difference` to ensure adequate waveform length

Example with a synthetic waveform to illustrate the window:
![](images/waveform_extraction_window.png)

### 🔹 Waveform Data Retrieval

Waveforms are downloaded using the FDSN Client with specific constraints:

#### **Channel Selection**
- **Channel Selection:** `channel_codes: [HN?, BN?, HH?]` from configuration, where HN and BN are Strong Motion channels and HH is Broadband
- **Three-component data:** Horizontal (N-S, E-W) and vertical components

#### **Error Handling**
- **Incomplete reads:** Up to 3 retry attempts for network issues
- **Missing data:** Graceful skipping when no data available
- **File size errors:** Catches ObsPy errors for corrupted/small files that can't be processed
- **Network timeouts:** Robust retry mechanism for network failures

### 🔹 Waveform Quality Filtering

Downloaded waveforms undergo quality assessment using ClipNet with configuration thresholds:

#### **Clipping Detection**
- **Method:** gmprocess ClipNet algorithm
- **Threshold:** `clip_threshold: 0.2` from config.yaml
- **Action:** Records exceeding threshold are flagged and skipped
- **Magnitude bounds:** `mag_clip_low: 3.0`, `mag_clip_high: 8.8`
- **Distance bounds:** `dist_clip_low: 0.0`, `dist_clip_high: 645.0`

The output of this is saved to a `geonet_clipped_records.csv` file to be used during the quality_db step.

#### **Component Splitting**

Some Stream objects require extra splitting for a single evid_station combinations as there can be many different "locations" or "channels" for the same record.

- **Processing:** Streams split into individual 3-component sets
- **Validation:** Ensures complete three-component data
- **Location codes:** Handles multiple location codes per station

### 🔹 Waveform File Management

Successfully processed waveforms are saved in standardised format:

#### **File Naming Convention**
```
{event_id}_{station}_{channel}_{location}.mseed
```

#### **Directory Structure**
- **Storage location:** `waveforms/` subdirectory
- **Organisation:** Hierarchical by year then event ID then mseed directory
- **Format:** ObsPy Stream objects saved as MSEED files

### 🔹 Station Magnitude Processing

For each station-event pair, magnitude information is extracted:

#### **Magnitude Extraction Logic**
1. **Primary attempt:** Match Z-channel with first 2 channel codes
2. **Secondary attempt:** Use any channel matching first 2 codes
3. **Fallback:** Set magnitude to None, type to preferred magnitude type

#### **Amplitude Information**
- **Amplitude values:** Extracted from event amplitude objects
- **Units:** Preserved from original GeoNet data
- **Quality flags:** Maintained for downstream processing


---


### 🔹 Station Magnitude Table (`station_magnitude_table.csv`)

Records station-specific magnitude measurements:

| Column | Description |
|--------|-------------|
| `mag_id` | Unique magnitude identifier |
| `net` | Network code |
| `sta` | Station code |
| `loc` | Location code |
| `chan` | Channel code |
| `event_id` | Associated event identifier |
| `sta_mag` | Station magnitude value |
| `sta_mag_type` | Station magnitude type |
| `sta_mag_method` | Magnitude calculation method |
| `amp` | Amplitude measurement |
| `amp_unit` | Amplitude units |


#### **Clipped Records (`geonet_clipped_records.csv`)**
- Records flagged by ClipNet as having excessive clipping (above `clip_threshold: 0.2`)
- Contains record identifier and clipping metrics


### 🔹 MSEED Waveform Files

- **Location:** `waveforms/` directory
- **Format:** Standard MSEED format compatible with ObsPy
- **Naming:** `{event_id}_{station}_{channel}_{location}.mseed`
- **Content:** Three-component seismic waveforms ready for processing