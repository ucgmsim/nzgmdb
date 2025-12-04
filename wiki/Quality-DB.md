# 🗃️ Quality Database

This step in the NZGMDB pipeline applies comprehensive quality filters to create the final database of high-quality seismic records. The Quality-DB step consolidates all intensity measure data with quality metrics from previous processing steps, removing low-quality records through a systematic 9-step filtering process.

---

## 🚀 Entry Point

To create the quality database, run the following Python script:

```bash
python -m nzgmdb.scripts.run_nzgmdb create-quality-db <main_dir>
```

- **<main_dir>** is the top-level output directory where NZGMDB stores its results.

Example:
```bash
python -m nzgmdb.scripts.run_nzgmdb create-quality-db nzgmdb_output/
```

Optional parameters include:
- `--bypass-records-ffp`: Path to bypass records file containing record IDs that should skip quality checks

This will create the quality database in:
```bash
nzgmdb_output/quality_db/
```

And update the skipped records file:
```bash
nzgmdb_output/flatfiles/quality_skipped_records.csv
```

---

## 📋 Prerequisites

The Quality-DB step requires the following inputs from previous pipeline step:
- **[Merge Flatfiles](Merge-Flatfiles.md)** - Provides the consolidated ground motion IM catalogue and flatfiles to filter

---

## ⚙️ Process

The Quality-DB step loads the `ground_motion_im_rotd50_flat.csv` file and applies a comprehensive series of quality filters through the `apply_all_filters()` function. Each filter removes records that don't meet specific quality criteria, with detailed tracking of why records were excluded.

### 🔹 Filter Sequence Overview

The filtering process follows a systematic 9-step approach:

1. **Filter by minimum magnitude**
2. **Filter by presence of GMC predictions**
3. **Filter by score mean**
4. **Filter by multi mean**
5. **Filter by fmax**
6. **Filter by fmin**
7. **Filter by missing station information**
8. **Ensure only ground level locations are used**
9. **Filter out clipped records**
10. **Filter out jerk records**
11. **Filter by sensitivity ignore list**
12. **Filter by empirical prediction residuals**
13. **Select the appropriate channel for duplicate HN/BN records**

### 🔹 Detailed Filter Descriptions

#### 1. Filter by Minimum Magnitude

**Purpose**: Removes records from events below a specified magnitude threshold.

**Implementation**:
- Uses the `mag` field in the event metadata
- Default threshold: `min_mag = 3.5` (configurable in `config.yaml`)
- Records with magnitude below threshold are removed


**Configuration**:
```yaml
# Minimum magnitude for quality filtering in config.yaml
quality_min_mag: 3.5  # Minimum magnitude for record inclusion
```

#### 2. Filter by Presence of GMC Predictions

**Purpose**: Ensures all records have quality scores from the Ground Motion Classification (GMC) step.

**Implementation**: 
- Checks for presence of `score_X` values (same across all components)
- Records with NaN score values are removed

**Bypass**: Records in the bypass list skip this filter and are retained regardless of missing GMC data.

**Typical Removals**: Records that failed GMC processing due to waveform issues or insufficient data for feature extraction.

---

#### 3. Filter by Score Mean

**Purpose**: Removes records with low-quality waveforms based on GMC quality scores.

**Configuration**: 
- Default threshold: `score_min = 0.5` (configurable in `config.yaml`)
- Can be overridden by function parameter

**Implementation**:
- Examines `score_X` and `score_Y` components by default
- Records where either horizontal component falls below threshold are removed
- Vertical component (`score_Z`) can optionally be included, but is not checked by default

**Bypass**: Records in bypass list are retained regardless of score values.

**Adjustable Parameters**:
```python
# In config.yaml
score_min: 0.5  # Minimum quality score threshold

# Can be adjusted when calling apply_all_filters()
catalogue, skipped = apply_all_filters(catalogue, clipped_records_ffp, 
                                   score_min=0.7)  # Custom threshold
```

---

#### 4. Filter by Multi Mean

**Purpose**: Removes records with higher chance of multi-earthquakes.

**Configuration**: 
- Default threshold: `multi_max = 0.2` (configurable in `config.yaml`)
- Can be overridden by function parameter

**Implementation**:
- Examines `multi_mean_X` and `multi_mean_Y` components by default
- Records where either horizontal component exceeds threshold are removed
- Vertical component can optionally be included, but is not checked by default

**Bypass**: Records in bypass list are retained regardless of multi mean values.

**Adjustable Parameters**:
```python
# In config.yaml
multi_max: 0.2  # Maximum multi-component inconsistency threshold

# Runtime adjustment
catalogue, skipped = apply_all_filters(catalogue, clipped_records_ffp,
                                   multi_max=0.15)  # Stricter filtering
```

---

#### 5. Filter by Fmax

**Purpose**: Ensures records have sufficient high-frequency content for analysis.

**Configuration**: 
- Default threshold: `fmax_min = 4.1` Hz (configurable in `config.yaml`)
- Can be overridden by function parameter

**Implementation**:
- Checks `fmax_X` and `fmax_Y` values by default
- Records where either horizontal component exceeds threshold are removed
- Vertical component can optionally be included, but is not checked by default

**Bypass**: Records in bypass list are retained regardless of fmax values.

**Adjustable Parameters**:
```python
# In config.yaml
fmax_min: 4.1  # Minimum maximum usable frequency (Hz)

# Runtime adjustment for higher frequency applications
catalogue, skipped = apply_all_filters(catalogue, clipped_records_ffp,
                                   fmax_min=8.0)  # Require higher frequencies
```
---

#### 6. Filter by Fmin

**Purpose**: Removes records where the minimum usable frequency is too high, indicating poor low-frequency signal quality.

**Configuration**: 
- Default threshold: `fmin_max = 2.0` Hz (configurable in `config.yaml`)
- Can be overridden by function parameter

**Implementation**:
- Examines `fmin_mean_X` and `fmin_mean_Y` components by default
- Records where either horizontal component exceeds threshold are removed
- Vertical component can optionally be included, but is not checked by default

**Bypass**: Records in bypass list are retained regardless of fmin values.

**Adjustable Parameters**:
```python
# In config.yaml
fmin_max: 2.0  # Maximum minimum usable frequency (Hz)

# Runtime adjustment for low-frequency studies
catalogue, skipped = apply_all_filters(catalogue, clipped_records_ffp,
                                   fmin_max=1.0)  # Require better low-frequency content
```

---

#### 7. Filter by Missing Station Information

**Purpose**: Removes records from stations lacking essential metadata for ground motion studies.

**Implementation**:
- Checks for missing values in station parameters:
  - Vs30 measurements (`Vs30`) - If this is missing then the other important site metrics will not be available

**Bypass**: Records in bypass list are retained even with missing station information.

**Adjustable Parameters**: This filter uses fixed criteria but can be bypassed entirely for specific records.

**Physical Meaning**: Station metadata is essential for ground motion prediction equations and site response analysis.

---

#### 8. Filter by Ground Level Locations

**Purpose**: Ensures only surface or near-surface recordings are included.

**Implementation**:
- Examines the `loc` (location) field in record identifiers
- Filters out specific location codes that indicate non-ground-level installations
- Retains records with the closest ground level location code (typically "20", "21", etc.) for each event/site/channel pairing.

**Bypass**: Records in bypass list are retained regardless of location code.

---

#### 9. Filter by Clipped Records

**Purpose**: Removes records identified as clipped during the Waveform Extraction processing step.

**Implementation**:
- Reads the `clipped_records.csv` file created during Waveform Extraction step
- Removes any records that appear in the clipped records list reason
- Clipping detection is based on configurable magnitude and distance thresholds in the Waveform Extraction Step.

**Configuration**:
```yaml
# Clipping detection parameters in config.yaml
mag_clip_low: 3.0       # Minimum magnitude for clipping assessment
mag_clip_high: 8.8      # Maximum magnitude for clipping assessment  
dist_clip_low: 0.0      # Minimum distance for clipping assessment
dist_clip_high: 645.0   # Maximum distance for clipping assessment
clip_threshold: 0.2     # Threshold for clipping detection
```

**Bypass**: Records in bypass list are retained even if identified as clipped.

---

#### 10. Filter by Jerk Records

**Purpose**: Removes records where it was identified that for some number of points the Jerk exceeded the median by 100 times for at least 1 trace during the Waveform Extraction processing step.

**Implementation**:
- Reads the `clipped_records.csv` file created during Waveform Extraction step
- Removes any records that appear in the Jerk records list reason

**Bypass**: Records in bypass list are retained even if identified as jerk records.

---

#### 11. Filter by Sensitivity Ignore List

**Purpose**: Removes records known to be problematic in the BroadBand sensors, such as early deployments with potential calibration errors.

**Implementation**:
- Loads a pre-defined ignore list from `sensitivity_ignore.csv` in the data registry.
- Matches records by `sta`, `chan`, `loc` and checks if their timestamp falls within the specified `start_date`–`end_date` range.

**Filtering Logic**:
- Removes matching records unless they are part of the bypass list.

**Bypass**: Records listed in the bypass list are not filtered, even if matched in the ignore file.

---

#### 12. Filter by Empirical Prediction Residuals

**Purpose**: Removes records with ground motion values significantly inconsistent with empirical ground motion prediction models.

**Implementation**:
- Uses the **Atkinson (2022)** GMM to compute predicted pSA values for records, based on magnitude, distance, Vs30, and other metadata.
- Separates records into tectonic types (Interface, Slab, Crustal) for model application.
- Computes:
  - **Mean residual**: Mean total residual across pSA periods (0.01–10.0s)
  - **Max residual**: Max total residual across pSA periods (0.01–10.0s)

**Thresholds**:
- `mean_residual_threshold` and `max_residual_threshold` can be set in the configuration file or passed explicitly to the function.

**Filtering Logic**:
- Records are removed if either:
  - Mean residual exceeds the configured threshold, or
  - Max residual exceeds the configured threshold
- Records in the bypass list are not filtered regardless of their residuals.

**Bypass**: All residual filters are skipped for records explicitly listed in the bypass array.

**NZGMDB Configuration**:
```yaml
mean_residual_threshold: 4
max_residual_threshold: 6
```

---

#### 13. Filter Duplicate Channels

**Purpose**: Retains the highest-priority record when multiple instruments record the same event at the same station.

**Implementation**:
- Combines event ID and station name into a unique `evid_sta` identifier.
- Assigns priority levels to records based on channel type and bypass status.
- Retains only the highest-priority channel per `evid_sta` group.

**Channel Priority Order**:
1. **Bypass records** (priority = 0)
2. **HN channels** (Strong motion sensors, high frequency response) – priority = 1  
3. **BN channels** (Strong motion sensors, lower frequency response) – priority = 2  
4. **HH channels** (Broadband sensors, high frequency) – priority = 3  
5. **All other channels** are discarded before selection.

**Selection Logic**:
- Bypass records are always kept regardless of channel type.
- Within each group of duplicate records (same `evid_sta`), the record with the **lowest priority score** is retained.
- Channels not in the HN, BN, or HH categories are excluded **prior** to duplicate resolution.

**Bypass**: Records in bypass list override all priority rules.

### 🔹 Configuration Parameters

Key parameters from `config.yaml` that control quality filtering:

```yaml
# Quality filtering thresholds
score_min: 0.5      # Minimum GMC quality score
multi_max: 0.2      # Maximum multi-component inconsistency
fmax_min: 4.1       # Minimum maximum usable frequency (Hz)
fmin_max: 2.0       # Maximum minimum usable frequency (Hz)

# Clipping detection parameters
mag_clip_low: 3.0
mag_clip_high: 8.8
dist_clip_low: 0.0
dist_clip_high: 645.0
clip_threshold: 0.2
```

### 🔹 Runtime Customisation

All filter thresholds can be adjusted when calling the function directly:

```python
from nzgmdb.data_processing.quality_db import apply_all_filters

# Custom filtering with stricter criteria
filtered_catalog, skipped_records = apply_all_filters(
    catalogue=input_catalog,
    clipped_records_ffp=clipped_file_path,
    bypass_records=custom_bypass_list,
    score_min=0.7,      # Stricter quality requirement
    multi_max=0.15,     # Lower inconsistency tolerance
    fmax_min=6.0,       # Higher frequency requirement
    fmin_max=1.5        # Better low-frequency requirement
)
```

### 🔹 Flatfile Filtering

After the rotd50 flatfile is filtered, the step then filters the rest of the flatfiles with the same record_ids as the rotd50 flatfile, ensuring consistency across all metadata tables. This means that any record removed from the rotd50 flatfile will also be removed from the other flatfiles, maintaining a consistent dataset.

---

## 📦 Output

### 🔹 Primary Output: Quality Database Directory

**`quality_db/`** directory containing filtered flatfiles:

The quality database creates filtered versions of all flatfiles, ensuring consistency across the entire dataset:

| File                                      | Description                                                                   |
|-------------------------------------------|-------------------------------------------------------------------------------|
| `ground_motion_im_rotd50_flat.csv`        | RotD50 filtered intensity measure catalogue with quality metrics                |
| `ground_motion_im_table_000_flat.csv`     | 000 component filtered intensity measure catalogue with quality metrics         |
| `ground_motion_im_table_090_flat.csv`     | 090 component filtered intensity measure catalogue with quality metrics         |
| `ground_motion_im_table_ver_flat.csv`     | Vertical component filtered intensity measure catalogue with quality metrics  n |
| `ground_motion_im_table_rotd0_flat.csv`   | RotD0 filtered intensity measure catalogue with quality metrics                 |
| `ground_motion_im_table_rotd100_flat.csv` | RotD100 filtered intensity measure catalogue with quality metrics               |
| `ground_motion_im_table_EAS_flat.csv`     | EAS filtered intensity measure catalogue with quality metrics                   |
| `ground_motion_im_table_geom_flat.csv`    | Geometric mean filtered intensity measure catalogue with quality metrics        |
| `earthquake_source_table.csv`             | Event source data for remaining records                                       |
| `earthquake_source_geometry.csv`          | Fault geometry: strike, dip, rake, corner coordinates                         |
| `site_table.csv`                          | Station metadata for sites with quality records                               |
| `propagation_path_table.csv`              | Distance metrics for filtered record set                                      |
| `phase_arrival_table.csv`                 | P and S-wave arrival times and probabilities from PhaseNet                    |
| `fmax.csv`                                | Maximum frequency data for quality records                                    |
| `gmc_predictions.csv`                     | GMC quality predictions for final dataset                                     |
| `station_magnitude_table.csv`             | Station-specific magnitude calculations per event-channel pair                |
| `snr_metadata.csv`                        | SNR metadata per waveform such as Ds, Dn, dt etc.                             |

### 🔹 Secondary Output: Quality Tracking

**`flatfiles/quality_skipped_records.csv`** documenting all removed records:

| Column | Description |
|--------|-------------|
| `record_id` | Unique identifier for the skipped record |
| `reason` | Specific reason for removal (e.g., "Score mean is less than 0.5") |

---

## 🔧 Technical Implementation

### 🔹 Core Functions

The Quality-DB step utilises several key functions from the `quality_db.py` module:

- **`create_quality_db()`**: Main orchestration function
- **`apply_all_filters()`**: Comprehensive filtering pipeline
- **`filter_has_score_mean()`**: GMC prediction presence check
- **`filter_score_mean()`**: Quality score filtering
- **`filter_multi_mean()`**: Multi-component consistency filtering
- **`filter_fmax()`**: Maximum frequency filtering
- **`filter_fmin()`**: Minimum frequency filtering
- **`filter_missing_sta_info()`**: Station metadata validation
- **`filter_ground_level_locations()`**: Location code filtering
- **`apply_clipNet_filter()`**: Clipped record removal
- **`filter_duplicate_channels()`**: Channel prioritisation and deduplication

### 🔹 Bypass Record Integration

The bypass mechanism allows users to retain specific records that would otherwise be filtered out:

- Accepts a CSV file with `record_id` column
- Bypass records skip all quality checks
- In duplicate channels, bypass records receive highest priority
- Useful for retaining scientifically important records or manual overrides

---

## 🔗 Related Steps

- **Previous**: [Merge Flatfiles](Merge-Flatfiles.md) - Provides the consolidated IM catalogue that serves as input for quality filtering
- **Optional**: [Upload to Dropbox](Upload-Dropbox.md) - Packages and uploads the all information for distribution