# 🔗 Merge Flatfiles

This step in the NZGMDB pipeline consolidates all individual flatfiles into final component-specific datasets while ensuring data consistency and removing filtered entries. **This critical step transforms the comprehensive intensity measure catalog into organized, analysis-ready flatfiles** that separate results by motion component and integrate metadata from earthquake source, site, and propagation tables.

---

## 🚀 Entry Point

To merge all flatfiles into final component-specific datasets, run:

```bash
python -m nzgmdb.scripts.run_nzgmdb merge-flat-files <main_dir> [--bypass-records-ffp PATH]
```

**Parameters:**
- **main_dir**: The top-level output directory where NZGMDB stores its results
- **bypass-records-ffp**: Optional path to bypass records file with custom fmin, fmax, and p_wave_ix values

**Example:**
```bash
python -m nzgmdb.scripts.run_nzgmdb merge-flat-files nzgmdb_output/
```

---

## 📋 Prerequisites

The Merge Flatfiles step requires the following inputs from previous pipeline steps:
- **[Calculate Aftershocks](Calculate-Aftershocks.md)** - Generates the earthquake source table with aftershock classifications
- **[Calculate Distances](Calculate-Distances.md)** - Provides the distance calculations and the geometry table containing rupture polygons
- **[Merge IM Results](Merge-IM-Results.md)** - Generates the ground_motion_im_catalogue.csv file

---

## ⚙️ Process Overview

The Merge Flatfiles step performs **four main consolidation functions**:

### 🔹 1. Data Quality Filtering

Ensures that only events and records that successfully passed all pipeline filtering steps are included in the final flatfiles:

**Quality Assurance:**
- Cross-references records in the ground_motion_im_catalogue with all supporting metadata tables
- Removes events that lack complete intensity measure data
- Validates that all records have corresponding site, source, and propagation information
- Documents missing sites in a separate output file for quality tracking

### 🔹 2. Component Separation

Splits the comprehensive ground_motion_im_catalogue into component-specific tables:

**Component Organization:**
- **000**: North-South horizontal component
- **090**: East-West horizontal component  
- **ver**: Vertical component
- **rotd0**: Minimum rotated horizontal component
- **rotd50**: Median rotated horizontal component
- **rotd100**: Maximum rotated horizontal component
- **EAS**: Effective Amplitude Spectrum (for FAS calculations)
- **geom**: Geometric mean horizontal component

### 🔹 3. Metadata Integration

Creates comprehensive "flat" versions that merge intensity measure data with supporting metadata:

**Integrated Metadata Sources:**
- **Earthquake Source Data**: Event magnitude, location, datetime, tectonic domain
- **Site Information**: Vs30, basin depths, station coordinates, site class
- **Propagation Metrics**: Distance measures (rrup, rjb, rx, ry) between source and site
- **Quality Scores**: GMC predictions, fmin/fmax values, multi-component scores
- **Processing Metadata**: SNR information, phase arrival times, component-specific quality flags

### 🔹 4. Location Code Integration

Enhances station metadata by merging location codes from GeoNet to analyse ground level recorders when we have many of the same station-event pairings with multiple location codes:

**Location Addition:**
- Adds the locations specific elevation adjustment from the site locations recorded elevation

---

## 📦 Output Files

### 🔹 Metadata Flatfiles

| File | Description |
|------|-------------|
| `earthquake_source_table.csv` | Event metadata: magnitude, location, datetime, tectonic classification |
| `earthquake_source_geometry.csv` | Fault geometry: strike, dip, rake, corner coordinates |
| `station_magnitude_table.csv` | Station-specific magnitude calculations per event-channel pair |
| `phase_arrival_table.csv` | P and S-wave arrival times and probabilities from PhaseNet |
| `site_table.csv` | Site characteristics: Vs30, basin depths, coordinates, site classification |
| `propagation_path_table.csv` | Distance metrics: rrup, rjb, rx, ry for each station-event pair |

### 🔹 Component-Specific Intensity Measure Flat Tables

| File | Description |
|------|-------------|
| `ground_motion_im_table_000_flat.csv` | 000 component IMs + complete metadata integration |
| `ground_motion_im_table_090_flat.csv` | 090 component IMs + complete metadata integration |
| `ground_motion_im_table_ver_flat.csv` | Vertical component IMs + complete metadata integration |
| `ground_motion_im_table_rotd0_flat.csv` | RotD0 IMs + complete metadata integration |
| `ground_motion_im_table_rotd50_flat.csv` | RotD50 IMs + complete metadata integration |
| `ground_motion_im_table_rotd100_flat.csv` | RotD100 IMs + complete metadata integration |
| `ground_motion_im_table_EAS_flat.csv` | EAS IMs + complete metadata integration |
| `ground_motion_im_table_geom_flat.csv` | Geometric mean IMs + complete metadata integration |

### 🔹 Quality Control Outputs

| File | Description                                                                  |
|------|------------------------------------------------------------------------------|
| `missing_sites.csv` | Sites with IM data but missing metadata (to be filtered in quality database) |

---

## 📊 Flat Table Columns

The "flat" versions of intensity measure tables contain comprehensive metadata integration with the following column groups and names with descriptions:

### 🔹 Record Identification
- `record_id` - Unique identifier (evid_station_channel_location)
- `evid` - Event identifier
- `sta` - Station code
- `chan` - Channel code
- `loc` - Location code
- `component` - Motion component (000, 090, ver, etc.)

### 🔹 Event Source Metadata
- `datetime` - Event origin time
- `mag` - Event magnitude
- `mag_type` - Magnitude type (Ml, Mw, etc.)
- `ev_lat`, `ev_lon`, `_evdepth` - Event location
- `strike`, `dip`, `rake`, `f_length`, `f_width`, `z_tor`, `z_bor` - Fault plane parameters
- `f_type` - Fault type (FF, CMT, CMT_UNC, domain)
- `reloc` - Defines if the event is relocated
- `domian_no` - Tectonic domain number
- `domain_type` - Type of tectonic domain
- `tect_type` - Tectonic classification
- `aftershock_flag_[0, 2, 5, 10]` - Aftershock identification for each distance cutoff
- `cluster_flag_[0, 2, 5, 10]` - Cluster identifier for spatial grouping for each distance cutoff

### 🔹 Site Characteristics
- `sta_lat`, `sta_lon`, `sta_elev` - Station coordinates and elevation
- `loc_elev` - Location-specific elevation adjustment
- `is_ground_level` - Indicates if the station is a ground-level recorder
- `vs30`, `vs30_std`, `Q_Vs30` - Shear wave velocity to 30m with std and quality
- `Z1.0`, `Z1.0_std`, `Q_Z1.0`, `Z2.5`, `Z2.5_std`, `Q_Z2.5` - Basin depths to 1.0 and 2.5 km/s velocity layers with std and quality
- `T0`, `T0_std`, `Q_T0` - Site fundamental period with std and quality
- `site_domain_no` - Site Domain number

### 🔹 Propagation Distances
- `r_epi` - Epicentral distance (km)
- `r_hyp` - Hypocentral distance (km)
- `r_jb` - Joyner-Boore distance (km)
- `r_rup` - Closest distance to rupture plane (km)
- `r_avg` - Average distance to rupture plane (km)
- `r_x` - Distance perpendicular to strike (km)
- `r_y` - Distance parallel to strike (km)
- `r_tvz` - Path length through Taupo VZ (km)
- `r_xvf` - Distance to Taupo VZ boundary  (km)

### 🔹 Quality Metrics
- `score_X`, `score_Y`, `score_Z` - GMC quality scores per component
- `fmin_X`, `fmin_Y`, `fmin_Z` - Minimum usable frequencies (Hz)
- `fmax_X`, `fmax_Y`, `fmax_Z` - Maximum usable frequencies (Hz)
- `multi_X`, `multi_Y`, `multi_Z` - Multi-Earthquake scores

### 🔹 Processing Metadata
- `HPF_h`, `LPF_h` - High-pass and low-pass filter frequencies for horizontal components (Hz)
- `HPF_v`, `LPF_v` - High-pass and low-pass filter frequencies for vertical component (Hz)

### 🔹 Intensity Measures

Only includes the columns where appropriate IMs were calculated for the component (See the IM Caculation step for details):

**Time-Domain IMs:**
- `PGA` - Peak Ground Acceleration (g)
- `PGV` - Peak Ground Velocity (cm/s)
- `CAV` - Cumulative Absolute Velocity (g·s)
- `CAV5` - CAV with 5 cm/s² threshold (g·s)
- `AI` - Arias Intensity (m/s)
- `Ds575` - Duration 5-75% Arias Intensity (s)
- `Ds595` - Duration 5-95% Arias Intensity (s)

**Spectral Acceleration (111 periods):**
- `pSA_0.01` through `pSA_20.0` - Pseudo-spectral acceleration at specific periods (g)

**Fourier Amplitude Spectrum (389 frequencies):**
- `FAS_0.013` through `FAS_100.0` - Fourier amplitudes at specific frequencies (g·s)

---

## ⚠️ Important Notes

- **Data Consistency**: Only records with complete intensity measure calculations are included in final outputs

---

## 🔗 Related Steps

- **Previous**: [Calculate Aftershocks](Calculate-Aftershocks.md) - Generates earthquake source table with aftershock classifications
- **Next**: [Quality DB](Quality-DB.md) - Applies final filtering and creates delivery-ready quality database
- **Related**: [IM-Calculation](IM-Calculation.md) - Details the intensity measure calculations and which IM's are computed per component