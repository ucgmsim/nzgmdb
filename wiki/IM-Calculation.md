# 📊 IM Calculation

This step computes Intensity Measures (IMs) for seismic waveform records processed through the NZGMDB pipeline. The implementation calculates various ground motion metrics including peak ground acceleration, spectral response, and frequency-domain characteristics for observed waveforms.

---

## 🚀 Entry Point

To calculate intensity measures for all processed waveform data, run the following Python script:

```bash
python -m nzgmdb.scripts.run_nzgmdb run-im-calculation <main_dir> <ko_directory>
```

- **<main_dir>** is the top-level output directory where NZGMDB stores its results
- **<ko_directory>** is the path to the directory containing Konno-Ohmachi smoothing matrices

Example:
```bash
python -m nzgmdb.scripts.run_nzgmdb run-im-calculation nzgmdb_output/ /path/to/ko_matrices/
```

Optional parameters include:
- `--output-dir`: Custom directory for IM output files (default: main_dir/IM/)
- `--n-procs`: Number of processes for parallel processing (default: 1)
- `--checkpoint`: Skip already completed files for incremental processing
- `--intensity-measures`: Comma-separated list of specific IMs to calculate, defaults to all configured IMs in `config.yaml`

---

## 📋 Prerequisites

The IM Calculation step requires the following inputs from previous pipeline steps:
- **[Process Records](Process-Records.md)** (generates .000, .090, and .ver processed waveform files)

The step requires processed waveform text files (.000, .090, .ver) in the `waveforms/` directory.

---

## ⚙️ Process

### 🔹 Waveform Input Processing

The system reads processed ASCII waveform files for each record:

**File Structure Expected:**
- `.000` file: North-South (000°) horizontal component
- `.090` file: East-West (090°) horizontal component  
- `.ver` file: Vertical component

**Quality Control:**
- Records are skipped if any component file is missing
- All three components must be available for successful IM calculation
- Sampling rate (dt) and waveform arrays are extracted from the ASCII files

### 🔹 Intensity Measures Computed

The pipeline calculates the following intensity measures based on configuration settings:

#### Time-Domain IMs
- **PGA** - Peak Ground Acceleration
- **PGV** - Peak Ground Velocity  
- **CAV** - Cumulative Absolute Velocity
- **CAV5** - Cumulative Absolute Velocity (threshold: 5 cm/s²)
- **AI** - Arias Intensity
- **Ds575** - Duration containing 5-75% of Arias Intensity
- **Ds595** - Duration containing 5-95% of Arias Intensity

#### Frequency-Domain IMs
- **pSA** - Pseudo-Spectral Acceleration (111 periods)
- **FAS** - Fourier Amplitude Spectrum (389 frequencies)

### 🔹 Spectral Acceleration Periods

The pSA calculations use **111 standardized periods** (in seconds):

```
0.010, 0.020, 0.022, 0.025, 0.029, 0.030, 0.032, 0.035, 0.036, 0.040,
0.042, 0.044, 0.045, 0.046, 0.048, 0.050, 0.055, 0.060, 0.065, 0.067,
0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100, 0.110, 0.120, 0.130,
0.133, 0.140, 0.150, 0.160, 0.170, 0.180, 0.190, 0.200, 0.220, 0.240,
0.250, 0.260, 0.280, 0.290, 0.300, 0.320, 0.340, 0.350, 0.360, 0.380,
0.400, 0.420, 0.440, 0.450, 0.460, 0.480, 0.500, 0.550, 0.600, 0.650,
0.667, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 1.000, 1.100, 1.200,
1.300, 1.400, 1.500, 1.600, 1.700, 1.800, 1.900, 2.000, 2.200, 2.400,
2.500, 2.600, 2.800, 3.000, 3.200, 3.400, 3.500, 3.600, 3.800, 4.000,
4.200, 4.400, 4.600, 4.800, 5.000, 5.500, 6.000, 6.500, 7.000, 7.500,
8.000, 8.500, 9.000, 9.500, 10.000, 11.000, 12.000, 13.000, 14.000, 
15.000, 20.000
```

### 🔹 FAS Frequency Vector

Fourier Amplitude Spectrum calculations use **389 logarithmically spaced frequencies**:
- **Range**: 0.01318257 → 100 Hz
- **Distribution**: Logarithmic spacing for optimal frequency resolution

### 🔹 Component Analysis

IMs are calculated for multiple component orientations:

| Component | Description |
|-----------|-------------|
| **000** | North-South horizontal |
| **090** | East-West horizontal |
| **ver** | Vertical |
| **geom** | Geometric mean horizontal |
| **rotd0** | Minimum rotated horizontal |
| **rotd50** | Median rotated horizontal |
| **rotd100** | Maximum rotated horizontal |

**Note:** For FAS calculations, an additional **EAS** component is computed specifically.

### 🔹 Parallel Processing

The step utilizes multiprocessing for computational efficiency:
- Distributes IM calculations across available CPU cores
- Processes multiple records simultaneously
- Optimized memory usage for large waveform datasets

---

## 📦 Output

### 🔹 Individual IM Files

Primary output consists of CSV files stored in the directory structure:
```
IM/
└── event_id/
    └── evid_station_channel_location_IM.csv
```

Each CSV file contains all calculated IMs with the following structure:

**File Format**: One row per component (8 rows total), with 509 columns containing all IM data.

| Column Type | Example Columns                                     | Description |
|-------------|-----------------------------------------------------|-------------|
| **Identifiers** | `record_id`, `component`                            | Record identifier and component type |
| **Time-Domain IMs** | `PGA`, `PGV`, `CAV`, `CAV5`, `AI`, `Ds575`, `Ds595` | Peak and duration-based measures |
| **Spectral Acceleration** | `pSA_0.01`, `pSA_0.1`, `pSA_1.0`, `pSA_20.0`        | Response spectra at 111 periods |
| **Fourier Spectra** | `FAS_0.013182570000000001`, `FAS_1.0`, `FAS_100.0`  | Amplitude spectra at 389 frequencies |

**Component Rows Structure**:
- Row 1-8: Each represents a different component (000, 090, ver, geom, rotd0, rotd50, rotd100, EAS)
- All IM values for that component are stored in the same row
- Period-specific values follow naming: `pSA_{period}` (e.g., `pSA_0.1` for 0.1-second period)
- Frequency-specific values follow naming: `FAS_{frequency}` (e.g., `FAS_1.0` for 1.0 Hz)

### 🔹 Component Coverage Table

| Intensity Measure | 000 | 090 | ver | geom | rotd0 | rotd50 | rotd100 | EAS |
|-------------------|-----|-----|-----|------|---|----|---------|-----|
| **PGA** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **PGV** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **CAV** | ✓ | ✓ | ✓ | ✓ | | | | |
| **CAV5** | ✓ | ✓ | ✓ | ✓ | | | | |
| **AI** | ✓ | ✓ | ✓ | ✓ | | | | |
| **Ds575** | ✓ | ✓ | ✓ | ✓ | | | | |
| **Ds595** | ✓ | ✓ | ✓ | ✓ | | | | |
| **pSA** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **FAS** | ✓ | ✓ | ✓ | | | | | ✓ |

### 🔹 Metadata Files

Additional output files are generated in the `flatfiles/` directory:

- **`IM_calc_skipped_records.csv`**: Contains records that could not be processed due to missing component files

| Column | Description |
|--------|-------------|
| `record_id` | Unique identifier (evid_station_channel_location) |
| `reason` | Detailed explanation of processing failure |

---

## ⚙️ Configuration Parameters

Key configuration values from `config.yaml`:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `ims` | See list above | Intensity measures to calculate |
| `psa_periods` | 111 periods | Spectral acceleration periods (seconds) |
| `common_frequency_start` | 0.01318257 | FAS start frequency (Hz) |
| `common_frequency_end` | 100 | FAS end frequency (Hz) |
| `common_frequency_num` | 389 | Number of FAS frequency points |
| `ko_bandwidth` | 40 | Konno-Ohmachi smoothing bandwidth |

---

## 🔍 Implementation Details

### 🔹 Observed Waveforms

For earthquake records from the NZGMDB:
- Uses processed ASCII waveform files (.000, .090, .ver)
- Applies rotational processing for RotD components
- Implements Konno-Ohmachi smoothing for FAS calculations
- Handles missing components gracefully with detailed error logging

### 🔹 Quality Assurance

- **File Validation**: Ensures all three component files exist before processing
- **Error Handling**: Logging of processing failures
- **Checkpoint Support**: Allows resumption of interrupted calculations

---

## 🔄 Merge IM Results

After individual IM calculations are complete, the results must be consolidated into a single comprehensive dataset. This subsequent step merges all individual IM CSV files and applies quality filtering.

### 🚀 Merge Entry Point

To merge all calculated IM results, run:

```bash
python -m nzgmdb.scripts.run_nzgmdb merge-im-results <im_dir> <output_dir>
```

- **<im_dir>** is the directory containing individual IM CSV files (typically main_dir/IM/)
- **<output_dir>** is the output directory for merged results (typically main_dir/flatfiles/)

Example:
```bash
python -m nzgmdb.scripts.run_nzgmdb merge-im-results nzgmdb_output/IM/ nzgmdb_output/flatfiles/
```

Optional parameters:
- `--gmc-ffp`: Path to GMC predictions file for quality score integration
- `--fmax-ffp`: Path to Fmax results file for frequency limit integration

### 🔹 Merge Process

The merge operation performs the following consolidation steps:

**1. IM File Aggregation**
- Recursively finds all `*_IM.csv` files in the IM directory
- Concatenates all individual record IM files into a single DataFrame
- Preserves all calculated intensity measures and component orientations

**2. Metadata Integration**
- **GMC Data**: Merges quality scores, minimum frequencies (fmin), and multi-component scores
- **Fmax Data**: Integrates maximum usable frequencies for each component
- **Record Parsing**: Extracts event ID, station, channel, and location from record identifiers

**3. Column Organization**
Structures the final dataset with logical column ordering:
```
record_id | evid | sta | loc | chan | component | 
PGA | PGV | CAV | CAV5 | AI | Ds575 | Ds595 |
score_mean_X | fmin_mean_X | fmax_mean_X | multi_mean_X |
score_mean_Y | fmin_mean_Y | fmax_mean_Y | multi_mean_Y |
score_mean_Z | fmin_mean_Z | fmax_mean_Z | multi_mean_Z |
[pSA_periods...] | [FAS_frequencies...]
```

### 🔹 Merge Output

**`ground_motion_im_catalogue.csv`**

The consolidated catalog contains:

| Field Group | Description | Example Columns |
|-------------|-------------|-----------------|
| **Identifiers** | Record and location info | `record_id`, `evid`, `sta`, `chan`, `loc` |
| **Components** | Motion component type | `component` (000, 090, ver, rotd50, etc.) |
| **Time-Domain IMs** | Peak and duration metrics | `PGA`, `PGV`, `CAV`, `AI`, `Ds575`, `Ds595` |
| **Quality Metrics** | GMC and frequency data | `score_mean_X`, `fmin_mean_X`, `fmax_mean_X` |
| **Spectral IMs** | Period-dependent response | `pSA_0.010`, `pSA_0.050`, `pSA_1.000` |
| **Frequency IMs** | Fourier amplitude data | `FAS_0.100`, `FAS_1.000`, `FAS_10.000` |

### 🔹 Integration Benefits

- **Single Source**: Consolidates thousands of individual files into one analyzable dataset
- **Metadata Enrichment**: Combines IM data with quality metrics and processing parameters


---

## 🔗 Related Steps

- **Previous**: [Process Records](Process-Records.md) - Generates the processed waveform files required for IM calculation
- **Next**: [Calculate Distances](Calculate-Distances.md) - Computes rupture distance metrics for the merged IM catalog
- **Related**: [Merge Flatfiles](Merge-Flatfiles.md) - Further processes the IM catalog into component-specific flatfiles