# 📊 IM Calculation

This step computes Intensity Measures (IMs) for seismic waveform records processed through the NZGMDB pipeline. The implementation calculates various ground motion metrics including peak ground acceleration, spectral response, and frequency-domain characteristics for observed waveforms.

---

## 🚀 Entry Point

To calculate intensity measures for all processed waveform data, run the following Python script:

```bash
python -m nzgmdb.scripts.run_nzgmdb run-im-calculation <main_dir>
```

- **<main_dir>** is the top-level output directory where NZGMDB stores its results

Example:
```bash
python -m nzgmdb.scripts.run_nzgmdb run-im-calculation nzgmdb_output/ --n-procs 8
```

Optional parameters include:
- `--n-procs`: Number of processes for parallel processing (default: 1)
- `--batch-size`: Number of records in each batch (default: 5000)
- `--intensity-measures`: Comma-separated list of specific IMs to calculate, defaults to all configured IMs in `config.yaml`

Konno-Ohmachi smoothing for the FAS is done inside [IM_calculation](https://github.com/ucgmsim/IM_calculation), so no smoothing matrices are needed.

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
- **PGD** - Peak Ground Displacement
- **CAV** - Cumulative Absolute Velocity
- **CAV5** - Cumulative Absolute Velocity (threshold: 5 cm/s²)
- **AI** - Arias Intensity
- **Ds575** - Duration containing 5-75% of Arias Intensity
- **Ds595** - Duration containing 5-95% of Arias Intensity

#### Frequency-Domain IMs
- **pSA** - Pseudo-Spectral Acceleration (111 periods)
- **FAS** - Fourier Amplitude Spectrum (389 frequencies)

### 🔹 Spectral Acceleration Periods

The pSA calculations use **111 standardised periods** (in seconds):

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

**Note:** IM_calculation also returns `rotd0_orientation` and `rotd100_orientation` rows, the rotation angle (degrees) of each RotD0 / RotD100 IM. These are saved as their own components.

### 🔹 Batching and Checkpointing

- Records are sorted and split into batches of `--batch-size` records
- The records in a batch are calculated in parallel over `--n-procs` processes
- Each batch saves a parquet file per component, then its skipped records file, which marks the batch as complete
- On a re-run, complete batches are skipped, and batch files from an earlier run with more batches are removed

---

## 📦 Output

### 🔹 Component IM Parquet Files

The IMs are saved as one parquet file per batch for each component in `flatfiles/im_merge_batch_dir/`.
Each `im_merge_<component>` directory is read as a single parquet dataset by the later steps (e.g. `pd.read_parquet(dir)`).

```
flatfiles/im_merge_batch_dir/
├── im_merge_000/batch_0.parquet ...       (000, 090, ver, geom)
├── im_merge_rotd50/batch_0.parquet ...    (rotd0, rotd50, rotd100)
├── im_merge_rotd0_orientation/batch_0.parquet ...   (rotd0_orientation, rotd100_orientation)
├── im_merge_000_fas/batch_0.parquet ...   (000, 090, ver, geom, eas FAS)
└── im_calc_skipped_records/batch_0.csv ...
```

Every table starts with `record_id`, `component`, `evid`, `sta`, `chan`, `loc`, followed by:

| Directory | Components | IM columns |
|-----------|------------|------------|
| `im_merge_<comp>` | 000, 090, ver, geom | `PGA`, `PGV`, `PGD`, `CAV`, `CAV5`, `Ds575`, `Ds595`, `AI`, `pSA_<period>` |
| `im_merge_<comp>` | rotd0, rotd50, rotd100 | `PGA`, `PGV`, `PGD`, `pSA_<period>` |
| `im_merge_<comp>` | rotd0_orientation, rotd100_orientation | `PGA`, `PGV`, `PGD`, `pSA_<period>`, holding the rotation angle (degrees) of the RotD0 / RotD100 value |
| `im_merge_<comp>_fas` | 000, 090, ver, geom, eas | `FAS_<frequency>` (only when FAS is calculated) |

- pSA columns are named `pSA_{period}` (e.g. `pSA_0.1`)
- FAS columns are named `FAS_{frequency}` to 6 significant figures (e.g. `FAS_0.0131826`). Every batch has all the configured frequencies, with NaN above a record's Nyquist frequency

### 🔹 Metadata Files

- **`flatfiles/im_calc_skipped_records.csv`**: The skipped records of all batches combined

| Column | Description |
|--------|-------------|
| `record_id` | Unique identifier (evid_station_channel_location) |
| `reason` | Missing waveform ascii files, or components containing NaN values |

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
| `ko_bandwidth` | 188.5 | Konno-Ohmachi smoothing bandwidth (NGA-West3) |

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
- **Checkpoint Support**: Complete batches are skipped when a run is resumed

---

## 🔗 Related Steps

- **Previous**: [Process Records](Process-Records.md) - Generates the processed waveform files required for IM calculation
- **Next**: [Calculate Distances](Calculate-Distances.md) - Computes rupture distance metrics for the records in the IM results
- **Related**: [Merge Flatfiles](Merge-Flatfiles.md) - Loads each component directory to create the component-specific flatfiles