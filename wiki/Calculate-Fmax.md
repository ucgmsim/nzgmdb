# 📊 Calculate Fmax

This step computes the maximum usable frequency (Fmax) for each seismic record by analyzing Signal-to-Noise Ratio (SNR) data.

---

## 🚀 Entry Point

To calculate Fmax, run the following Python script:

```bash
python -m nzgmdb.scripts.run_nzgmdb calc-fmax <main_dir>
```

- `<main_dir>` is the top-level output directory where NZGMDB stores its results.

Example:
```bash
python -m nzgmdb.scripts.run_nzgmdb calc-fmax nzgmdb_output/
```

Optional parameters include:
- `--meta-output-dir`: Custom directory for metadata and skipped records output
- `--waveform-dir`: Custom directory containing mseed files
- `--snr-fas-output-dir`: Custom directory for SNR and FAS data
- `--n-procs`: Number of processes to use for parallel processing (default: 1)
- `--bypass-records-ffp`: Path to bypass records file for custom Fmax values

---

## 📋 Prerequisites

- **Calculate SNR**: SNR and FAS files must be generated beforehand
- **SNR metadata**: The `snr_metadata.csv` file containing processing information for each record

---

## ⚙️ Process

### 🔹 Initial Filtering and Smoothing

For each record in the SNR metadata table, the algorithm performs the following steps:

**1. Calculate Scaled Nyquist Frequency**
$$scaled f_s = \frac{1}{\delta} \cdot 0.5 \cdot 0.8$$
Where ${\delta}$ is the time difference between each data point in the waveform.

**2. Smooth SNR Values**
- Applies a rolling window smoothing with a window size of 5 frequency points
- Uses centre-aligned averaging with a minimum of 1 observation per window
- This reduces noise in the SNR frequency spectrum

**3. Initial Quality Screening**
Performs a quality check requiring:
- At least **5 frequency points** between 0.5 and 10 Hz
- SNR values must exceed **3.0** for vertical component (ver)
- SNR values must exceed **5.0** for horizontal components (000, 090)

### 🔹 Fmax Calculation

If the quality screening passes:

**1. Filter High Frequencies**
- Selects SNR values for frequencies greater than **4 Hz** (configurable via `min_freq_Hz`)

**2. Find Fmax Threshold**
- Identifies the first frequency point where SNR drops below the threshold of **3.0**
- If no frequency points have SNR > threshold, Fmax is set to the minimum of:
  - The scaled Nyquist frequency
  - The last available frequency point

**3. Component-wise Processing**
- Calculates Fmax independently for each component (000, 090, ver)
- Uses consistent thresholds across all components

### 🔹 Fallback Handling

If quality screening fails:
- Sets Fmax to the scaled Nyquist frequency for all components
- Records the reason for skipping in the skipped records file
- Ensures all records have valid Fmax values for downstream processing

---

## 📦 Output

### 🔹 Primary Output: `fmax.csv`

Located in the flatfiles directory with the following columns:

| Column      | Description                                          |
|-------------|------------------------------------------------------|
| `record_id` | Unique identifier (evid_station_channel_location)   |
| `fmax_000`  | Maximum usable frequency for 000 component (Hz)     |
| `fmax_090`  | Maximum usable frequency for 090 component (Hz)     |
| `fmax_ver`  | Maximum usable frequency for ver component (Hz)     |

### 🔹 Secondary Output: `fmax_skipped_records.csv`

Contains records that could not be processed:

| Column      | Description                                          |
|-------------|------------------------------------------------------|
| `record_id` | Unique identifier for the skipped record            |
| `reason`    | Detailed explanation of why the record was skipped  |

Example skip reasons include insufficient frequency points above SNR thresholds in the 0.5-10 Hz interval.

---

## ⚙️ Configuration Parameters

Key parameters from `config.yaml` that control Fmax calculation:

### 🔹 Frequency Analysis
- `nyquist_freq_scaling_factor`: 0.8 (scales Nyquist frequency)
- `min_freq_Hz`: 4.0 (minimum frequency for Fmax search)
- `snr_thresh`: 3.0 (SNR threshold for Fmax determination)

### 🔹 Quality Screening
- `initial_screening_min_freq_Hz`: 0.5 (start of screening interval)
- `initial_screening_max_freq_Hz`: 10.0 (end of screening interval)
- `initial_screening_snr_thresh_ver`: 3.0 (vertical component threshold)
- `initial_screening_snr_thresh_horiz`: 5.0 (horizontal component threshold)
- `initial_screening_min_points_above_thresh`: 5 (minimum valid points required)

### 🔹 Smoothing Parameters
- `window`: 5 (rolling window size for SNR smoothing)
- `centre`: True (centre-aligned smoothing)
- `min_periods`: 1 (minimum observations per window)

---
## 🔗 Related Steps

- **Previous**: [Calculate SNR](Calculate-SNR.md) - Provides the SNR and FAS data required for Fmax computation
- **Next**: [GMC](GMC.md) - Uses MSEED files for fmin and score classification
- **Related**: [Process Records](Process-Records.md) - Combines Fmax with GMC predictions for waveform filtering