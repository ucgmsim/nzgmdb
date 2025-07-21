# 📍 Phase Arrival

This step in the NZGMDB pipeline generates P and S wave arrival times for seismic records using PhaseNet, a deep learning model for automated seismic phase picking. PhaseNet processes three-component waveform data to identify P and S wave arrivals with associated confidence probabilities.

---

## 🚀 Entry Point

To generate the phase arrival table, run the following Python script:

```bash
python -m nzgmdb.scripts.run_nzgmdb generate-phase-arrival-table <main_dir> <run_phasenet_script_ffp> <conda_sh> <env_activate_command> --n-procs <n_procs> --bypass-records-ffp <bypass_records_ffp>
```

- **<main_dir>** is the top-level output directory where NZGMDB stores its results
- **<run_phasenet_script_ffp>** is the path to the `run_phasenet.py` script (located in `NZGMDB/phase_arrival/`)
- **<conda_sh>** is the path to your conda.sh script for environment activation
- **<env_activate_command>** is the command to activate the PhaseNet installed environment
- **--n-procs** (optional) is the number of processes to use (default: 1)
- **--bypass-records-ffp** (optional) is the path to a file containing custom P-wave index values for specific records

Example:
```bash
python -m nzgmdb.scripts.run_nzgmdb generate-phase-arrival-table nzgmdb_output/ nzgmdb/phase_arrival/run_phasenet.py ~/.conda/conda.sh "conda activate phasenet" --n-procs 6
```

This will create the files:

```bash
nzgmdb_output/flatfiles/phase_arrival_table.csv
nzgmdb_output/flatfiles/phase_arrival_skipped_records.csv
nzgmdb_output/phase_arrival/batch_*/phase_arrival_table.csv
nzgmdb_output/phase_arrival/batch_*/skipped_records.csv
nzgmdb_output/phase_arrival/batch_*/prob_series.h5
```

## ⚙️ Process

### 🔹 Batch Processing Setup
- Recursively finds all `.mseed` files in the main directory
- Splits the mseed files into even batches based on the number of available processes
- Creates batch directories (`batch_0`, `batch_1`, etc.) under the `phase_arrival` folder based on the number of processes
- Each batch contains a text file listing the mseed files to process

### 🔹 PhaseNet Processing
PhaseNet is a convolutional neural network that processes three-component seismic waveforms to identify P and S wave arrivals:

- **Input Requirements**: Three-component waveform data (vertical, north-south, east-west)
- **Sampling Rate**: PhaseNet requires 100 Hz sampling rate - automatic resampling is performed if needed
- **Processing**: The model outputs probability time series for P and S wave arrivals
- **Peak Detection**: P and S wave arrival times are determined by finding the maximum probability values

### 🔹 Probability Series Storage
- P and S wave probability time series are saved in HDF5 format (`prob_series.h5`)
- Data is compressed using LZF compression and stored as float32 to optimize storage
- Each record's probability series is stored in a separate group within the HDF5 file

### 🔹 Datetime Conversion
- Converts arrival time indices to absolute UTC datetime stamps
- Uses the mseed file's start time and sampling rate to calculate precise arrival times
- Stores both index positions and datetime stamps for P and S waves

### 🔹 Quality Control
Records are skipped if they encounter processing errors such as:
- Zero size arrays after resampling
- Missing or corrupted three-component data
- Issues with processing the record such as missing Inventory information from the FDSN Client or no sensitivity values

### 🔹 Merge with Event Information
- Extracts event ID from record names using the naming convention
- Merges phase arrival results with earthquake source table to add event datetime information
- Final phase arrival table includes both arrival times and event metadata

---

## 📦 Output

The phase arrival processing generates several output files:

### Phase Arrival Table (`phase_arrival_table.csv`)
Contains phase picking results for all successfully processed records:

| Column              | Description                                               |
|--------------------|-----------------------------------------------------------|
| `record_id`        | Unique identifier for the seismic record                 |
| `p_wave_ix`        | Sample index of P wave arrival                           |
| `p_wave_datetime`  | UTC datetime of P wave arrival                           |
| `p_wave_prob`      | PhaseNet confidence probability for P wave pick          |
| `s_wave_ix`        | Sample index of S wave arrival                           |
| `s_wave_datetime`  | UTC datetime of S wave arrival                           |
| `s_wave_prob`      | PhaseNet confidence probability for S wave pick          |
| `evid_datetime`    | Event origin time from earthquake source table           |

### Skipped Records (`phase_arrival_skipped_records.csv`)
Documents records that could not be processed:

| Column       | Description                                                |
|-------------|-----------------------------------------------------------|
| `record_id` | Unique identifier for the skipped record                  |
| `reason`    | Explanation for why the record was skipped                |

### Probability Series (`prob_series.h5`)
HDF5 file containing the full PhaseNet probability time series for each record:
- **p_prob_series**: P wave probability time series (float32, LZF compressed)
- **s_prob_series**: S wave probability time series (float32, LZF compressed)

---

### Bypass Records
The optional `bypass_records_ffp` parameter allows you to specify custom P-wave arrival times for specific records:
- Useful for records where PhaseNet picks may be inaccurate
- Allows manual override of automated picks
- Should contain record IDs and corresponding custom P-wave index values

---

## 📋 Prerequisites

- **MSEED files**: Waveform data must be downloaded and available in the directory structure
- **PhaseNet environment**: Conda environment with PhaseNet package installed and configured
- **Three-component data**: Records must contain vertical, north-south, and east-west components
- **Earthquake source table**: Required for merging event datetime information (After Tectonic Domain Step)