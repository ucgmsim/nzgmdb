# 🤖 Ground Motion Classification (GMC)

This step in the NZGMDB pipeline uses machine learning to classify seismic records and generate quality scores and minimum usable frequency (Fmin) values. The GMC step leverages the `gm_classifier` repository to extract features from waveform data and apply a trained neural network model for automated quality assessment.

---

## 🚀 Entry Point

To run GMC processing, use the following Python script:

```bash
python -m nzgmdb.scripts.run_nzgmdb run-gmc-processing <main_dir> <gm_classifier_dir> <ko_matrices_dir> <conda_sh> <gmc_activate> <gmc_predict_activate> --n-procs <n_procs> --waveform-dir <waveform_dir> --output-dir <output_dir> --bypass-records-ffp <bypass_records_ffp>
```

**Required Parameters:**
- **<main_dir>** - The main directory for NZGMDB output
- **<gm_classifier_dir>** - Directory path to the cloned `gm_classifier` repository
- **<ko_matrices_dir>** - Directory containing pre-generated Konno-Ohmachi matrices
- **<conda_sh>** - Path to your conda.sh script for environment activation
- **<gmc_activate>** - Command to activate the GMC environment for feature extraction (e.g., "conda activate gmc_features")
- **<gmc_predict_activate>** - Command to activate the GMC prediction environment (e.g., "conda activate gmc_predict")

**Optional Parameters:**
- **--n-procs** - Number of processes for parallel processing (default: 1)
- **--waveform-dir** - Custom directory containing waveform files (default: uses main_dir/waveforms)
- **--output-dir** - Custom output directory for GMC predictions (default: uses main_dir/flatfiles)
- **--bypass-records-ffp** - Path to bypass records file containing custom Fmin / Score values

**Example:**
```bash
python -m nzgmdb.scripts.run_nzgmdb run-gmc-processing nzgmdb_output/ ~/gm_classifier/ ~/ko_matrices/ ~/.conda/conda.sh "conda activate gmc_features" "conda activate gmc_predict" --n-procs 4
```

This will create the following output files:
```bash
nzgmdb_output/flatfiles/gmc_predictions.csv
nzgmdb_output/gmc/batch_*/gmc_predictions.csv
nzgmdb_output/gmc/batch_*/features_comp_*.csv
nzgmdb_output/gmc/batch_*/extract_features.log
nzgmdb_output/gmc/batch_*/predict.log
```

---

## 📋 Prerequisites

The GMC step requires the following inputs from previous pipeline steps:
- **[Parse Geonet](Parse-Geonet.md)** - Provides MSEED waveform files
- **[Phase Arrival](Phase-Arrival.md)** - Generates phase arrival table and probability series files
- **Pre-generated KO matrices** - Required for feature extraction (created using gm_classifier tools)
- **Two conda environments** - `gmc_features` and `gmc_predict` (set up following gm_classifier documentation)

---

## ⚙️ Process

### 🔹 Batch Processing Setup

The GMC processing is optimized for parallel execution:

1. **Locate MSEED Files** - Recursively finds all `.mseed` files in the waveform directory
2. **Create Batches** - Splits files into even batches based on `n_procs` parameter
3. **Generate Batch Directories** - Creates `batch_0`, `batch_1`, etc. under the `gmc/` folder
4. **Create Batch Lists** - Generates text files (`batch_*.txt`) containing record IDs to process

### 🔹 Feature Extraction Phase

For each batch, the system performs feature extraction using the `gmc_features` environment:

**Command Executed:**
```bash
python <gm_classifier_dir>/gm_classifier/scripts/extract_features.py <gmc_dir> <waveform_dir> mseed --ko_matrices_dir <ko_matrices_dir> --record_list_ffp <batch_txt> --phase_arrival_table <phase_arrival_table_ffp> --prob_series <prob_series_ffp>
```

**Feature Extraction Process:**
- **Waveform Processing** - Loads MSEED files and applies preprocessing (demean, detrend, taper)
- **P-wave Arrival Integration** - Uses phase arrival table to identify signal windows
- **Konno-Ohmachi Smoothing** - Applies frequency domain smoothing using pre-computed matrices
- **Multi-component Analysis** - Extracts features for X (000), Y (090), and Z (ver) components
- **Feature Categories** - Generates both scalar features and SNR-based spectral features

**Output Files per Batch:**
- `features_comp_X.csv` - Features for X component (East-West)
- `features_comp_Y.csv` - Features for Y component (North-South) 
- `features_comp_Z.csv` - Features for Z component (Vertical)
- `failed_records_*/` - Directory containing records that failed feature extraction

### 🔹 Machine Learning Prediction Phase

After feature extraction, the system applies trained models using the `gmc_predict` environment:

**Command Executed:**
```bash
python <gm_classifier_dir>/gm_classifier/scripts/predict.py <gmc_dir> <predictions_output>
```

**Prediction Process:**
- **Model Loading** - Loads pre-trained neural network models for classification
- **Multi-output Prediction** - Generates predictions for quality scores, Fmin values, and binary classification
- **Component-wise Analysis** - Produces separate predictions for each seismic component

**Model Outputs:**
- **Quality Scores** - Numerical scores (0-1) indicating waveform quality for each component
- **Fmin Values** - Minimum usable frequency (Hz) for each component
- **Binary Classification** - Multi-earthquake classification (usable/not usable)
- **Uncertainty Estimates** - Standard deviations for all predictions

### 🔹 Result Aggregation

After all batches complete processing:
1. **Combine Batch Results** - Merges `gmc_predictions.csv` from all batch directories
2. **Apply Bypass Records** - Incorporates any custom Fmin values from bypass file
3. **Generate Final Output** - Creates consolidated `gmc_predictions.csv` in flatfiles directory

---

## 📦 Output

### 🔹 Primary Output: `gmc_predictions.csv`

Located in the flatfiles directory with the following columns:

| Column | Description                                                                     |
|--------|---------------------------------------------------------------------------------|
| `record_id` | Unique identifier including component (evid_station_channel_location_component) |
| `score_mean` | Mean quality score for this component [0-1]                                     |
| `score_std` | Standard deviation of quality score prediction                                  |
| `fmin_mean` | Mean minimum usable frequency for this component (Hz)                           |
| `fmin_std` | Standard deviation of Fmin prediction                                           |
| `multi_mean` | Multi-Mean classification score [0-1]                                           |
| `multi_std` | Standard deviation of Multi-Mean classification                                 |
| `record` | Base record identifier (evid_station_channel_location)                          |
| `component` | Component identifier (X, Y, Z)                                                  |
| `station` | Station code                                                                    |
| `event_id` | Event identifier                                                                |

### 🔹 Secondary Outputs

**Batch-level Results (`gmc/batch_*/`):**
- `gmc_predictions.csv` - Predictions for records in this batch
- `features_comp_*.csv` - Extracted features for each component
- `extract_features.log` - Feature extraction log file
- `predict.log` - Prediction log file
- `failed_records_*/` - Records that failed processing with error details

**Log Files:**
All processing logs are retained for debugging and quality assurance, including detailed error messages for failed records.

---

## 🔗 Related Steps

- **Previous**: [Calculate Fmax](Calculate-Fmax.md) - Provides maximum usable frequency data that complements GMC Fmin predictions
- **Next**: [Process Records](Process-Records.md) - Uses GMC quality scores and Fmin values to filter records for waveform processing
- **Related**: [Phase Arrival](Phase-Arrival.md) - Provides P-wave picks essential for GMC feature extraction