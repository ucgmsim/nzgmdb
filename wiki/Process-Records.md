# 🔄 Process Records

This step in the NZGMDB pipeline performs advanced waveform processing to convert MSEED files into ASCII text files suitable for downstream intensity measure calculations.

---

## 🚀 Entry Point

To process records and convert MSEED files to ASCII format, run the following Python script:

```bash
python -m nzgmdb.scripts.run_nzgmdb process-records <main_dir>
```

- **<main_dir>** is the top-level output directory where NZGMDB stores its results.

Example:
```bash
python -m nzgmdb.scripts.run_nzgmdb process-records nzgmdb_output/
```

Optional parameters include:
- `--n-procs`: Number of processes to use for parallel processing (default: 1)
- `--bypass-records-ffp`: Path to bypass records file for custom fmin/fmax values

This will process MSEED files and create ASCII outputs in:

```bash
nzgmdb_output/waveforms/year/event_id/processed/evid_station_channel_location.000
nzgmdb_output/waveforms/year/event_id/processed/evid_station_channel_location.090  
nzgmdb_output/waveforms/year/event_id/processed/evid_station_channel_location.ver
```

---

## 📋 Prerequisites

The Process Records step requires the following inputs from previous pipeline steps:
- **[Parse Geonet](https://github.com/ucgmsim/nzgmdb/wiki/Parse-Geonet)** (generates MSEED files in the waveforms directory)
- **[Calculate Fmax](https://github.com/ucgmsim/nzgmdb/wiki/Calculate-Fmax)** (provides maximum usable frequency values)
- **[GMC](https://github.com/ucgmsim/nzgmdb/wiki/GMC)** (provides minimum frequency and quality classification scores)

---

## ⚙️ Process

### 🔹 Record Frequency Filtering Extraction

This step extracts frequency bounds from the `gmc_predictions.csv` and `fmax.csv` files, which are essential for filtering waveforms. It supports Horizontal/Vertical-specific frequency extraction and allows for custom overrides via a bypass records file.

**1. Fmin Extraction**
- Retrieves GMC predictions from `gmc_predictions.csv`
- **Component-specific fmin extraction**:
  - `fmin_h`: Maximum `fmin_mean` value from horizontal components (X, Y)
  - `fmin_v`: `fmin_mean` value from vertical component (Z)

**2. Fmax Extraction**
- Loads Fmax values from `fmax.csv` 
- **Component-specific fmax extraction**:
  - `fmax_h`: Minimum Fmax from horizontal components (000, 090)
  - `fmax_v`: Fmax value from vertical component (ver)
- Ensures frequency band validity for processing

**3. Bypass Records Support**
- Optionally loads custom frequency bounds from bypass records file
- Overrides GMC/Fmax values with user-specified component-specific fmin/fmax when provided
- Handles NaN values appropriately during override process

### 🔹 Waveform Processing Pipeline

Each MSEED file undergoes a comprehensive processing workflow:

#### Initial Preprocessing
1. **Component Validation** - Ensures all 3 components (000, 090, ver) are present
2. **Demean and Detrend** - Remove offset and linear trends
3. **Taper Application** - Apply 5% Tukey taper to both ends
4. **Zero Padding** - Add 5 seconds of zeros at start and end
5. **Inventory Response Removal** - Remove instrument response using station metadata
6. **Component Rotation** - Rotate horizontal components to North-East-Vertical (NEZ)
7. **Gravity normalisation** - Divide acceleration data by the acceleration due to gravity (9.81 m/s²)

#### Advanced Signal Processing
1. **Component-Specific Bandpass Filtering** 
   - **Horizontal Components (000, 090)**: Apply Butterworth bandpass filter using `fmin_h/1.25` to `fmax_h`
   - **Vertical Component (ver)**: Apply Butterworth bandpass filter using `fmin_v/1.25` to `fmax_v`
   - **Frequency Scaling**: fmin values are divided by 1.25 to ensure fmin is actually T-useable
   - **Fallback Values**: Uses `low_cut_default` (0.04 Hz) if fmin unavailable, `1/(2.5*dt)` if fmax unavailable
2. **Zero Padding Removal** - Trim zero-padded sections after filtering
3. **Integration to Velocity** - Calculate velocity via numerical integration
4. **Integration to Displacement** - Calculate displacement via double integration
5. **Polynomial Detrending** - Fit 6th-order polynomial to displacement
6. **Baseline Correction** - Subtract 2nd derivative of polynomial from original acceleration

#### Quality Assurance
- **Differentiation Check** - Verify successful velocity/displacement calculations
- **Component-Specific Filter Validation** - Ensure `fmin_h < fmax_h` and `fmin_v < fmax_v` for valid frequency bands
- **Component Consistency** - Process all components with their respective optimal parameters

### 🔹 Configuration Parameters

Key configuration values from `config.yaml`:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `taper_fraction` | 0.05 | Fraction for Tukey taper application |
| `zero_padding_time` | 5 | Zero padding duration (seconds) |
| `g` | 9.81 | Gravitational acceleration (m/s²) |
| `order_default` | 4 | Butterworth filter order |
| `poly_order_default` | 6 | Polynomial detrending order |
| `low_cut_default` | 0.04 | Default low-cut frequency if fmin unavailable |

---

## 📦 Output

### 🔹 Processed Waveform Files

The primary output consists of ASCII text files stored in the directory structure:
```
waveforms/
├── year/
│   └── event_id/
│       └── processed/
│           ├── evid_station_channel_location.000  # North component
│           ├── evid_station_channel_location.090  # East component  
│           └── evid_station_channel_location.ver  # Vertical component
```

Each ASCII file contains:
- **Header Line**: Metadata including station, channel, sampling rate, number of points
- **Data Lines**: One acceleration value per line in g units
- **Format**: Plain text with consistent decimal precision

### 🔹 Metadata Files

Additional output files are generated in the `flatfiles/` directory:

- **`processed_records_skipped.csv`**: Contains failed records with detailed failure reasons:
  - "File did not contain 3 components"
  - "Failed to find Inventory information"  
  - "Failed to remove sensitivity"
  - "Failed to rotate the data"
  - "Unable to differentiate record"
  - "Lowcut frequency greater than highcut frequency" (component-specific)

---

## 🔧 Technical Implementation

### 🔹 Core Processing Functions

The record processing utilises functions from the `nzgmdb.data_processing` module:

- **`process_observed.process_single_mseed()`**: Main processing function for individual MSEED files
- **`waveform_manipulation.initial_preprocessing()`**: Handles basic waveform preprocessing
- **`waveform_manipulation.high_and_low_cut_processing()`**: Performs advanced frequency filtering and baseline correction
- **`waveform_manipulation.butter_bandpass_filter()`**: Applies Butterworth bandpass filtering

---

## ⚠️ Important Notes

- **Component Dependency**: All three components must be present and successfully processed for record inclusion
- **Component-Specific Frequency Validation**: The step ensures `fmin_h < fmax_h` and `fmin_v < fmax_v`; invalid ranges result in record skipping
- **Component Split Filtering**: Different frequency bounds for horizontal vs vertical components

---

## 🔗 Related Steps

- **Previous**: [GMC](GMC.md) - Provides fmin values and quality scores for record filtering
- **Previous**: [Calculate Fmax](Calculate-Fmax.md) - Provides maximum usable frequency bounds
- **Next**: [IM Calculation](IM-Calculation.md) - Uses processed ASCII files for intensity measure computation
- **Related**: [Parse Geonet](Parse-Geonet.md) - Provides raw MSEED files for processing