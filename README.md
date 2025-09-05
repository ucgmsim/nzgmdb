# NZGMDB

## 📌 Overview

The **NZGMDB** (New Zealand Ground Motion Database) repository contains a set of tools that automates the full pipeline for generating a quality-assured ground motion database for New Zealand. It processes raw data downloaded from the Geonet FDSN client (https://www.geonet.org.nz/data/access/FDSN) and processes it using the following steps:
- Extract seismic waveforms  
- Perform signal processing  
- Calculate intensity measures (IMs)  
- Produce standardised flatfiles for analysis

This tool is designed for researchers working with strong motion records.

This codebase and documentation is currently up-to-date with the latest NZGMDB release **Version 4.3 (September 2025)**.
More information on the NZGMDB and the steps involved can be found on the [NZGMDB Wiki](https://github.com/ucgmsim/nzgmdb/wiki).

---

## 🛠️ Installation

### 1. Clone the NZGMDB Repository

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ucgmsim/nzgmdb.git
cd nzgmdb
pip install -e .
```

### 2. Clone and Set Up `gm_classifier`

`NZGMDB` depends on the [`gm_classifier`](https://github.com/ucgmsim/gm_classifier) repository, for performing:

- **P-wave arrival detection**
- **Ground motion classification using machine learning**

Clone the repository:

```bash
git clone https://github.com/ucgmsim/gm_classifier.git
```

Then follow the instructions in the gm_classifier repository to create two Conda environments:

- **gmc_features**: For feature extraction

- **gmc_predict**: For feature classification

## ✅ Pre-requisites

Before running the full pipeline, ensure the following:

- Your Python environment has the `nzgmdb` package installed

- You have generated **KO (Konno-Ohmachi) matrices**. These are required by different steps in the NZGMDB pipeline and can be created using the tools in the `gm_classifier` repository. The readme in the `gm_classifier` repository provides instructions on how to generate these matrices.

- You know the paths to:
  - The cloned `gm_classifier` directory
  - Your `conda.sh` initialization script
  - The directory containing the generated KO matrices
  - Your two Conda environments:
    - `gmc_features` (for feature extraction)
    - `gmc_predict` (for feature classification)

## 🚀 Running the Pipeline

To run the NZGMDB pipeline you can look in the scripts
directory and locate [run_nzgmdb.py](nzgmdb/scripts/run_nzgmdb.py). This script will contain all the
functions to run the automated pipeline. You can run the entire automated pipeline
by running the following command:

```bash
python scripts/run_nzgmdb.py run-full-nzgmdb \
  /output_dir \
  2000-01-01 \
  2024-12-31 \
  /path/to/gm_classifier \
  /path/to/conda.sh \
  "conda activate gmc_features" \
  "conda activate gmc_predict" \
  /path/to/KO_matrices
```

### 🔍 Argument Breakdown

| Argument                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `/output_dir`                   | Output directory for processed results                                      |
| `2000-01-01`                    | Start date for event selection                                              |
| `2024-12-31`                    | End date for event selection                                                |
| `/path/to/gm_classifier`        | Path to the cloned `gm_classifier` repository                               |
| `/path/to/conda.sh`             | Path to the Conda initialization script (e.g., `/opt/anaconda3/etc/profile.d/conda.sh`) |
| `"conda activate gmc_features"` | Command to activate the feature extraction environment                      |
| `"conda activate gmc_predict"`  | Command to activate the classification environment                          |
| `/path/to/KO_matrices`          | Directory containing the pre-generated KO matrices                          |

### 📝 Additional Arguments

There are also a variety of extra arguments that can be passed to the script, allowing control and modification of different steps, such as event / site filtering or the number of processes to use for the given system.
For a full list of arguments and their descriptions, run:

```bash
python scripts/run_nzgmdb.py --help
```

## 📄 Documentation

For detailed documentation on the NZGMDB pipeline, including how to configure and run each step and some more scientific details, refer to the [NZGMDB Wiki](https://github.com/ucgmsim/nzgmdb/wiki)