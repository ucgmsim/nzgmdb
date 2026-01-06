# 📤 Upload to Dropbox

The final step in the NZGMDB pipeline that packages all generated files into organised ZIP archives and uploads them to a centralised Dropbox repository for distribution and long-term storage. This is however an optional step that should only be done for official release runs for the UCGMSIM team.

---

## 🚀 Entry Point

To upload NZGMDB results to Dropbox, run the following Python script:

```bash
python -m nzgmdb.scripts.upload_to_dropbox upload-to-dropbox <input_directory>
```

- **<input_directory>** is the top-level output directory containing NZGMDB results

Example:
```bash
python -m nzgmdb.scripts.upload_to_dropbox upload-to-dropbox nzgmdb_output/
```

Optional parameters include:
- `--version`: Custom version string for Dropbox folder (default: directory name)
- `--n-procs`: Number of processes for parallel upload (default: 1)

### Additional Functions

**Upload Failed Files:**
```bash
python -m nzgmdb.scripts.upload_to_dropbox upload-failed-files <failed_files.txt>
```

**Download Archive:**
```bash
python -m nzgmdb.scripts.upload_to_dropbox download-dropbox-archive <output_dir> <version>
```

---

## 📋 Prerequisites

The Upload to Dropbox step is an **optional** final step that requires:
- **[Merge Flatfiles](https://github.com/ucgmsim/nzgmdb/wiki/Merge-Flatfiles)** (generates final flatfile outputs)
- **Rclone Configuration** (see Setup section below)
- **[Quality DB]()** (generates quality database files, optional)

---

## ⚙️ Setup Requirements

### 🔹 Rclone Installation

Ensure Rclone is installed and properly configured:

1. **Install Rclone**: Follow installation guide at https://rclone.org/install/

2. **Configure Dropbox Access**: Place the following configuration in `~/.config/rclone/rclone.conf`:

```ini
[dropbox]
type = dropbox
client_id = 
client_secret = 
token = {"access_token":"XXXXXXXXXXXXXXX","token_type":"bearer","expiry":"0001-01-01T00:00:00Z"}
```

3. **Access Permissions**: Contact the UCGMSIM team for Dropbox folder access and API tokens.

### 🔹 Dropbox Structure

Files are uploaded to: `dropbox:/QuakeCoRE/Public/NZGMDB/{version}/`

---

## ⚙️ Process

### 🔹 Archive Creation

The process creates several thematic ZIP archives:

**1. Flatfiles Archive (`flatfiles_{version}.zip`)**
- Contains all final CSV flatfiles from the flatfiles directory
- Includes earthquake source table, IM results, metadata files
- Uses `file_structure.FlatfileNames` for consistent file selection

**2. Pre-Flatfiles Archive (`pre_flatfiles_{version}.zip`)**
- Contains intermediate processing files
- Uses `file_structure.PreFlatfileNames` for file selection

**3. Skipped Records Archive (`skipped_{version}.zip`)**
- Documents all records that failed processing at various stages
- Includes failure reasons and diagnostic information
- Uses `file_structure.SkippedRecordFilenames` for comprehensive coverage

**4. SNR/FAS Archive (`snr_fas_{version}.zip`)**
- Packages all SNR and Fourier Amplitude Spectra CSV files
- Recursively includes all `*.csv` files from the SNR/FAS directory

**5. Quality Database Archive (`quality_flatfiles_{version}.zip`)** *(if present)*
- Contains quality-filtered flatfiles from the quality_db directory
- Only created if quality database processing was enabled

**6. Station XML Inventory Archive (`stationxml_{version}.zip`)** *(if present)*
- Contains station XML inventory files
- Only created if station XML inventories were generated
- Recursively includes all `*.xml` files from the stationxml directory

### 🔹 Waveform Packaging

**Hierarchical Waveform Archives:**

**Year-Level Archives:**
- Each year's waveforms are packaged into `{year}.zip` files
- Contains all events and associated raw/processed waveform data for that year

**Event-Level Archives:**
- Within each year, individual events are packaged into separate ZIP files
- Allows users to download specific events without entire year datasets
- Structure: `waveforms/{year}/{event_id}.zip`

**Content Organisation:**
- **Raw Data**: Original MSEED files from GeoNet
- **Processed Data**: ASCII waveform files (.000, .090, .ver components)

### 🔹 Upload Process

**1. Parallel Upload Management**
- Utilises multiprocessing pools for concurrent uploads
- Configurable process count via `--n-procs` parameter
- Optimised for large dataset transfers

**2. Upload Verification**
- Each uploaded file undergoes size verification
- Compares local file size with remote file size using `rclone lsf`
- Failed uploads are tracked for retry processing

**3. Error Handling and Recovery**
- Failed uploads are logged to `failed_files.txt`
- Provides `upload_failed_files` function for targeted retry
- Preserves upload state for incremental completion

---

## 📦 Output

### 🔹 Dropbox Directory Structure

```
dropbox:/QuakeCoRE/Public/NZGMDB/{version}/
├── flatfiles_{version}.zip           # Final flatfiles
├── pre_flatfiles_{version}.zip       # Intermediate processing files  
├── skipped_{version}.zip             # Failed records documentation
├── snr_fas_{version}.zip             # SNR and FAS data
├── quality_flatfiles_{version}.zip   # Quality-filtered database (optional)
├── stationxml_{version}.zip          # Station XML inventories (optional)
└── waveforms/
    ├── {year1}.zip                   # Year-level waveform archive
    ├── {year2}.zip                   # Year-level waveform archive
    ├── {year1}/
    │   ├── {event_id1}.zip           # Individual event archive
    │   └── {event_id2}.zip           # Individual event archive
    └── {year2}/
        ├── {event_id3}.zip           # Individual event archive
        └── {event_id4}.zip           # Individual event archive
```

### 🔹 Local ZIP Directory

All created ZIP files are stored locally in `{input_directory}/zips/` before upload:

```
{input_directory}/zips/
├── flatfiles_{version}.zip
├── pre_flatfiles_{version}.zip
├── skipped_{version}.zip
├── snr_fas_{version}.zip
├── quality_flatfiles_{version}.zip
├── stationxml_{version}.zip
├── failed_files.txt # (if any uploads failed)
└── waveforms/
    ├── {year}.zip
    ├── {year}/
        └── {event_id}.zip
```

### 🔹 Upload Status Reports

**Successful Upload:**
```
All files uploaded successfully.
```

**Failed Uploads:**
```
Failed to upload {N} files. See {output_dir}/zips/failed_files.txt for paths.
```

---

## 🔧 Technical Implementation

### 🔹 Core Functions

**`zip_files(file_list, output_dir, zip_name)`**
- Creates compressed ZIP archives using `zipfile.ZIP_DEFLATED`
- Preserves original filenames as archive names
- Returns ZIP file path for upload tracking

**`upload_zip_to_dropbox(local_file, dropbox_path)`**
- Executes Rclone copy operations via subprocess
- Performs post-upload size verification
- Returns failed file path on error, None on success

**`determine_dropbox_path(dropbox_version_dir, local_file)`**
- Helper function for retry operations
- Reconstructs proper Dropbox paths from local ZIP structure
- Handles nested waveform directory hierarchies

### 🔹 Multiprocessing Strategy

- **Archive Creation**: Parallel ZIP generation for year and event-level waveform archives
- **Upload Operations**: Concurrent uploads using process pools

---

## ⚙️ Configuration Parameters

Key configuration values from `config.yaml`:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `DROPBOX_PATH` | "dropbox:/QuakeCoRE/Public/NZGMDB" | Base Dropbox path |

---

## 🔄 Archive Download and Reconstruction

The `download_dropbox_archive` function provides complete archive retrieval:

### 🔹 Download Process

**1. Archive Retrieval**
- Downloads all thematic ZIP files from specified version
- Recreates local directory structure for NZGMDB pipeline compatibility

**2. Automatic Extraction**
- Extracts all archives to appropriate pipeline directories
- Reconstructs waveform directory structure with proper event Organisation

**3. Waveform Reorganisation**
- Automatically sorts waveform files by type (MSEED vs processed)
- Creates proper event subdirectories (mseed/, processed/)
- Maintains pipeline-compatible file Organisation

---

## ⚠️ Important Notes

- **Optional Step**: Upload to Dropbox is not required for pipeline completion
- **Large Datasets**: Upload times depend on data volume and network bandwidth
- **Access Control**: Requires proper Dropbox API credentials and folder permissions
- **Resume Capability**: Failed uploads can be resumed using the `upload_failed_files` function

---

## 🔗 Related Steps
- **Previous**: [Quality DB](Quality-DB.md) - Creates quality database files included in the upload (if enabled)