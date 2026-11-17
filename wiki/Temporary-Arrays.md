# 📡 Temporary Arrays

Temporary seismic arrays are short-term deployments (often by universities or overseas groups) whose data is held by providers such as IRIS and AUSPASS, not GeoNet. This page covers how that data is collected, how it is added to over time, and how the NZGMDB pipeline uses it.

---

## 🔀 How this differs from the main pipeline

| | GeoNet (main pipeline) | Temporary arrays |
|---|---|---|
| **When data is fetched** | During `extract-waveforms`, one event window at a time | **Before** the pipeline, as bulk continuous data |
| **Where data comes from** | GeoNet FDSN, live | Local folder of continuous mseed (`--tmp-array-dir`) |
| **Why** | GeoNet serves small requests quickly | Providers rate limit, embargo data and reject many small requests, so pulling continuous data once is more reliable |
| **Code** | `nzgmdb/data_retrieval`, `nzgmdb/scripts/run_nzgmdb.py` | `nzgmdb/tmp_array`, `nzgmdb/scripts/tmp_arrays.py` |

The temporary array tools do not change the pipeline. The pipeline only touches temporary array data in two places:

1. **Inventories / site table:** with `--add-tmp-arrays`, the pipeline fetches station metadata for every provider and network listed under `tmp_array_providers_networks` in [config.yaml](https://github.com/ucgmsim/nzgmdb/blob/main/nzgmdb/config/config.yaml).
2. **Waveform extraction:** any record whose provider is not GEONET is cut from the local continuous files in `--tmp-array-dir` instead of being downloaded.

So a network must be **both** downloaded and listed in `config.yaml` to end up in the database. `get-stations` prints any networks it finds that are not in the config.

---

## 🗂️ Data folder layout

```
<data_dir>/
├── waveforms/                     # pass this folder to --tmp-array-dir
│   └── <NET>/<NET>_<STA>_<CHAN>_<LOC>/
│       └── <NET>.<STA>.<LOC>.<CHA>__<START>__<END>.mseed
├── stationxml/<NET>/<NET>_<STA>_<CHAN>_<LOC>/<NET>.<STA>.xml
├── download_results_<start>_<end>.csv
├── check_<start>_<end>.csv
├── pending_data.csv               # rows still missing data (embargoed / not released)
├── newly_available_<date>.csv     # written by recheck-pending
└── dropbox_manifest.csv
```

Files are split on **calendar months** (e.g. `__20260101T000000Z__20260201T000000Z`), so different download windows always line up and never overlap. Data downloaded before this change used 15-day chunks. It can stay where it is, and new windows are added next to it.

---

## 🚀 Commands

All commands are in `nzgmdb/scripts/tmp_arrays.py`. Run `python nzgmdb/scripts/tmp_arrays.py <command> --help` for the full options.

### 1. `get-stations` – find stations

```bash
python nzgmdb/scripts/tmp_arrays.py get-stations stations_2026.csv 2026-01-01 2027-01-01
```

- Searches FDSN providers for stations inside NZ that operated between the dates.
- Keeps `HH, BH, HN, BN` channels and drops `NZ, IU` (from GeoNet in the main pipeline) and `SY` (IRIS synthetics).
- Providers that fail or have no station service are skipped with a message.
- At the end it lists networks **not in `config.yaml`**. Check whether each one should be used, and if so add it under `tmp_array_providers_networks`.
- Useful options: `--providers IRIS,AUSPASS`, `--channels HH,HN`, `--exclude-networks NZ,IU,SY,AM`, `--no-coastline-filter`.

The CSV has one row per provider / network / station / channel prefix / location. `end_date` is blank for stations that are still running. `restricted` is `True` when the provider marks the data as closed or hides it when restricted data is excluded, `False` when it is open, and blank when the provider doesn't say.

### 2. `download-data` – download waveforms and StationXML

```bash
python nzgmdb/scripts/tmp_arrays.py download-data stations_2026.csv /data/tmp_arrays 2026-01-01 2027-01-01
```

- Downloads each station's data, clipped to the window and the station's operating dates. For stations that are still running, the window end is used.
- Adds to whatever is already in the data folder. Months that already have files are skipped, so it is safe to stop and re-run.
- If a provider rejects a request as too large, the chunk is halved and retried, down to 1 hour.
- Writes `download_results_<start>_<end>.csv` with one status per row:
  - `ok`: every month in the window has data
  - `partial`: some months have no data
  - `embargoed`: no data, and the data is restricted (flagged in the station CSV, or a short test request got HTTP 401/403)
  - `no_data`: no data and not known to be restricted (HTTP 204). Some providers return this for embargoed data too
  - `error`: the request failed. Re-run the same command
  - `outside_window`: the station was not running in the window
- Rows that are `partial`, `embargoed` or `no_data` are added to `pending_data.csv` (see [Embargoed data](#-embargoed-and-unreleased-data)), with `missing_reason` set to `restricted`, `no_data` or `unknown`.

> Use start and end dates on month boundaries (`YYYY-MM-01`). A window ending mid-month creates a short file, and a later window would then download an overlapping file for that month.

### 3. `check-data` – check what has been downloaded

```bash
python nzgmdb/scripts/tmp_arrays.py check-data stations_2026.csv /data/tmp_arrays 2026-01-01 2027-01-01
```

Prints and writes `check_<start>_<end>.csv`, giving each row a status of `complete`, `partial` or `missing`, the number of months done, and the number of **overlapping files**. If overlaps are reported, remove the duplicate files before running the pipeline.

### 4. `recheck-pending` – retry embargoed / unreleased data

```bash
python nzgmdb/scripts/tmp_arrays.py recheck-pending /data/tmp_arrays
```

Retries only the rows in `pending_data.csv`, writes the rows that gained data to `newly_available_<date>.csv`, and prints the sites and dates to re-run the pipeline for. Use `--providers AUSPASS` to recheck only some providers.

### 5. `backup-data` – back up to Dropbox

```bash
python nzgmdb/scripts/tmp_arrays.py backup-data /data/tmp_arrays dropbox:/QuakeCoRE/NZGMDB_tmp_arrays
```

- Zips each station folder and uploads it with rclone (see [Upload to Dropbox](Upload-Dropbox.md) for rclone setup), checking the uploaded size.
- `dropbox_manifest.csv` stores the file count and size of each folder. A folder is uploaded again when new data is added to it.
- A manifest from the old backup script is accepted: its `DONE` folders are treated as up to date.

---

## 📅 Adding a new period (e.g. 2026)

Run this once all of the period's data should be available. Temporary array data is often released months or years later, so re-running for earlier periods is cheap and fills in data that has since been released.

1. **Find stations** operating in the period:
   ```bash
   python nzgmdb/scripts/tmp_arrays.py get-stations /data/tmp_arrays/stations_2026.csv 2026-01-01 2027-01-01
   ```
2. **Update `config.yaml`** with any new networks it reports that should be included.
3. **Download** into the existing data folder:
   ```bash
   python nzgmdb/scripts/tmp_arrays.py download-data /data/tmp_arrays/stations_2026.csv /data/tmp_arrays 2026-01-01 2027-01-01
   ```
4. **Check**, then re-run step 3 for any rows that failed (`error`). Embargoed rows are tracked in `pending_data.csv`:
   ```bash
   python nzgmdb/scripts/tmp_arrays.py check-data /data/tmp_arrays/stations_2026.csv /data/tmp_arrays 2026-01-01 2027-01-01
   ```
5. **Back up**:
   ```bash
   python nzgmdb/scripts/tmp_arrays.py backup-data /data/tmp_arrays dropbox:/QuakeCoRE/NZGMDB_tmp_arrays
   ```
6. **Run the pipeline for the period** with temporary arrays enabled, into a new output folder (see the [README](https://github.com/ucgmsim/nzgmdb/blob/main/README.md) for the positional arguments):
   ```bash
   python nzgmdb/scripts/run_nzgmdb.py run-full-nzgmdb /output/nzgmdb_2026 2026-01-01 2026-12-31 \
     /path/to/nzcvm_data /path/to/gm_classifier /path/to/conda.sh \
     "conda activate gmc_features" "conda activate gmc_predict" /path/to/KO_matrices \
     --add-tmp-arrays --tmp-array-data-dir /data/tmp_arrays/waveforms
   ```
7. **Merge** the new period into the existing database:
   ```bash
   python nzgmdb/scripts/run_nzgmdb.py merge-databases /output/nzgmdb_v4p4/flatfiles /output/nzgmdb_2026/flatfiles /output/nzgmdb_merged/flatfiles
   ```
   Records in the second database replace duplicates in the first.

---

## 🔒 Embargoed and unreleased data

Temporary array data is often held back by the deployment team for a few years before it's public. The station metadata is visible, but data requests return HTTP 403 (restricted) or HTTP 204 (no data).

**How it is tracked**

- `get-stations` flags restricted stations in the `restricted` column.
- `download-data` still tries every row, since an embargo can lift at any time. Rows that end up without data go into `pending_data.csv` with:
  - the window
  - how many months are missing
  - the reason (`restricted`, `no_data` or `unknown`)
  - when the row was first seen and last checked
- Rows are removed from the list once their data is downloaded.

**Adding data once it is released**

Run this every few months, or when a deployment team says its data is public:

1. **Retry the pending rows:**
   ```bash
   python nzgmdb/scripts/tmp_arrays.py recheck-pending /data/tmp_arrays
   ```
   New months are downloaded into the same `waveforms/` folder, and the output looks like:
   ```
   net        sites      start        end
    X5  AKB03,AKB04 2023-07-01 2026-02-01
   ```
2. **Check `config.yaml`** lists the network under `tmp_array_providers_networks`.
3. **Run the pipeline for just those sites and dates** into a new output folder:
   ```bash
   python nzgmdb/scripts/run_nzgmdb.py run-full-nzgmdb /output/nzgmdb_backfill_X5 2023-07-01 2026-02-01 \
     /path/to/nzcvm_data /path/to/gm_classifier /path/to/conda.sh \
     "conda activate gmc_features" "conda activate gmc_predict" /path/to/KO_matrices \
     --only-sites AKB03,AKB04 --add-tmp-arrays --tmp-array-data-dir /data/tmp_arrays/waveforms
   ```
4. **Merge** into the current database. The new records are added, and any duplicates are replaced:
   ```bash
   python nzgmdb/scripts/run_nzgmdb.py merge-databases /output/nzgmdb_current/flatfiles /output/nzgmdb_backfill_X5/flatfiles /output/nzgmdb_merged/flatfiles
   ```
5. **Back up** with `backup-data`. Only the folders that changed are uploaded.

> `--only-sites` matches station codes. If two networks share a station code, both are included.

> Check the merged database after the first backfill. `merge-databases` has mostly been used to merge whole periods, so check that the event and site tables come through correctly when the second database only covers a few sites.

---

## ⚠️ Things to watch

- **Embargoed data:** see [Embargoed and unreleased data](#-embargoed-and-unreleased-data). `pending_data.csv` is the list of everything still waiting.
- **Events at the end of a period:** an event late on the last day needs data from the next window. That data only exists once the next period is downloaded, so those records may be skipped until then.
- **Config networks:** a network that is downloaded but not in `config.yaml` is ignored by the pipeline. A network in `config.yaml` that is not downloaded is skipped with `No Waveform Data`.
- **Provider duplicates:** when several providers serve the same station, the first in `--providers` order is kept.
