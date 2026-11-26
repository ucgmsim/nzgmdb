# ⏱️ Real-Time Runs

`nzgmdb/scripts/real_time_eq_runs.py` runs the NZGMDB pipeline for new GeoNet events in near-real-time. Each event gets its own NZGMDB output folder. Optionally, the results are pushed to [SeismicNow](https://quakecoresoft.canterbury.ac.nz/seismicnow/), the progress is posted to a Slack thread and the [Auto CMT](Auto-CMT.md) pipeline is launched once the event is final.

---

## 🚀 Entry Points

### Poll for new events

```bash
python -m nzgmdb.scripts.real_time_eq_runs poll-earthquake-data \
    <main_dir> <gm_classifier_dir> <conda_sh> <gmc_activate> <gmc_predict_activate> \
    <ko_matrix_path> <nzcvm_data_ffp> [--add-seismic-now] [--machine <machine>] [--no-profile]
```

Every minute, this checks the GeoNet catalogue for M3.5+ events in the NZ bounding box over the last 10 minutes. It runs `run-event` for each new event in `<main_dir>/<event_id>/`. Events whose folder already exists are skipped. Events with no stations are remembered, so they are not run again.

### Run a single event

```bash
python -m nzgmdb.scripts.real_time_eq_runs run-event \
    <event_id> <event_dir> <gm_classifier_dir> <conda_sh> <gmc_activate> <gmc_predict_activate> \
    <ko_matrix_path> <nzcvm_data_ffp> [--add-seismic-now] [--machine <machine>] [--no-profile]
```

`<event_dir>` must already exist. Without `--add-seismic-now`, nothing is sent to SeismicNow or Slack and no CMT is launched, which is useful for testing:

```bash
mkdir -p /data/real_time_test/2026p886116
python -m nzgmdb.scripts.real_time_eq_runs run-event 2026p886116 /data/real_time_test/2026p886116 \
    ~/gm_classifier ~/anaconda3/etc/profile.d/conda.sh \
    "conda activate gmc_features" "conda activate gmc_predict" \
    /path/to/KO_matrices /path/to/nzcvm_data --machine local
```

**Arguments**
- **`<main_dir>`** / **`<event_dir>`** - Folder for all events (poll) or for this event (run-event)
- **`<gm_classifier_dir>`**, **`<conda_sh>`**, **`<gmc_activate>`**, **`<gmc_predict_activate>`**, **`<ko_matrix_path>`** - As for [Phase Arrival](Phase-Arrival.md) and [GMC](GMC.md). `<gmc_activate>` is used for PhaseNet and GMC feature extraction
- **`<nzcvm_data_ffp>`** - The `nzcvm_data` folder, as for [Fetching Site Table](Fetching-site-table.md)

**Options**
- **`--add-seismic-now`** - Add / update / remove the event on SeismicNow, post to Slack and launch the Auto CMT (default: off)
- **`--machine`** - Machine in `nzgmdb/config/machine_config.yaml` that sets the processes for each step (default: `local`)
- **`--profile` / `--no-profile`** - Write a cProfile dump to `<event_dir>/nzgmdb_profile.prof` (default: on)

---

## ⚙️ Process

`run-event` runs the pipeline twice: a fast basic pass so the event can be shown quickly, then the full pass with GMC.

### 🔹 1. Basic pass
1. [Fetching Site Table](Fetching-site-table.md)
2. [Parse Geonet](Parse-Geonet.md) for this event only, using GeoNet's near-real-time FDSN service (`real_time_url` in `config.yaml`)
3. [Waveform Extraction](Waveform-Extraction.md) with `--real-time`, which waits until each request's end time is at least 30 s old
4. [Add Tectonic Domain](Add-Tectonic-Domain.md)
5. Basic processing of the mseeds, without GMC fmin / fmax
6. [IM Calculation](IM-Calculation.md) for every IM in `config.yaml` (including MMI)
7. [Calculate Distances](Calculate-Distances.md), [Merge Aftershocks](Merge-Aftershocks.md) and [Merge flatfiles](Merge-Flatfiles.md)

With `--add-seismic-now`, the event's latest GeoNet values are fetched, and it is added to SeismicNow at "basic" status with a Slack thread. Events at `best` quality below M3.5 are not added. An event with no stations is skipped and its folder is removed.

### 🔹 2. Full pass
1. [Phase Arrival](Phase-Arrival.md), [Calculate SNR](Calculate-SNR.md) and [Calculate Fmax](Calculate-Fmax.md)
2. [GMC](GMC.md) with `--allow-empty`, so an event whose records all fail feature extraction still finishes
3. [Process records](Process-Records.md) with the GMC fmin and the fmax
4. IM Calculation again with `--force-recompute`, as the records were reprocessed under the same record IDs
5. Calculate Distances, Merge Aftershocks and Merge flatfiles

With `--add-seismic-now`, SeismicNow is updated with the final results. An event that has dropped below M3.5 at `best` quality is removed instead. After a successful update, the [Auto CMT](Auto-CMT.md) pipeline is launched in the background in `<event_dir>/cmt_1d/`.

Records with too little pre-event noise (common for triggered HN recordings of small events) are skipped by SNR and GMC. They stay in the flatfiles at basic processing.

---

## 💬 Slack

Slack updates need two environment variables:

```bash
export SLACK_BOT_TOKEN=<bot token>
export SLACK_CHANNEL=<channel id>
```

Each event gets one parent message showing its status and summary, with timestamped replies in its thread for each stage, failure or SeismicNow change. Slack errors are printed and never stop the pipeline. A crash in `run-event` is reported in the thread before it is raised.

---

## 📦 Output

`<event_dir>/` holds a normal NZGMDB output folder for one event (see File Structure on [Home](Home.md)), plus:

| File | Description |
|------|-------------|
| `nzgmdb_timings.csv` | Seconds and percentage for each pipeline step, written even when the run fails |
| `nzgmdb_profile.prof` | cProfile dump of the main process (with `--profile`) |
| `cmt_1d/` | Auto CMT results (with `--add-seismic-now`, once the event is final) |

Old CMT run folders can be cleaned up with `cleanup_old_cmts`, see [Auto CMT](Auto-CMT.md#cleanup).
