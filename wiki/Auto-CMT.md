# Auto CMT

`nzgmdb/scripts/run_auto_cmt.py` runs an automated 1-D centroid moment tensor (CMT) inversion for one event. It runs the inversion three ways and keeps the best solution. The inversion itself comes from the [`cmt_solutions`](https://github.com/ucgmsim/cmt_solutions) package (`auto_cmt.run_cmt`), which uses BayesISOLA.

The real-time pipeline (`real_time_eq_runs.py`) launches it in the background once an event is finalised in SeismicNow, and posts the result to the event's Slack thread. You can also run it by hand for any past event.

## Workflow

| Run | Folder | Stations | Azimuth control | Deviatoric |
|-----|--------|----------|-----------------|------------|
| 1 | `all_stations` | all stations in the search radius | off | no |
| 2 | `azimuth` | azimuth-thinned (min 3 occupied 45° sectors, max 2 per sector, min 5 stations) | on | no |
| 3 | `drop_outliers_dev` | all stations minus outliers found in **both** runs 1 and 2 | off | yes |

1. Runs 1 and 2 run in parallel, then run 3 runs.
2. The run with the highest overall variance reduction (VR) is selected, and its `cmt_solution.csv` is copied to the top-level output folder. If two runs tie, the earlier run in the table wins.
3. Each successful run is cleaned up as soon as it finishes (see [Cleanup](#cleanup)).

### Outlier stations

- A station's VR is the mean VR over its used components, taken from `results/station_df.csv`.
- In each run, a station is an outlier when **both** of these hold:
  - its robust z-score, `(vr - median) / (1.4826 * MAD)`, is below `-outlier_z` (default 5.0);
  - its VR is below `outlier_vr_floor` (default -1.0), meaning the residual is larger than the data.
- Only stations that are outliers in both run 1 and run 2 are dropped, since being bad in both points to a problem with the station itself.
- Nothing is dropped in these cases (run 3 still runs):
  - dropping would leave fewer than 5 stations;
  - run 1 or run 2 failed.

For example, on the 2013 Cook Strait event the all-stations median VR is 0.07. TON8 has a VR of -319 and is dropped. HIZ (VR -0.57, z -6.6) is kept because its VR is above the -1.0 floor.

### Inversion settings

These are set in `auto_cmt.run_cmt.run_cmt_inversion`:

| Setting | Value |
|---|---|
| Frequency band | 0.02–0.06 Hz |
| Location / time uncertainty | 3.0 km / 3.0 s |
| Horizontal grid step | 1 km for M ≤ 5.0, otherwise 2 km |
| Station jackknife | on, min 4 stations |
| Threads | 12 per run |

## Usage

```bash
# Past event: the event CSV is built from the GeoNet quake API
python -m nzgmdb.scripts.run_auto_cmt run-cmt-pipeline 2013p543824 /path/out/2013p543824/cmt_1d

# Past event already processed by NZGMDB: reuse its source table
python -m nzgmdb.scripts.run_auto_cmt run-cmt-pipeline 2013p543824 /path/event_dir/cmt_1d \
    --event-csv /path/event_dir/flatfiles/earthquake_source_table_geonet.csv

# Just-happened event: use GeoNet's near-real-time FDSN service
python -m nzgmdb.scripts.run_auto_cmt run-cmt-pipeline <event_id> <event_dir>/cmt_1d --event-csv <csv> --real-time
```

| Option | Default | Description |
|---|---|---|
| `--event-csv` | built from GeoNet | Event CSV (`evid, datetime, lat, lon, depth, mag`) |
| `--real-time` | off | Use GeoNet's near-real-time FDSN service |
| `--threads` | 12 | Axitra threads per run |
| `--outlier-z` | 5.0 | Robust z-score threshold for outliers |
| `--outlier-vr-floor` | -1.0 | Station VR threshold for outliers |
| `--nz-3dvm-path` | bundled `nz3dvm_2p3.csv` | 3-D NZ velocity model |
| `--skip-cleanup` | off | Keep all run files, for debugging |
| `--profile` | off (on for real-time launches) | Write a cProfile dump per run to `<run>/results/profile.prof` |

Disk use peaks at roughly 2 × 40 GB while runs 1 and 2 are running at the same time. Most of that is the Green's functions.

## Outputs

```
cmt_1d/
├── all_stations/        # run 1 (cleaned)
├── azimuth/             # run 2 (cleaned)
├── drop_outliers_dev/   # run 3 (cleaned)
├── event.csv            # event row used
├── station_vr.csv       # per-station VR, z-score and outlier flags for runs 1 and 2
├── cmt_selection.csv    # one row per run: status, VR, Mw, depth, dropped stations, selected
├── cmt_solution.csv     # copy of the selected run's solution
├── timings.csv          # pipeline step timings + each run's BayesISOLA stage timings
└── cmt_1d_run.log       # pipeline log (real-time launches only)
```

Each run folder has its own `cmt_run.log` with that run's full output. Its `results/` folder also has `timings.csv`, and `profile.prof` when profiling is on.

## Timing and profiling

`cmt_1d/timings.csv` combines two sets of timings:

- **`run = pipeline` rows:** event CSV, velocity grid, the parallel all-stations/azimuth runs, outlier detection, the drop-outliers run, selection and the total.
- **One block per run:** BayesISOLA stage timings from `cmt_solutions` (download, Green's functions, MT inversion, jackknife, plots, …), plus that run's cleanup time. `seconds` is exclusive time (nested stages are not double counted), and `percent` is the share of that run's total.

For real-time events the Slack completion message adds a timing line, e.g. `Timing: total 1h42m (all_stations 50m, azimuth 48m, drop_outliers_dev 52m); slowest stages: greens_functions 61%, mt_inversion 18%, plots 9%`.

The NZGMDB part of the real-time run (`real_time_eq_runs.py run-event`, up to the CMT launch) writes its own timings to `<event_dir>/nzgmdb_timings.csv`, one row per pipeline step. With `--profile` (the default) it also writes a cProfile dump of the main process to `<event_dir>/nzgmdb_profile.prof`. Steps that run in worker pools or conda subprocesses show up there only as waiting time, so use `nzgmdb_timings.csv` to see which step is slow.

View any profile with:

```bash
snakeviz <path>/profile.prof
```

If a run fails, its error is recorded in `cmt_selection.csv` and the other runs carry on. The command only fails (non-zero exit) if all three runs fail.

## Cleanup

Each successful run is cleaned automatically. Cleanup keeps:

- `results/`, `figures/` and `html/`
- `cmt_solution.csv`, `report.html` and `index.html`
- `log.txt`, `log_green.txt` and any `*.log` file

Everything else is deleted, including `green/` (around 40 GB), `raw/`, `mouse/`, `input/`, `metadata/` and the top-level PNGs. Failed runs are left untouched so you can debug them.

To clean up past runs, point the command at a single run folder, an event's `cmt_1d` folder or a whole folder of events. It cleans every folder containing `results/solution_summary.csv`:

```bash
python -m nzgmdb.scripts.run_auto_cmt cleanup-cmt /path/to/cmt_runs --dry-run   # list what would be deleted
python -m nzgmdb.scripts.run_auto_cmt cleanup-cmt /path/to/cmt_runs
```

To tidy up old live event data, use `cleanup_old_cmts`. It checks a folder of live events (one sub folder per event ID) and looks for sub folders with `cmt` anywhere in their name, e.g. `cmt_1d` or `cmt_1d_deviatoric`. It cleans every completed run inside them. CMT folders with no completed run (e.g. failed runs) are listed but left untouched:

```bash
python -m nzgmdb.scripts.cleanup_old_cmts /path/to/live_data --dry-run
python -m nzgmdb.scripts.cleanup_old_cmts /path/to/live_data --event-ids 2026p450287 --event-ids 2026p455759
```
