import os
import pandas as pd
from obspy.clients.fdsn.mass_downloader import (
    MassDownloader,
    Restrictions,
    GlobalDomain,
)
from obspy import UTCDateTime
import multiprocessing


# ---------------- USER SETTINGS ---------------- #
# CSV_FILE = '/media/joel/data/nzgmdb/tmp_arrays/all_nz_sta_providers_desired_channels_mustang.csv'
CSV_FILE = "/media/joel/data/nzgmdb/tmp_arrays/HR1_inventory.csv"
# CSV_FILE = '/scratch/jobs/jri83/runs/tmp_array/download_completeness_evaluation_3.csv'

OUTPUT_DIR = "/media/joel/data/nzgmdb/tmp_arrays/hr1"
# OUTPUT_DIR = '/scratch/jobs/jri83/runs/tmp_array/mass_data_row_mp'
MSEED_DIR = "waveforms"
STATIONXML_DIR = "stationxml"

RESULTS_CSV = os.path.join(OUTPUT_DIR, "download_results.csv")

# month length in seconds (15 days)
MONTH_SECONDS = 15 * 24 * 3600
# Minimum chunk size (seconds) when backing off after 413 / manifest-too-large.
# 3600s = 1 hour.
MIN_CHUNK_SECONDS: int = 3600

# ------------------------------------------------ #


def _format_end_for_filename(end_dt):
    """Format an obspy UTCDateTime or parseable date string to the filename timestamp form."""
    if not isinstance(end_dt, UTCDateTime):
        end_dt = UTCDateTime(end_dt)
    return end_dt.strftime("%Y%m%dT%H%M%SZ")


def save_results(results, results_csv=RESULTS_CSV):
    """
    Create one CSV from a list of result dicts.
    - Finds all unique keys across results.
    - Uses a preferred column order for common fields.
    - Normalizes missing keys to empty string and converts non-scalar values to strings.
    """
    import os
    import json
    import pandas as pd

    if not results:
        # ensure output dir exists and write an empty file
        out_dir = os.path.dirname(results_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame().to_csv(results_csv, index=False)
        return

    # collect all keys
    all_keys = set()
    for r in results:
        if isinstance(r, dict):
            all_keys.update(r.keys())
        else:
            # non-dict entries will be recorded under 'value'
            all_keys.add("value")

    # preferred column order for readability
    preferred = [
        "idx",
        "provider",
        "net",
        "sta",
        "status",
        "error",
        "attempts",
        "chunklength",
        "timestamp",
        "mseed_path",
        "xml_path",
    ]
    cols = [k for k in preferred if k in all_keys] + sorted(
        k for k in all_keys if k not in preferred
    )

    # normalize rows
    norm_rows = []
    for r in results:
        if not isinstance(r, dict):
            row = {"value": str(r)}
        else:
            row = {}
            for k in cols:
                v = r.get(k, "")
                # convert lists/dicts/other non-primitives to JSON or string
                if isinstance(v, (dict, list)):
                    try:
                        v = json.dumps(v, ensure_ascii=False)
                    except Exception:
                        v = str(v)
                elif v is None:
                    v = ""
                else:
                    # keep numbers/strings as-is; cover other types
                    if not isinstance(v, (str, int, float, bool)):
                        v = str(v)
                row[k] = v
        norm_rows.append(row)

    # ensure output directory exists
    out_dir = os.path.dirname(results_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(norm_rows, columns=cols)
    df.to_csv(results_csv, index=False, encoding="utf-8")


def is_row_done(row):
    """
    Check the mseed output directory for this row to see if a file exists
    whose final `__` timestamp equals the row end_date.
    Returns True if done, False otherwise.
    """
    net = str(row["net"]).strip()
    sta = str(row["sta"]).strip()
    loc_field = str(row["loc"])
    # loc_field = "" if loc_field == "NA" else loc_field
    chan_prefix = str(row["chan"]).strip()

    record_sub = f"{net}_{sta}_{chan_prefix}_{loc_field}"
    mseed_path = os.path.join(OUTPUT_DIR, MSEED_DIR, net, record_sub)

    if not os.path.isdir(mseed_path):
        return False

    target_end = _format_end_for_filename(row["end_date"])

    try:
        for fname in os.listdir(mseed_path):
            if not fname.endswith(".mseed"):
                continue
            stem = os.path.splitext(fname)[0]

            # Check the CHAN field that it ends in Z
            chan_check = stem.split(".")[3]
            chan_check = chan_check.split("__")[0]  # remove any suffix after __
            if not chan_check.endswith("Z"):
                continue

            # filename parts expected like: NET.STA..CHAN__START__END
            # take last segment after the final '__'
            if "__" in stem:
                last = stem.rsplit("__", 1)[-1]
                if last == target_end:
                    return True
    except Exception:
        # any filesystem error -> treat as not done so row will be retried
        return False

    return False


def create_output_dirs(net, sta, chan_prefix, loc):
    """
    Create directories for the network if needed.
    """
    record_sub = f"{net}_{sta}_{chan_prefix}_{loc}"

    mseed_path = os.path.join(OUTPUT_DIR, MSEED_DIR, net, record_sub)
    xml_path = os.path.join(OUTPUT_DIR, STATIONXML_DIR, net, record_sub)

    os.makedirs(mseed_path, exist_ok=True)
    os.makedirs(xml_path, exist_ok=True)

    return mseed_path, xml_path


def worker(task):
    """
    Worker that performs a single row download.
    task: (idx, provider, row_dict)
    """
    idx, provider, row = task
    try:
        net = row["net"]
        sta = row["sta"]
        loc_field = str(row["loc"])
        loc = "*" if loc_field == "NA" else loc_field.strip()
        chan_prefix = str(row["chan"]).strip()
        channel = f"{chan_prefix}?"  # add ? automatically

        start = UTCDateTime(row["start_date"])
        end = UTCDateTime(row["end_date"])

        # Chunk length: month (30 days) but not longer than the full requested window
        total_window = end - start
        chunk_base = int(min(total_window, MONTH_SECONDS))

        max_attempts = 4
        attempt = 1

        print(
            f"[{idx}] Provider={provider} Downloading {net}.{sta} {channel} {start} -> {end} chunk={chunk_base}s"
        )

        # create output dirs using the raw loc field for naming (keeps 'NA' if present)
        mseed_path, xml_path = create_output_dirs(net, sta, chan_prefix, loc_field)

        while attempt <= max_attempts:
            chunklength = int(min(total_window, chunk_base))
            if chunklength < 1:
                chunklength = 1

            domain = GlobalDomain()
            restrictions = Restrictions(
                starttime=start,
                endtime=end,
                chunklength_in_sec=chunklength,
                network=net,
                station=sta,
                location=loc,
                channel=channel,
                reject_channels_with_gaps=False,
                minimum_length=0.0,
                minimum_interstation_distance_in_m=0.0,
            )

            try:
                mdl = MassDownloader(providers=[provider])
                mdl.download(
                    domain,
                    restrictions,
                    mseed_storage=mseed_path,
                    stationxml_storage=xml_path,
                )
                print(f"[{idx}] Done")
                return {
                    "idx": idx,
                    "status": "ok",
                    "provider": provider,
                    "net": net,
                    "sta": sta,
                }
            except Exception as e:
                err_text = repr(e) + " " + str(e)
                # Detect 413 / manifest-too-large responses from server text
                is_manifest_too_large = (
                    "Estimated manifest size" in err_text
                    or "Request Entity Too Large" in err_text
                    or "413" in err_text
                )

                if is_manifest_too_large:
                    # halve the base chunk and retry, unless already at minimum
                    if chunk_base <= MIN_CHUNK_SECONDS:
                        print(
                            f"[{idx}] Server denied request and chunk is already at minimum ({chunk_base}s). Giving up."
                        )
                        raise
                    old = chunk_base
                    chunk_base = max(MIN_CHUNK_SECONDS, chunk_base // 2)
                    print(
                        f"[{idx}] Server denied request (413). Reducing chunk base {old}s -> {chunk_base}s and retrying."
                    )
                    attempt += 1
                    continue
                raise e

    except Exception as e:
        print(
            f"[{idx}] ERROR provider={provider} net={row.get('net')} sta={row.get('sta')}: {e}"
        )
        return {"idx": idx, "status": "error", "error": str(e), "provider": provider}


def main():
    # Use explicit start method to avoid forking issues in some environments
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # start method already set
        pass

    df = pd.read_csv(CSV_FILE, dtype={"loc": str}, keep_default_na=False)

    # Filter down for ones that are False in completed column
    # df = df[df["completed"] == False]

    # Filter down to ones thar are True in started column
    # df = df[df["started"] == False]

    # Filter net to Y3 net (kept from original script)
    # df = df[df["provider"] == "IRIS"]

    required_cols = {"net", "sta", "loc", "chan", "start_date", "end_date", "provider"}

    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain: {required_cols}")

    # Build tasks for all rows (one task per CSV row)
    tasks = []
    skipped = 0
    for idx, row in df.reset_index(drop=True).iterrows():
        if is_row_done(row):
            skipped += 1
            continue
        tasks.append((int(idx), row["provider"], row.to_dict()))

    processes = 1
    print(
        f"Starting multiprocessing pool with {processes} processes for {len(tasks)} tasks (skipped {skipped} already done)"
    )

    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map(worker, tasks)

    save_results(results, results_csv=RESULTS_CSV)

    # Simple summary
    oks = sum(1 for r in results if r.get("status") == "ok")
    errs = sum(1 for r in results if r.get("status") == "error")
    print(
        f"Completed: {oks} succeeded, {errs} failed (skipped {skipped} previously done)"
    )


if __name__ == "__main__":
    main()
