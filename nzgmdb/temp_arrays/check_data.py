import pandas as pd
import os
from obspy import UTCDateTime


OUTPUT_DIR = "/scratch/jobs/jri83/runs/tmp_array/mass_data_row_mp"
MSEED_DIR = "waveforms"
STATIONXML_DIR = "stationxml"

# month length in seconds (30 days)
MONTH_SECONDS = 30 * 24 * 3600


def _format_end_for_filename(end_dt):
    """Format an obspy UTCDateTime or parseable date string to the filename timestamp form."""
    if not isinstance(end_dt, UTCDateTime):
        end_dt = UTCDateTime(end_dt)
    return end_dt.strftime("%Y%m%dT%H%M%SZ")


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


def is_row_started(row):
    """
    Check whether *any* mseed file exists for this row.
    Returns True if at least one .mseed file is present.
    """
    net = str(row["net"]).strip()
    sta = str(row["sta"]).strip()
    loc_field = str(row["loc"])
    chan_prefix = str(row["chan"]).strip()

    record_sub = f"{net}_{sta}_{chan_prefix}_{loc_field}"
    mseed_path = os.path.join(OUTPUT_DIR, MSEED_DIR, net, record_sub)

    if not os.path.isdir(mseed_path):
        return False

    try:
        for fname in os.listdir(mseed_path):
            if fname.endswith(".mseed"):
                return True
    except Exception:
        return False

    return False


def evaluate_download_completeness(csv_file, output_csv):
    """
    Evaluate download state with two levels:
      - completed: full end_date reached
      - started: at least one file exists
    """
    df = pd.read_csv(csv_file, dtype={"loc": str}, keep_default_na=False)
    df = df[df["provider"] == "IRIS"].reset_index(drop=True)

    print(f"Evaluating {len(df)} rows for completeness...")

    # Two independent checks
    df["completed"] = df.apply(is_row_done, axis=1)
    df["started"] = df.apply(is_row_started, axis=1)

    total = len(df)

    completed = int(df["completed"].sum())
    started = int(df["started"].sum())

    partial = int(((df["started"]) & (~df["completed"])).sum())
    none = int((~df["started"]).sum())

    print("\n===== DOWNLOAD SUMMARY =====")
    print(f"Total rows           : {total}")
    print(f"Fully completed      : {completed}")
    print(f"Started (any data)   : {started}")
    print(f"Partial (started only): {partial}")
    print(f"No data at all       : {none}")

    # Rows with no data whatsoever
    if none > 0:
        print("\n===== ROWS WITH NO DATA =====")
        cols = ["provider", "net", "sta", "loc", "chan", "start_date", "end_date"]
        print(df.loc[~df["started"], cols].to_string(index=False))

    # Rows with partial data (useful for retries)
    if partial > 0:
        print("\n===== PARTIALLY DOWNLOADED ROWS =====")
        cols = ["provider", "net", "sta", "loc", "chan", "start_date", "end_date"]
        print(df.loc[df["started"] & ~df["completed"], cols].to_string(index=False))

    df.to_csv(output_csv, index=False)
    print(f"\nWrote evaluation CSV to:\n  {output_csv}")

    return df


CSV_FILE = "/scratch/jobs/jri83/runs/tmp_array/all_nz_sta_providers_desired_channels_mustang.csv"
OUTPUT_CSV = "/scratch/jobs/jri83/runs/tmp_array/download_completeness_evaluation.csv"

if __name__ == "__main__":
    evaluate_download_completeness(
        csv_file=CSV_FILE,
        output_csv=OUTPUT_CSV,
    )
