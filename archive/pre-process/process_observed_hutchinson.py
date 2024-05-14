"""Processes V1A or miniseed files from observed events and saves these as text files (in units g)"""
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List

import numpy as np
import obspy
import pandas as pd
from scipy import signal
from obspy.clients.fdsn import Client as FDSN_Client

import qcore.timeseries as ts
from geoNet_file import GeoNet_File
from process import adjust_gf_for_time_delay, SMD

SMD_DEFAULT_KWARGS = {
    "lowcut": 0.05,
    "gpInt": False,
    "highcut": 50.0,
    "fs": (1.0 / 0.005),
    "order": 4,
    "ft": 1.0,
    "output": None,
}

g = 9.810


TEXT_OUTPUT_TYPES = ["accBB", "velLF", "velBB"]


class Record:
    """Represents a observed event record

    Attributes
    ----------
    acc_h1, acc_h2: array of floats
        The unrotated horizontal acceleration components (in g)
    acc_000, acc_090: array of floats
        The roated horizontal acceleration components (in g)
    acc_v: array of floats
        The vertical aacceleration component (in g)
    dt: float
        Time step size
    station_name: string
    ffp: path
        File path of the original record file
    """

    def __init__(
        self,
        acc_h1: np.ndarray,
        acc_h2: np.ndarray,
        acc_000: np.ndarray,
        acc_090: np.ndarray,
        acc_v: np.ndarray,
        dt: float,
        station_name: str,
        ffp: Path,
    ):
        self.acc_h1 = acc_h1
        self.acc_h2 = acc_h2
        self.acc_000 = acc_000
        self.acc_090 = acc_090
        self.acc_v = acc_v

        self.dt = dt

        self.station_name = station_name
        self.ffp = ffp


def rotate(acc_h1, acc_h2, angle_000: float):
    """
            N
        W<-- -->E
            S

    GeoNet comp_1st and comp_2nd angles axis angles are measured
    from N. We rotate clock wise with angle theta where

        theta = 360 - comp_1st.angle

    [comp_090, comp_180] = Rot(theta) * [comp_1st, comp_2nd]
    Then perform a reflection in the y-axis
    comp_000 = -comp_180
    self.rot_angle:
           theta = 360 - comp_1st.angle
    acc_000:
            positive axis 1 parallel to E
    acc_090:
            positive axis 2 parallel to N

    """
    # For compatibility with orientation as defined by Brendon et al.
    rot_angle = -(90.0 - angle_000)
    R = rot_matrix(rot_angle)
    acc_090 = R[0, 0] * acc_h1 + R[0, 1] * acc_h2
    acc_000 = R[1, 0] * acc_h1 + R[1, 1] * acc_h2

    return acc_000, acc_090


def pre_process(acc: np.ndarray):
    """De-mean and de-trends the acceleration series"""
    return signal.detrend(acc - acc.mean(), type="linear")


def rot_matrix(theta: float):
    """
    Compute the rotation matrix

    Parameters
    ----------
    theta: float
        Rotation angle in degrees

    Returns
    -------
    array:
        Rotation matrix
    """
    theta = theta * np.pi / 180.0
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


def process(
    record: Record,
    output_format: str,
    output_type: str = None,
    write_unrotated: bool = False,
    f_min_df: pd.DataFrame = None,
    f_max_df: pd.DataFrame = None,
    **kwargs,
):
    # Fill any parameter gaps with the default params
    cur_default_kwargs = SMD_DEFAULT_KWARGS.copy()
    cur_default_kwargs["fs"] = 1.0 / record.dt

    if f_min_df is not None:
        # Get the low and highcut from the csv files
        min_rows = f_min_df[f_min_df.record == record.ffp.name.split(".")[0]]
        if len(min_rows) > 0:
            cur_default_kwargs["lowcut"] = min_rows.fmin_mean.max()

        xml = record.ffp.parent.parent.parent.glob("*.xml")
        evid = [x for x in xml][0].name.split(".")[0]
        sta = record.ffp.name.split(".")[0].split("_")[2]
        chan = record.ffp.name.split(".")[0].split("_")[4]

        max_rows = f_max_df[f_max_df.ev_sta == f"{evid}_{sta}_{chan}"]
        # Find rows that contain the given site
        if len(max_rows) > 0:
            cur_default_kwargs["highcut"] = min(max_rows["fmax_000"].values[0], max_rows["fmax_090"].values[0], max_rows["fmax_ver"].values[0])
        else:
            cur_default_kwargs["highcut"] = 1.0 / (2.5 * record.dt)
    else:
        cur_default_kwargs["highcut"] = 1.0 / (2.5 * record.dt)

    kwargs = {**cur_default_kwargs, **kwargs}

    # Perform integration - also does bandpass
    smd_comp_h1, smd_comp_h2 = None, None
    if write_unrotated:
        smd_comp_h1 = SMD(record.acc_h1, record.dt, **kwargs)
        smd_comp_h2 = SMD(record.acc_h2, record.dt, **kwargs)
    smd_comp_000 = SMD(record.acc_000, record.dt, **kwargs)
    smd_comp_090 = SMD(record.acc_090, record.dt, **kwargs)
    smd_comp_v = SMD(record.acc_v, record.dt, **kwargs)

    # Get the displacement series
    displacement_000 = smd_comp_000.dispBB
    displacement_090 = smd_comp_090.dispBB
    displacement_v = smd_comp_v.dispBB

    # Fit six-order polynomial to the displacement series
    coeff_000 = np.polyfit(np.arange(len(displacement_000)), displacement_000, 6)
    coeff_090 = np.polyfit(np.arange(len(displacement_090)), displacement_090, 6)
    coeff_v = np.polyfit(np.arange(len(displacement_v)), displacement_v, 6)

    # Find the second derivative of the coefficients
    coeff_000_2nd = np.polyder(coeff_000, 2)
    coeff_090_2nd = np.polyder(coeff_090, 2)
    coeff_v_2nd = np.polyder(coeff_v, 2)

    # Generate polynomial values from the coefficients
    poly_000 = np.polyval(coeff_000_2nd, np.arange(len(smd_comp_000.accBB)))
    poly_090 = np.polyval(coeff_090_2nd, np.arange(len(smd_comp_090.accBB)))
    poly_v = np.polyval(coeff_v_2nd, np.arange(len(smd_comp_v.accBB)))

    # Subtract the polynomial from the original acc series
    smd_comp_000.setAccBB(smd_comp_000.accBB - poly_000)
    smd_comp_090.setAccBB(smd_comp_090.accBB - poly_090)
    smd_comp_v.setAccBB(smd_comp_v.accBB - poly_v)

    if output_format == "text":
        assert output_type in TEXT_OUTPUT_TYPES
        output_dir = record.ffp.parent / output_type
        output_dir.mkdir(exist_ok=True)
        ts.seis2txt(
            smd_comp_000.__getattribute__(output_type),
            smd_comp_000.dt,
            str(output_dir) + "/",
            record.station_name,
            "000",
        )
        ts.seis2txt(
            smd_comp_090.__getattribute__(output_type),
            smd_comp_090.dt,
            str(output_dir) + "/",
            record.station_name,
            "090",
        )
        ts.seis2txt(
            smd_comp_v.__getattribute__(output_type),
            smd_comp_v.dt,
            str(output_dir) + "/",
            record.station_name,
            "ver",
        )

        if write_unrotated:
            ts.seis2txt(
                smd_comp_h1.__getattribute__(output_type),
                smd_comp_h1.dt,
                str(output_dir) + "/",
                record.station_name,
                "H1",
            )
            ts.seis2txt(
                smd_comp_h2.__getattribute__(output_type),
                smd_comp_h2.dt,
                str(output_dir) + "/",
                record.station_name,
                "H2",
            )
    else:
        raise NotImplementedError()


def get_comp_trace(traces: List[obspy.Trace], components: str):
    """Retrieves the trace for the specified components"""
    return [
        cur_trace.data
        for cur_trace in traces
        if cur_trace.stats.channel[-1] in components
    ][0]


def process_v1a_file(
    ffp: Path,
    output_format: str,
    output_type,
    write_unrotated: bool,
    ix: int,
    n_files: int,
):
    """Processes a single V1A file and saves as text file, to be used with pool.starmap"""
    print(f"Processing V1A file {ffp.name}, {ix + 1}/{n_files}")
    try:
        gf = GeoNet_File(ffp.name, base_dir=str(ffp.parent))

        if gf.comp_1st.acc.size < 5.0 / gf.comp_1st.delta_t:
            print(f"{ffp.name} has less than 5 secs of data, skipping!")
            return None

        # When appended zeroes at the beginning of the record are removed, the
        # record might then be empty, skip processing in such a case
        agf = adjust_gf_for_time_delay(gf)
        if agf.comp_1st.acc.size <= 10:
            print(f"No elements in {ffp.name}, skipping!")
            return None

        # Pre-processing (de-mean & detrend)
        acc_h1, acc_h2 = (
            pre_process(gf.comp_1st.acc),
            pre_process(gf.comp_2nd.acc),
        )
        acc_v = pre_process(gf.comp_up.acc)

        # Rotate to get horizontal components in north-south & east-west direction
        acc_000, acc_090 = rotate(acc_h1, acc_h2, gf.comp_1st.angle)

        # Create a record
        cur_record = Record(
            acc_h1,
            acc_h2,
            acc_000,
            acc_090,
            acc_v,
            gf.comp_1st.delta_t,
            ffp.name.split(".")[0].split("_")[2],
            ffp,
        )

        # Process
        process(
            cur_record,
            output_format,
            output_type=output_type,
            write_unrotated=write_unrotated,
        )
        return True

    except Exception as ex:
        print(f"Failed to process V1A file {ffp} with exception:\n {ex}")
        return False


def process_v1a_files(
    ffps: List[Path],
    output_format: str,
    output_type: str = None,
    write_unrotated: bool = False,
    n_procs: int = 1,
):
    """Processes the specified V1A files"""
    n_files = len(ffps)
    with mp.Pool(n_procs) as p:
        success_mask = p.starmap(
            process_v1a_file,
            [
                (cur_ffp, output_format, output_type, write_unrotated, ix, n_files)
                for ix, cur_ffp in enumerate(ffps)
            ],
        )

    return success_mask


def process_miniseed_file(
    ffp: Path,
    inventory: obspy.Inventory,
    output_format: str,
    output_type: str,
    write_unrotated: bool,
    ix: int,
    n_files: int,
    f_min_df: pd.DataFrame = None,
    f_max_df: pd.DataFrame = None,
):
    """Processes a single miniseed file and saves as text file, to be used with pool.starmap"""
    print(f"Processing miniseed file {ffp.name}, {ix + 1}/{n_files}")
    try:
        st = obspy.read(str(ffp))
        if len(st) == 3:
            # Get start and endtime of data
            starttime = st[0].stats.starttime
            endtime = st[0].stats.endtime

            # Remove mean and trend
            st.detrend('demean')
            st.detrend()

            for tr in st:
                # Taper the data 5% at each end
                st.taper(0.05,side='both',max_length=5)
                # Add zero-padding (five seconds beginning and end)
                tr.trim(tr.stats.starttime-5,tr.stats.endtime,pad=True,fill_value=tr.data[0])
                tr.trim(tr.stats.starttime,tr.stats.endtime+5,pad=True,fill_value=tr.data[-1])


    #             st = st.remove_response(inventory=inventory, water_level=0, output='ACC')
            # Check if data is from CPLB (which has no instrument response) or not
            if st[0].stats.station == 'CPLB':
                # Divide trace counts by 10^6 to convert to units g
                for tr in st:
                    tr.data = tr.data / 10**6
                if st[0].stats.channel[1] == 'H':
                    st.differentiate()

                acc_h1 = []
                acc_h2 = []

                if write_unrotated:
                    for tr in st:
                        if tr.stats.component == '1':
                            # Get the unrotated horizontal components
                            tr.trim(starttime,endtime)
                            acc_h1 = get_comp_trace(st.traces, ["1"])
                        if tr.stats.component == '2':
                            tr.trim(starttime,endtime)
                            acc_h2 = get_comp_trace(st.traces, ["2"])

                # Remove zero pads but retain transient filter length
                st.trim(starttime,endtime)

                # Convert to ACC if it isn't already
                assert st.traces[0].stats.channel[-2] in ["N", "H"]

                # Pre-processing (de-mean & detrend)
                acc_000 = get_comp_trace(st.traces, ["N","Y","1"])
                acc_090 = get_comp_trace(st.traces, ["E","X","2"])
                acc_v = get_comp_trace(st.traces, "Z")

            else:
                # Remove sensitivity
                st = st.remove_sensitivity(inventory=inventory)
                if st[0].stats.channel[1] == 'H':
                    st.differentiate()

                # If necessary, high-pass filter using a two-pass, two-pol acausal BW filter. Select different
                #     corners for horizontal and vertical components
    #             st.filter(type='highpass',freq=0.04,corners=2,zerophase=True)
    #             st.filter(type='bandpass',freqmin=0.05,freqmax=50,corners=4)
                acc_h1 = []
                acc_h2 = []

                if write_unrotated:
                    for tr in st:
                        if tr.stats.component == '1':
                            # Get the unrotated horizontal components
                            tr.trim(starttime,endtime)
                            acc_h1 = get_comp_trace(st.traces, ["1"]) / g
                        if tr.stats.component == '2':
                            tr.trim(starttime,endtime)
                            acc_h2 = get_comp_trace(st.traces, ["2"]) / g

                # Rotate
                st.rotate("->ZNE", inventory=inventory)

                # Remove zero pads but retain transient filter length
                st.trim(starttime,endtime)

                # Convert to ACC if it isn't already
                assert st.traces[0].stats.channel[-2] in ["N", "H"]

                # Pre-processing (de-mean & detrend) & convert to units g
                acc_000 = get_comp_trace(st.traces, ["N"]) / g
                acc_090 = get_comp_trace(st.traces, ["E"]) / g
                acc_v = get_comp_trace(st.traces, ["Z"]) / g

            # Process
            cur_record = Record(
                acc_h1,
                acc_h2,
                acc_000,
                acc_090,
                acc_v,
                st.traces[0].stats.delta,
                st.traces[0].stats.station,
                ffp,
            )
            process(
                cur_record,
                output_format,
                output_type=output_type,
                write_unrotated=write_unrotated,
                f_min_df=f_min_df,
                f_max_df=f_max_df,
            )
            return True
        else:
            print(f"File {ffp.name} does not have 3 components, skipping!")
            return None
    except Exception as ex:
        print(f"Failed to process miniseed file {ffp} with exception:\n {ex}")
        return False


def process_miniseed_files(
    ffps: List[Path],
    output_format: str,
    output_type: str = None,
    write_unrotated: bool = False,
    n_procs: int = 1,
    gmc_dir: Path = None,
    f_max_ffp: Path = None,
):
    """Processes the specified miniseed files"""
    print("Loading the station inventory (this may take a few seconds)")
    client_NZ = FDSN_Client("GEONET")
    client_IU = FDSN_Client('IRIS')
    inventory_NZ = client_NZ.get_stations(level='response')
    inventory_IU = client_IU.get_stations(network='IU',station='SNZO',level='response')
    inventory = inventory_NZ+inventory_IU

    # Load the f_min df
    if gmc_dir is None:
        f_min_df = None
    else:
        # Read every csv file in the gmc_dir and create a dataframe
        f_min_df = pd.DataFrame()
        for csv_file in gmc_dir.glob("*.csv"):
            cur_df = pd.read_csv(csv_file)
            f_min_df = pd.concat([f_min_df, cur_df], ignore_index=True)
    if f_max_ffp is None:
        f_max_df = None
    else:
        f_max_df = pd.read_csv(f_max_ffp)

    n_files = len(ffps)
    with mp.Pool(n_procs) as p:
        success_mask = p.starmap(
            process_miniseed_file,
            [
                (
                    cur_ffp,
                    inventory,
                    output_format,
                    output_type,
                    write_unrotated,
                    ix,
                    n_files,
                    f_min_df,
                    f_max_df,
                )
                for ix, cur_ffp in enumerate(ffps)
            ],
        )

    return success_mask


def main(
    base_data_dir: Path,
    input_format: str,
    output_format: str,
    output_type: str,
    write_unrotated: bool,
    n_procs: int,
    gmc_dir: Path = None,
    f_max_ffp: Path = None,
):
    # Find all files to process
    print("Finding all relevant files to process")
    ffps = list(base_data_dir.rglob(f"*.{input_format}"))
    print(f"Found {len(ffps)} files to process")

    if input_format.lower() == "v1a":
        success_mask = process_v1a_files(
            ffps,
            output_format,
            output_type=output_type,
            write_unrotated=write_unrotated,
            n_procs=n_procs,
        )
    else:
        success_mask = process_miniseed_files(
            ffps,
            output_format,
            output_type=output_type,
            write_unrotated=write_unrotated,
            n_procs=n_procs,
            gmc_dir=gmc_dir,
            f_max_ffp=f_max_ffp,
        )

    unprocessed_files = [
        f"{ffps[ix]}\n" for ix, result in enumerate(success_mask) if result is None
    ]
    if len(unprocessed_files) > 0:
        print(
            f"The following were not processed as they did not meet the data requirements."
            f"Check the log for the reason.\n{''.join(unprocessed_files)}"
        )

    failed_files = [
        f"{ffps[ix]}\n" for ix, result in enumerate(success_mask) if result is False
    ]
    if len(failed_files) > 0:
        print(
            f"The following files failed to process, "
            f"check the log for exceptions.\n{''.join(failed_files)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Converts V1A or miniseed files to processed binary or text files."
    )

    parser.add_argument(
        "base_data_dir",
        type=str,
        help="The base directory for the observed GM records. "
        "All V1A or miniseed files below this directory will be processed.",
    )
    parser.add_argument(
        "input_format",
        type=str,
        help="The data format of the observed record files",
        choices=["V1A", "mseed"],
    )
    parser.add_argument(
        "output_format",
        type=str,
        help="The format of the output file, either 3 text files "
        "(one for each component) or a single binary file.",
        choices=["text", "binary"],
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Output type to save the data in, only relevant when saving in text format",
        default="accBB",
        choices=TEXT_OUTPUT_TYPES,
    )
    parser.add_argument(
        "--n_procs", type=int, help="Number of processes to use", default=1
    )
    parser.add_argument(
        "--write_unrotated",
        help="If specified then the two unrotated components are also saved (.H1 & .H2)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--gmc_dir",
        type=Path,
        help="Directory containing the GMC filter csv file information",
        default=None,
    )
    parser.add_argument(
        "--f_max_ffp",
        type=Path,
        help="File containing the maximum frequency information",
        default=None,
    )

    args = parser.parse_args()
    main(
        Path(args.base_data_dir),
        args.input_format,
        args.output_format,
        args.output_type,
        args.write_unrotated,
        args.n_procs,
        args.gmc_dir,
        args.f_max_ffp,
    )
