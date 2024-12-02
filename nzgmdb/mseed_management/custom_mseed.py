from datetime import datetime
from pathlib import Path

import mseedlib
import numpy as np


class Trace:
    def __init__(self, data, channel_id):
        self.data = data
        self.channel_id = channel_id


class Mseed:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.start_time = None
        self.end_time = None
        self.dt = None
        file_split = file_path.stem.split("_")
        self.event_id = file_split[0]
        self.station = file_split[1]
        self.location = file_split[2]
        self.channel = file_split[3]
        self.traces = []

        # Read the file and extract information
        nptype = {"i": np.int32, "f": np.float32, "d": np.float64, "t": np.char}
        mstl = mseedlib.MSTraceList()
        mstl.read_file(str(file_path), unpack_data=False, record_list=True)

        for traceid in mstl.traceids():
            for segment in traceid.segments():
                # Fetch estimated sample size and type
                (sample_size, sample_type) = segment.sample_size_type
                dtype = nptype[sample_type]
                # Allocate NumPy array for data samples
                data_samples = np.zeros(segment.samplecnt, dtype=dtype)
                # Unpack data samples into allocated NumPy array
                segment.unpack_recordlist(
                    buffer_pointer=np.ctypeslib.as_ctypes(data_samples),
                    buffer_bytes=data_samples.nbytes,
                )
                # Get the component
                comp = mseedlib.sourceid2nslc(traceid.sourceid)[-1][-1]
                # Add the trace
                self.traces.append(Trace(data_samples, comp))
                # Add extra metadata
                if self.start_time is None:
                    self.start_time = datetime.fromtimestamp(segment.starttime_seconds)
                    self.end_time = datetime.fromtimestamp(segment.endtime_seconds)
                    self.dt = 1 / segment.samprate
