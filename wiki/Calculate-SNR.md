Computes SNR and FAS Intensity Measure files

# Prerequisites
Takes in an input of a phase arrival table and mseed files to write SNR files in the snr_fas directory.
- Parse Geonet (generates mseeds)
- Phase Arrival Table

# Process
A common frequency vector is used which is defined as
 0.01318257 → 100 (with 389 numbers)

## Waveform Processing
Reads the raw mseed files and performs some processing defined below:
* Demean and detrend
* Taper both ends by 5%
* Add 5s of zero padding at the start and end
* Remove sensitivity based on the Inventory (If failed to find will skip record)
* Rotate the components to NEZ (After rotation can still be shown as XYZ in odd occasions)
* Divide the data by gravity

Code can be found [here](https://github.com/ucgmsim/nzgmdb/blob/2fa80fa0917989c1103ed0a1e4821be7bb8f0e73/nzgmdb/data_processing/waveform_manipulation.py#L11)

## TP Selection
Attempts to grab the tp value (P-Wave phase arrival) from the phase arrival table, if fails will skip the record.
Currently just tries to match the given event_id and station, so if there are records with more than one channel type or location, even with the same channel, then it will find the P-wave for that entry and use it for all types of that same record for the given event id and station.

## SNR Calculation
First separates into signal and noise based on data being after and before time tp, respectively.
Checks that the noise duration is not less than 1s (If so, then the record is skipped).
Then applies a Hanning taper of 5% to both the start and ends of the signal and noise selections of the waveform.
The fa_spectrum is then calculated for signal and noise.
The absolute value is then taken of this output.
If smoothing is enabled (which is by default), then applies ko_matrix smoothing b = 40.
Interpolation is then done on the FAS to the common frequency vector.
We then set values to NaN for the frequencies that are outside the bounds of the sample rate / 2.

Then we calculate SNR (inter_signal / inter_noise is the interpolated signal / noise)

```python
snr = (inter_signal / np.sqrt(signal_duration)) / (
        inter_noise / np.sqrt(noise_duration)
    )
```

# Output
The snr_fas ouptut directory is layered like "year" / "event_id" / evid_station_channel_location_snr_fas.csv.
Metadata and skipped records are placed in the flatfiles directory.