Computes SNR and FAS Intensity Measure files

# Prerequisites
Takes in an input of a phase arrival table and mseed files to write SNR files in the snr_fas directory.
So Need Phase Geonet and Phase Arrival Steps to be run.

# Process
A common frequency vector is used which is defined as
 0.01318257 → 100 (with 389 numbers)

## Waveform Processing
Reads the raw mseed files and performs some processing defined below:
* Demean and detrend
* Taper both ends by 5%
* Add zero padding of time 5s
* Remove sensitivity based on the Inventory (If failed to find will skip record)
* Rotate the components to NEZ (can still be XYZ?)
* Divide the data by gravity

## TP Selection
Attempts to grab the tp value (P-Wave phase arrival) from the phase arrival table, if fails will skip the record.
Currently just tries to match the given event_id and station, so if there is records with more than one channel type or location even with the same channel then it will find the P-wave for that entry and use it for all type of that same record for the given event id and station.

## SNR Calculation
First separates into signal and noise based on the tp value
Checks that the noise duration is not under 1s (If so then the record is skipped)
Then applies a hanning taper to the start and end of the waveform 5% to both the signal and noise
the fa_spectrum is then calculated for signal and noise
The absolute value is then taken of this output
If smoothing is enabled which is by default True Then applies ko_matrix smoothing b = 40
Interpolation is then done on the FAS to the common frequency vector
We then set values to NAN for the frequencies that are outside the bounds of the sample rate / 2
Then we calculate SNR (inter_signal / inter_noise is the interpolated signal / noise)
`snr = (inter_signal / np.sqrt(signal_duration)) / (`
        `inter_noise / np.sqrt(noise_duration)`
    `)`

# Output
The snr_fas ouptut directory is layered like "year" / "event_id" / evid_station_channel_location_snr_fas.csv
Metadata and skipped records are outputted to the flatfiles directory.