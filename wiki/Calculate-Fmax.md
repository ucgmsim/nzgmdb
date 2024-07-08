Computes Fmax from SNR data
(Is a converted Matlab script into python for ease of work with the pipeline)

# Prerequisites
SNR Calculation

# Process
Gets the row of the snr_metadata to iterate over ever record that had SNR computed.

## Filter
Calculates the scaled_nyquist_freq which is equal to
`scaled_nyquist_freq = (
        (1 / current_row["delta"].iloc[0])
        * 0.5
        * 0.8
    )`
Smoothes the SNR values based on a window of 5 from the center with a minimum amount of periods to 1
Then performs a check that there is a minimum of 5 frequency points between the frequency values of 0.5 and 10 Hz that have a value of SNR > 3 for vertical SNR and 5 for horizontal.

## Fmax Calc
If filter passes then gets the snr values that are higher than a frequency of 4 and finds the value at which the snr value is just less than the snr_threshold of 3.
If there is no found frequency point then grabs the last frequency point for fmax.
fmax then compare against the scaled_nyquist_freq and which ever is smaller is chosen for the fmax value.

# Output
The main fmax.csv file comtains 4 columns
* record id (evid_station_channel_location)
* fmax_000 (Frequency of fmax for component 000)
* fmax_090 (Frequency of fmax for component 090)
* fmax_ver (Frequency of fmax for component ver)

The fmax skipped file contains the record_id column and a reason column which details the variables used for the screening at runtime for records that could not find a fmax value.