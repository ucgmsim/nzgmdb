Machine Learning Model to classify records and produce Fmin

# Prerequisites
- mseed files generated (Parse Geonet)
- Fmax

# Process
Currently a black box (Only understand input and output and unsure of the processing that occurs between)

PhaseNet is used to determine the p-wave however which is known, this is developed in Tensorflow v1 which causes issues
with python environments which stops this process from being able to be run automatically in the pipeline.

Uncertain as to what waveform processing is performed for fmin calculation.

# Output
gmc_predictions.csv which contains the score_mean, fmin_mean, and multi_mean columns for each record