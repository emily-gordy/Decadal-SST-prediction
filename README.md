# Decadal-SST-prediction
This is the github repo for _Incorporating uncertainty into neural networks enables identification of decadal state-dependent predictability_, Gordon and Barnes 2022

## Contents of this repo
* Preprocessing
* Training and saving ANNs
* Creating figures in Gordon and Barnes 2022

## Dependencies
The code in this repo require python 3.9 with the usual packages (numpy, scipy, xarray) plus the following packages
* [Tensorflow 2](https://www.tensorflow.org/install)
* [Tensorflow probability](https://www.tensorflow.org/probability/install)
* [Cmasher](https://cmasher.readthedocs.io/user/introduction.html#how-to-install)

## Required Data
* Ocean Heat Content (OHC) integrated to 100 m, 300 m, 700 m, on 45x90 grid
* Sea surface temperature (SST) on 36x72 grid
* Surface land fraction from CESM on both 45x90 grid and 36x72 grid

## Step-by-step repo contents
### Preprocessing
My pre-processing is laborious and chippy choppy so that I can check and double check the data at every step. This means there are a bunch of scripts to run in order, and intermediate netCDF files that get generated
1. deseason-detrend.py to deseason and detrend raw data (works for all OHC levels and SST)
2. OHCrunningmean.py apply lookback running mean to OHC
3. SSTrunningmean.py apply lookforward running mean to SST
4. nninput_output.py save the OHC and SST into the format for input/output into ANN
5. SSTpersistence.py make an extra SST netCDF4 as the persistence model

### TrainANNs
Two scripts: training the ANNs then loading back to extract some metrics
1. trainnn.py train 10 ANNs at each grid point in the ocean
2. loadANN.py load in each ANN and compute metrics for each model to be saved in netcdf
3. ANNmetrics.py function file called by loadANN.py to compute each metric
