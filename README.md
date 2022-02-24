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
