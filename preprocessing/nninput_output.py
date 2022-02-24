#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:20:17 2021

@author: emgordy
"""

# MAKE NN inputs with OHC levels
# Make NN output with SST 
# time lines up so the same index in each is an input/output pair

import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt

writeout = False

run = 60 # set input run (60 months)
lf = 50
grid1 = 72 # sst grid 
ohc700str = glob.glob('/Users/emgordy/Documents/Experiments/DecadalPrediction_ForReal/data/ohc_heat700-r*%d*detrended*spinup*%d*.nc' %(run,lf))[0]
ohc700_dataset = xr.open_dataset(ohc700str)

ohc700 = np.asarray(ohc700_dataset.ohc)
lon = ohc700_dataset.lon
lat = ohc700_dataset.lat

ohc300str = glob.glob('/Users/emgordy/Documents/Experiments/DecadalPrediction_ForReal/data/ohc_heat300-r*%d*detrended*spinup*%d*.nc' %(run,lf))[0]
ohc300_dataset = xr.open_dataset(ohc300str)
ohc300 = np.asarray(ohc300_dataset.ohc)

ohc100str = glob.glob('/Users/emgordy/Documents/Experiments/DecadalPrediction_ForReal/data/ohc_heat100-r*%d*detrended*spinup*%d*.nc' %(run,lf))[0]
ohc100_dataset = xr.open_dataset(ohc100str)
ohc100 = np.asarray(ohc100_dataset.ohc)

sststr = glob.glob('/Users/emgordy/Documents/Experiments/DecadalPrediction_ForReal/data/sst-r*detrended*spinup*lookforward*%d*%d*.nc' %(lf,grid1))[0]
sst_dataset = xr.open_dataset(sststr)
lon2 = sst_dataset.lon
lat2 = sst_dataset.lat

sst = np.asarray(sst_dataset.sst)

#%% set the lead time, lag is number of months to first lead year, so lag=36 means lead year 3-7 
lag = 36
ly1 = int(lag/12)
ly2 = ly1+4

arrdims = np.shape(ohc700)
ohc_input = []

# flatten arrays
ohc700flat = np.reshape(ohc700,(arrdims[0],arrdims[1]*arrdims[2]))
ohc300flat = np.reshape(ohc300,(arrdims[0],arrdims[1]*arrdims[2]))
ohc100flat = np.reshape(ohc100,(arrdims[0],arrdims[1]*arrdims[2]))

time = np.arange(arrdims[0])

for ind in range(arrdims[0]-lag):
    ohcint = np.concatenate((ohc700flat[ind,:],ohc300flat[ind,:],ohc100flat[ind,:]),axis=0)
    ohc_input.append(ohcint)

ohc_input = np.asarray(ohc_input)
sst_output = sst[lag:]

if lag != 0:
    timeohc = time[:-lag]
else:
    timeohc = time
timesst = time[lag:]
#%%
# remove burnt ends
cut1 = np.shape(sst_output)[0]
ohc_input = ohc_input[:cut1]
timeohc = timeohc[:cut1]

ohc_input[np.isnan(ohc_input)] = 0

# where lookback is nan
cut2 = run-1
ohc_input = ohc_input[cut2:]
sst_output = sst_output[cut2:]

timeohc = timeohc[cut2:]
timesst = timesst[cut2:]

# where lookforward is nan i.e. more burnt ends
sstpoint = sst_output[:,16,12]
ohc_input = ohc_input[~np.isnan(sstpoint)]
sst_output = sst_output[~np.isnan(sstpoint)]
timeohc = timeohc[~np.isnan(sstpoint)]
timesst = timesst[~np.isnan(sstpoint)]

#%% make netCDF files
latxlon1 = np.meshgrid(lat,lon)
latxlon1 = np.reshape(latxlon1[0],90*45)
latxlon1 = np.tile(latxlon1,3)

latxlon2 = np.meshgrid(lat2,lon2)
latxlon2 = np.reshape(latxlon2[0],72*36)
latxlon2 = np.tile(latxlon2,3)

ohc_datasetout = xr.Dataset(
    {"ohc": (("time","space"), ohc_input)},
    coords={
        "time": timeohc,
        "space": latxlon1
    },
)

sst_datasetout = xr.Dataset(
    {"sst": (("time","lat","lon"), sst_output)},
    coords={
        "time": timesst,
        "lat": lat2,
        "lon": lon2
    },
)


ohcstrout = "ohc-input_heatx3_detrended_spinup_runningmean%d_lf%d_ly%d-%d_r90x45.nc" %(run,lf,ly1,ly2)
sststrout = "sst-output_detrended_spinup_runningmean%d_ly%d-%d_lf%d_r72x36_heatx3.nc" %(run,ly1,ly2,lf)

#%% and save
if writeout:
    print("wrote some files")
    sst_datasetout.to_netcdf(sststrout)
    ohc_datasetout.to_netcdf(ohcstrout)
else:
    print("did not write files")







