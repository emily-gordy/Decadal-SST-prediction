#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:12:53 2021

@author: emgordy
"""

# SST grids for the persistence model (basically making sure the time lines up, that's why we load in the OHC, it's a whole lotta checking)

import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt

writeout = False

run = 60 # set input run (12 or 24 or 60 months)
lf = 50
grid1 = 72
ohc700str = glob.glob('/Users/emgordy/Documents/Experiments/DecadalPrediction_ForReal/data/ohc_heat700-r*%d*detrended*spinup*%d*.nc' %(run,lf))[0]
ohc700_dataset = xr.open_dataset(ohc700str)

ohc700 = np.asarray(ohc700_dataset.ohc)
lon = ohc700_dataset.lon
lat = ohc700_dataset.lat

sststr = glob.glob('/Users/emgordy/Documents/Experiments/DecadalPrediction_ForReal/data/sst-r*detrended*spinup*lookback*%d*%d*.nc' %(lf,grid1))[0]
sst_dataset = xr.open_dataset(sststr)
lon2 = sst_dataset.lon
lat2 = sst_dataset.lat

sstforward = glob.glob('sst-r*detrended*spinup*lookforward*%d*%d*.nc' %(lf,grid1))[0]
sstf_dataset = xr.open_dataset(sststr)

sst = np.asarray(sst_dataset.sst)
sstf = np.asarray(sstf_dataset.sst)

#%% set lag for each persistence model, 12 for lead year 1-5, 36 for lead 3-7 
lag = 12
ly1 = int(lag/12)
ly2 = ly1+4

arrdims = np.shape(ohc700)

ohc700flat = np.reshape(ohc700,(arrdims[0],arrdims[1]*arrdims[2]))
time = np.arange(arrdims[0])

ohc_input = np.asarray(ohc700flat)

#%%
sst_output = sstf[lag:]

if lag != 0:
    timeohc = time[:-lag]
else:
    timeohc = time
timesst = time[lag:]
#%%
# remove burnt ends
cut1 = np.shape(sst_output)[0]

ohc_input = ohc_input[:cut1]
sstp = sst[:cut1]
timeohc = timeohc[:cut1]

ohc_input[np.isnan(ohc_input)] = 0
#%%
# where lookback is nan
cut2 = run-1
ohc_input = ohc_input[cut2:]
sstp = sstp[cut2:]
sst_output = sst_output[cut2:]

timeohc = timeohc[cut2:]
timesst = timesst[cut2:]
#%%
# where lookforward is nan i.e. more burnt ends
sstpoint = sst_output[:,16,12]
ohc_input = ohc_input[~np.isnan(sstpoint)]
sst_output = sst_output[~np.isnan(sstpoint)]
sstp = sstp[~np.isnan(sstpoint)]
timeohc = timeohc[~np.isnan(sstpoint)]
timesst = timesst[~np.isnan(sstpoint)]

#%% make netCDF out
sst_datasetout = xr.Dataset(
    {"sst": (("time","lat","lon"), sstp)},
    coords={
        "time": timesst,
        "lat": lat2,
        "lon": lon2
    },
)


sststrout = "sst-persistence_detrended_spinup_runningmean%d_ly%d-%d_lf%d_r72x36_heatx3.nc" %(run,ly1,ly2,lf)

#%% and save
if writeout:
    print("wrote some files")
    sst_datasetout.to_netcdf(sststrout)
else:
    print("did not write files")
