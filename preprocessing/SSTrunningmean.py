#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:47:47 2021

@author: emgordy
"""

# make SST look forward running mean for predictions

import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt

run = 60 # set run (60 months)
writeout = False
grid = '72x36'

sststr = glob.glob('/Users/emgordy/Documents/Experiments/DecadalPrediction_ForReal/data/sst-d*detrended*1200*'+grid+'*.nc')[0]
lmstr = glob.glob('/Users/emgordy/Documents/Experiments/DecadalPrediction_ForReal/data/sf*'+grid+'*.nc')[0]

sst_dataset = xr.open_dataset(sststr)
lm_dataset = xr.open_dataset(lmstr)

sst = np.asarray(sst_dataset.sst)
lon = sst_dataset.lon
lat = sst_dataset.lat
lm = np.asarray(lm_dataset.sftlf)

#%% mask lf >50%
lf=50
sst[:,lm>lf] = np.nan

#%% running mean but conserve time nans

arrdims = np.shape(sst)

sst_run = np.empty(arrdims)
sst_run[:] = np.nan

timeind = np.arange(arrdims[0])
timevec = np.empty(arrdims[0])
timevec[:] = np.nan

for ind in range(arrdims[0]-run+1):
    sstint = np.nanmean(sst[ind:ind+run,:,:],axis=0)
    sst_run[ind,:,:] = sstint

#%% make xarray

sst_datasetout = xr.Dataset(
    {"sst": (("time","lat","lon"), sst_run)},
    coords={
        "time": timeind,
        "lat": lat,
        "lon": lon
    },
)

sststrout = "sst-runningmean"+str(run)+"_detrended_spinup_lookforward_lf"+str(lf)+"_r"+grid+".nc"

#%% and save
if writeout:
    sst_datasetout.to_netcdf(sststrout)
    print("wrote files")
else:
    print("did not write files")
