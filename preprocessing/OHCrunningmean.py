#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:09:40 2021

@author: emgordy
"""

# Apply look back running mean to OHC

import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt

writeout = False
run = 60 # set run, 60 months/5 years

varin = 'heat700' #'heat100', 'heat300'

ohcstr = glob.glob('ohc_' + varin + '-d*detrended*1200*.nc')[0] #load in the ohc data
lmstr = glob.glob('sf*r90x45*.nc')[0] #load in surface land fraction grid

ohc_dataset = xr.open_dataset(ohcstr)
lm_dataset = xr.open_dataset(lmstr)

if varin == 'heat100':
    ohc = np.asarray(ohc_dataset.ohc_heat100)
elif varin == 'heat700':
    ohc = np.asarray(ohc_dataset.ohc_heat700)
elif varin == 'heat300':
    ohc = np.asarray(ohc_dataset.ohc_heat300)
# ohc = np.asarray(ohc_dataset.ohc_heat300)

lon = ohc_dataset.lon
lat = ohc_dataset.lat
lm = np.asarray(lm_dataset.sftlf)

#%% we gonna mask 50 percent land fraction (mask out grid points >50% land)
lf=50
ohc[:,lm>lf] = np.nan

#%% running mean but conserve time nans

arrdims = np.shape(ohc)

ohc_run = np.empty(arrdims)
ohc_run[:] = np.nan

timeind = np.arange(arrdims[0])
timevec = np.empty(arrdims[0])
timevec[:] = np.nan

for ind in range(arrdims[0]-run+1):
    ohcint = np.nanmean(ohc[ind:ind+run,:,:],axis=0)
    ohc_run[ind+run-1,:,:] = ohcint
    

#%% make xarray

ohc_datasetout = xr.Dataset(
    {"ohc": (("time","lat","lon"), ohc_run)},
    coords={
        "time": timeind,
        "lat": lat,
        "lon": lon
    },
)

ohcstrout = "ohc_"+varin+"-runningmean%d_detrended_spinup_lookback_lf%d_r90x45.nc" %(run,lf) #

#%% and save
if writeout:
    ohc_datasetout.to_netcdf(ohcstrout)
    print("wrote files")
else:
    print("did not write files")
    
    
