#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:49:27 2021

@author: emgordy
"""

#make deseasoned and detrended sst or ohc for longlong control run 

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import glob

writeout = False #boolean whether to actually write the files
grid = '90x45' # adjust these to use with SST/OHC
varstr = 'ohc100'
# varstr = 'sst'
# load in files
filestr1 = glob.glob('+varstr+'*CESM*'+grid+'*.nc')[0]

var_dataarray = xr.open_dataset(filestr1,decode_times=False)

if varstr =='sst' :
    var = var_dataarray.sst # for sea surface temp
elif varstr == 'ohc100':
    var = var_dataarray.ohc_100 # for OHC100 etc.
    varstr = 'ohc_heat100'
elif varstr == 'ohc_heat700':
    var = var_dataarray.heat700
elif varstr == 'ohc_heat300':
    var = var_dataarray.heat300
else: 
    print('var not recognised')

lat = var_dataarray.lat
lon = var_dataarray.lon
time = var_dataarray.time

var = np.asarray(var)


#%% cut-off first 100 years (1200 months) for spinup
spinup = 1200
var = var[spinup:,:,:]
time = time[spinup:]

#%% deseason the data 

arrdims = np.shape(var)
months = np.arange(12)

# empty to be filled with deseasoned data
var_deseasoned = np.empty(arrdims)

# to be filled with seasonal cycle, useful to check
varseasonal = np.empty((12,arrdims[1],arrdims[2]))

for ii in months: #index through months
    inds = np.arange(ii,arrdims[0],12) # all indexes corresponding to that month
    
    varmonth = var[inds,:,:] #pull ssts of month ii
    monthmean = np.nanmean(varmonth,axis=0) #take mean of those sst
    nomean = varmonth-monthmean # subtract monthly mean sst from that month data
    var_deseasoned[inds,:,:] = nomean #save to sst deseasoned
    varseasonal[ii,:,:] = monthmean
    
#%% 3rd degree polynomial detrend
xvec = np.arange(arrdims[0])
vartrendmat = np.empty(arrdims)

for ii in range(arrdims[1]):
    for jj in range(arrdims[2]):
        
        xvec = np.arange(arrdims[0]) 
        varin = var_deseasoned[:,ii,jj] # deal with nans
        xvecnan = xvec[~np.isnan(varin)]
        varinnan = varin[~np.isnan(varin)]
        
        if varinnan.shape[0] != 0:
            var_p = np.polyfit(xvecnan,varinnan,deg=3)
            var_poly = var_p[0]*(xvec**3) + var_p[1]*(xvec**2) + var_p[2]*(xvec) + var_p[3]
            print(var_p[0])
        else:
            var_poly = np.empty(arrdims[0])
            var_poly[:] = np.nan
        vartrendmat[:,ii,jj] = var_poly
        
# subtract 3rd degree polynomial time series at each grid point from the deseasoned data
var_deseasoned_detrended = var_deseasoned-vartrendmat

#%% Write out
var_dataset = xr.Dataset(
    {varstr: (("time","lat","lon"), var_deseasoned_detrended)},
    coords={
        "time": time,
        "lat": lat,
        "lon": lon
    },
)

#%% save to netCDF

varstrout = "/Users/emgordy/Documents/Experiments/DecadalPrediction_ForReal/data/"+varstr+"-deseasoned_detrended_"+str(spinup)+"-24000_"+grid+".nc" 

if writeout:
    var_dataset.to_netcdf(varstrout)
    print("wrote file")
else:
    print("did not write file")




