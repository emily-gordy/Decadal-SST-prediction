#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:22:22 2022

@author: emgordy
"""

#extract best random seed from validation data

import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys 
import matplotlib as mpl

sys.path.append('functions/')
import nndata
import pickle 

runin = 60
hiddens = [60,4]
lr = 1e-4
drate = 0.8

lys = [[1,5],[3,7]]

metricstrs = '../ExperimentMetricsAll_validation_ly*_hiddens'+str(hiddens[0])+str(hiddens[1])+'_lr'+str(lr)+'_drate'+str(drate)+'.nc'
metricstrs = glob.glob(metricstrs)
metricstrs = sorted(metricstrs)


#%% load in metric ds

metricds1 = xr.open_dataset(metricstrs[0])
testloss = np.asarray(metricds1.testloss)
arrdims = np.shape(testloss)

def nanargmin_fix(arr, axis): # fix to nanargmin that skips nans
    try:
        return np.nanargmin(arr, axis)
    except ValueError:
        return np.nan
    
#%% loop through lead year ranges
for ily in range(2):
    metricstrloop = metricstrs[ily]
    print(metricstrloop)
    ly1 = lys[ily][0]
    ly2 = lys[ily][1]
    Y_train, Y_val, Y_test, _, _ = nndata.makedata_heatx3_SSTonly_ALL(ly1,ly2) # load in SST data from external function
    metricsds = xr.open_dataset(metricstrloop)
    
    testpercentiles = np.asarray(metricsds.testpercentiles)
    test_muvar = np.asarray(metricsds.test_muvar)

    lat = np.asarray(metricsds.lat)
    lon = np.asarray(metricsds.lon)
    seeds = metricsds.seed
    percentiles = metricsds.percentile

    #calculate variance in Y_test
    Y_testvar = np.expand_dims(np.std(Y_test,axis=0),axis=2)
    #get rid of land + flatliners
    test_muvar = test_muvar/Y_testvar
    
    nomodel = np.isnan(testloss)
    flatliner = test_muvar<0.1

    testpercentiles[nomodel | flatliner,:] = np.nan
    test_muvar[nomodel | flatliner] = np.nan

    bestind = np.empty((arrdims[0],arrdims[1]))+np.nan
              
    for ilat,latloop in enumerate(lat):
        for ilon,lonloop in enumerate(lon):            
            bestind[ilat,ilon] = nanargmin_fix(testpercentiles[ilat,ilon,:,9],0) # grab seed NN with best MAE on 10% most confident
            
        
    # pickle the "bestind" array
    with open('bestind_val_ly'+str(ly1)+'-'+str(ly2)+'_array.pkl','wb') as f:
        pickle.dump(bestind,f)

