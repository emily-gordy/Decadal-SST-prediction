#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 14:17:53 2021

@author: emgordy
"""

# functions for loading stuff from netcdf files

import glob
import numpy as np
import xarray as xr
from scipy.stats import norm
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs

def getmask(): # get the land mask in OHC

    ohcstr = glob.glob("ohc-i*heatx3*detrended*spinup*%d*%d-%d*.nc" %(60,1,5))[0]
    
    ohcinputset = xr.open_dataset(ohcstr)
    
    ohc = np.asarray(ohcinputset.ohc)
    # remove land from ohc
    
    ohclm = ~np.isnan(ohc[0,:]/ohc[0,:])
    return ohclm

def addcyclicpoint(matrix,lonin): # add cyclic point to latxlon grid for plotting
    addpoint = np.expand_dims(matrix[:,0],axis=1)
    addlon = [lonin[0]]
    
    matcon = np.concatenate((matrix,addpoint),axis=1)
    loncon = np.concatenate((lonin,addlon),axis=0)
    
    return matcon, loncon

def makedata_heatx3_OHConly(ly1=1,ly2=5,runin=60): # make the OHC ANN input data
    
    ohcstr = glob.glob("ohc-i*heatx3*detrended*spinup*%d*%d-%d*.nc" %(runin,ly1,ly2))[0]
    
    ohcinputset = xr.open_dataset(ohcstr)
    
    ohc = np.asarray(ohcinputset.ohc)
    
    # remove land from ohc
    
    ohclm = ~np.isnan(ohc[0,:]/ohc[0,:])
    ohc = ohc[:,ohclm]
    
    samplesize = np.shape(ohc)[0]
    
    train_val_test = [0,0.7,0.85,1]
    # ohc = ohc[:,8100:]
    ohc_std = []
    sst_std = []
    
    for ii in range(3):
        split1 = int(samplesize*train_val_test[ii])
        split2 = int(samplesize*train_val_test[ii+1])
        
        ohcint = ohc[split1:split2,:]
        # ohcint[np.isnan(ohcint)] = 0
        
        ohc_std.append(ohcint)
        
    X_train = ohc_std[0]
    X_val = ohc_std[1]
    X_test = ohc_std[2]

    X_std = np.nanstd(X_train,axis=0)

    
    X_train = np.divide(X_train,X_std)
    X_train[np.isnan(X_train)] = 0
    
    X_val = np.divide(X_val,X_std)
    X_val[np.isnan(X_val)] = 0
    
    X_test = np.divide(X_test,X_std)
    X_test[np.isnan(X_test)] = 0

    
    return X_train,X_val,X_test
  

def latlon90x45(): # load in latxlon data on 45x90 grid
    filestr = glob.glob("ohc_heat700*CESM*90x45*.nc")[0]
    ds = xr.open_dataset(filestr)
    lat = ds.lat
    lon = ds.lon
    
    return lat, lon

def gaussian_filter(matrix,width): # spatial gaussian filter that skips nans
    matrixtile = np.tile(matrix,3)
    matdimbig = matrixtile.shape
    
    xvec = np.arange(matdimbig[1])
    yvec = np.arange(matdimbig[0])
    
    smoothmat = np.empty(matdimbig)
    
    lats = np.linspace(-90,90,matdimbig[0])
    
    latweights = np.cos(lats*np.pi/180)
    latweightstile = np.transpose(np.tile(latweights,(matdimbig[1],1)))
    latweightstile = latweightstile*np.sum(latweightstile)
    
    for mu_y in yvec:
        for mu_x in xvec:
            xgaussian = norm.pdf(xvec,loc=mu_x,scale=width)
            ygaussian = norm.pdf(yvec,loc=mu_y,scale=width)
            xx,yy = np.meshgrid(xgaussian,ygaussian)
            filterout = xx*yy
            # filterout = filterout*latweightstile
            
            datapoint = filterout*matrixtile
            smoothmat[mu_y,mu_x] = np.nanmean(datapoint)
    
    matind1 = matdimbig[1]/3
    smoothmatsmall = smoothmat[:,int(matind1):int(2*matind1)]
    return smoothmatsmall
            
def sstpersistence_ALL(ly1,ly2): #load in the SST persistence time series and split into train/val/test
    datastr = glob.glob("sst-persistence*ly%d-%d*.nc" %(ly1,ly2))[0]

    sst_ds= xr.open_dataset(datastr)
    sstsel = sst_ds.sst
    latsel = sst_ds.lat
    lonsel = sst_ds.lon

    sstsel = np.asarray(sstsel)
    
    #samplesize = np.shape(sstsel)[0]
    if ly1 == 1:
        samplesize = 22670 # magic number!!! Only works for ly1-5
    elif ly1 == 3:
        samplesize = 22646 # magic number!!! Only works for ly3-7
    train_val_test = [0,0.7,0.85,1]
    sst_std = []
    
    for ii in range(3):
        split1 = int(samplesize*train_val_test[ii])
        split2 = int(samplesize*train_val_test[ii+1])
        
        sstint = sstsel[split1:split2,:,:]
        sst_std.append(sstint)
        
    
    Y_train = sst_std[0]
    Y_val = sst_std[1]
    Y_test = sst_std[2]
    
    Y_mean = np.nanmean(Y_train,axis=0)
    Y_std = np.nanstd(Y_train,axis=0)
    
    Y_train = np.divide((Y_train-Y_mean),Y_std)
    
    Y_val = np.divide((Y_val-Y_mean),Y_std)
    
    Y_test = np.divide(Y_test-Y_mean,Y_std)
    
    return Y_train, Y_val, Y_test, latsel, lonsel

def makedata_heatx3_SSTonly_ALL(ly1=1,ly2=5): # make the SST output data

    sststr = glob.glob("sst-output_detrended*spinup*%d-%d*heatx3.nc" %(ly1,ly2))[0]

    sstoutputset = xr.open_dataset(sststr)

    sstsel = sstoutputset.sst
    latsel = sstoutputset.lat
    lonsel = sstoutputset.lon

    sstsel = np.asarray(sstsel)
    
    samplesize = np.shape(sstsel)[0]
    
    train_val_test = [0,0.7,0.85,1]
    # ohc = ohc[:,8100:]
    sst_std = []
    
    for ii in range(3):
        split1 = int(samplesize*train_val_test[ii])
        split2 = int(samplesize*train_val_test[ii+1])
        
        sstint = sstsel[split1:split2,:,:]
        sst_std.append(sstint)
        
    
    Y_train = sst_std[0]
    Y_val = sst_std[1]
    Y_test = sst_std[2]
    
    Y_mean = np.nanmean(Y_train,axis=0)
    Y_std = np.nanstd(Y_train,axis=0)
    
    Y_train = np.divide((Y_train-Y_mean),Y_std)
    
    Y_val = np.divide((Y_val-Y_mean),Y_std)
    
    Y_test = np.divide(Y_test-Y_mean,Y_std)
    
    return Y_train,Y_val,Y_test,latsel,lonsel

def AMVindex(run,ly1): # make AMV index time series and split to train/val/test

    grid = '72x36'
        
    sststr = glob.glob('sst-d*detrended*1200*'+grid+'*.nc')[0]
    lmstr = glob.glob('sf*'+grid+'*.nc')[0]

    sstds = xr.open_dataset(sststr)
    lon = sstds.lon
    lat = sstds.lat
    sst = sstds.sst

    lfds = xr.open_dataset(lmstr)
    lf = lfds.sftlf
    #AMV --> no removing global mean because of removal of model drift already
    Atllon1 = 280
    Atllon2 = 360
    Atllat1 = 0
    Atllat2 = 80

    Atllat = lat.sel(lat=slice(Atllat1,Atllat2))
    Atllon = lon.sel(lon=slice(Atllon1,Atllon2))

    ATLlonxlat = np.meshgrid(Atllon,Atllat)[1]
    ATLweights = np.cos(ATLlonxlat*np.pi/180)

    AtlanticSST = np.asarray(sst.sel(lon=slice(Atllon1,Atllon2),lat=slice(Atllat1,Atllat2)))
    AtlanticLF = np.asarray(lf.sel(lon=slice(Atllon1,Atllon2),lat=slice(Atllat1,Atllat2)))

    AtlanticSST[:,AtlanticLF>50] = np.nan

    AtlanticSSTw = AtlanticSST*ATLweights
    AMV1 = np.nanmean(AtlanticSSTw,axis=(1,2))
    AMV1 = np.divide(AMV1,np.std(AMV1)) #standardize
    #now make it comparable with ANN input, running mean and cut off burnt ends
    if ly1 == 1:
        samplesize = 22670 # magic number!!! Only works for ly1-5
    elif ly1 == 3:
        samplesize = 22646
        
    AMVrun = []
    for ii in range(AMV1.shape[0]-run):
        runint = np.nanmean(AMV1[ii:ii+run])
        AMVrun.append(runint)
    
    AMVrun = np.asarray(AMVrun)
    AMVrun = np.divide(AMVrun,np.std(AMVrun))
    train_val_test = [0,0.7,0.85,1]
    
    AMVtrain = AMVrun[int(train_val_test[0]*samplesize):int(train_val_test[1]*samplesize)]
    AMVval = AMVrun[int(train_val_test[1]*samplesize):int(train_val_test[2]*samplesize)]    
    AMVtest = AMVrun[int(train_val_test[2]*samplesize):int(train_val_test[3]*samplesize)]
    
    return AMVtrain,AMVval,AMVtest
    
def IPOindex(run,ly1): # make IPO index and divide into train/val/test
    grid = '72x36'
        
    sststr = glob.glob('sst-d*detrended*1200*'+grid+'*.nc')[0]
    sstds = xr.open_dataset(sststr)
    sst = sstds.sst

    Nlat1 = 25
    Nlat2 = 45
    Nlon1 = 140
    Nlon2 = 215

    Elat1 = -10
    Elat2 = 10
    Elon1 = 170
    Elon2 = 270

    Slat1 = -50
    Slat2 = -15
    Slon1 = 150
    Slon2 = 200

    NPac = np.asarray(sst.sel(lon=slice(Nlon1,Nlon2),lat=slice(Nlat1,Nlat2)))
    SPac = np.asarray(sst.sel(lon=slice(Slon1,Slon2),lat=slice(Slat1,Slat2)))
    EPac = np.asarray(sst.sel(lon=slice(Elon1,Elon2),lat=slice(Elat1,Elat2)))

    NPacweights = np.expand_dims(np.cos((np.arange(Nlat1,Nlat2,5)+0.5)*np.pi/180),axis=(0,2))
    SPacweights = np.expand_dims(np.cos((np.arange(Slat1,Slat2,5)+0.5)*np.pi/180),axis=(0,2))
    EPacweights = np.expand_dims(np.cos((np.arange(Elat1,Elat2,5)+0.5)*np.pi/180),axis=(0,2))

    NPacw = NPac*NPacweights
    SPacw = SPac*SPacweights
    EPacw = EPac*EPacweights

    IPOindex = np.nanmean(EPacw,axis=(1,2)) - 0.5*(np.nanmean(NPacw,axis=(1,2))+np.nanmean(SPacw,axis=(1,2)))
    IPOrun = []
    
    for ii in range(IPOindex.shape[0]-run):
        runint = np.mean(IPOindex[ii:ii+run])
        IPOrun.append(runint)
        
    IPOrun = np.asarray(IPOrun)
    IPOrun = np.divide(IPOrun,np.std(IPOrun))
    
    if ly1 == 1:
        samplesize = 22670 # magic number!!! Only works for ly1-5
    elif ly1 == 3:
        samplesize = 22646
        
    train_val_test = [0,0.7,0.85,1]
    
    IPOtrain = IPOrun[int(train_val_test[0]*samplesize):int(train_val_test[1]*samplesize)]
    IPOval = IPOrun[int(train_val_test[1]*samplesize):int(train_val_test[2]*samplesize)]    
    IPOtest = IPOrun[int(train_val_test[2]*samplesize):int(train_val_test[3]*samplesize)]    
    
    return IPOtrain, IPOval, IPOtest    
    

