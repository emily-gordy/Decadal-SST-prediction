#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:27:45 2021

@author: emgordy
"""

# load back all ANNs

import xarray as xr
import glob
import numpy as np
import sys

sys.path.append('functions/') # path to grab the external functions called here
import ANNmetrics

import math
import tensorflow as tf
import random
import tensorflow_probability as tfp
from scipy.stats import norm

ly1 = 1 # lead years (1 to 5, or 3 to 7)
ly2 = 5
runin = 60

#%% define some funcs

# load in the model, don't need as much here because we are loading in the pre-saved weights
def loadmodel(HIDDENS,random_seed,ridgepenL2,lr,drate):
    
    n_layers = np.shape(HIDDENS)[0]
    
    # define the model
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(HIDDENS[0], activation='relu',input_shape=(7947,),
                bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                kernel_regularizer=tf.keras.regularizers.L2(l2=ridgepenL2)))#,

    # add hidden layers
    for layer in range(1,n_layers):
        model.add(tf.keras.layers.Dense(HIDDENS[layer], activation='relu',
                        bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed))) 
    
    # final layer
    model.add(tf.keras.layers.Dense(output_nodes,activation=None,
                    bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                    kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),))
                    # kernel_regularizer=keras.regularizers.L1L2(l1=0,l2=0)))
    
    
    model.compile(optimizer=tf.keras.optimizers.SGD(lr),  # optimizer
                loss=loss_function,   # loss function   
                metrics=[tf.keras.metrics.MeanAbsolutePercentageError(),
                         tf.keras.metrics.MeanAbsoluteError(),
                         tf.keras.metrics.MeanSquaredError()]) 
    
    return model
    


def modelstrfunc(folderstr,lat1,lon1,ly1,ly2,HIDDENS,ridgepenL2,lr,random_seed,drate):
    n_layers = np.shape(HIDDENS)[0]
    if n_layers == 1:
        modelstrout = "models/"+ folderstr +"/polydetrendlat%d_lon%d_ly%d-%d_layers%d_%d_ridge%f_drate%f_lr%f_seed%d.h5" %(
            lat1,lon1,ly1,ly2,n_layers,HIDDENS[0],ridgepenL2,lr,drate,random_seed)
    elif n_layers == 2:
         modelstrout = "models/"+ folderstr +"/polydetrendlat%d_lon%d_ly%d-%d_layers%d_%d%d_ridge%f_drate%f_lr%f__seed%d.h5" %(
            lat1,lon1,ly1,ly2,n_layers,HIDDENS[0],HIDDENS[1],ridgepenL2,lr,drate,random_seed)  
    elif n_layers == 3:
         modelstrout = "models/"+ folderstr +"/polydetrendlat%d_lon%d_ly%d-%d_layers%d_%d%d%d_ridge%f_drate%f_lr%f_seed%d.h5" %(
            lat1,lon1,ly1,ly2,n_layers,HIDDENS[0],HIDDENS[1],HIDDENS[2],ridgepenL2,drate,lr,random_seed)  
    else:
          modelstrout = "models/"+ folderstr +"/polydetrendlat%d_lon%d_ly%d-%d_layers%d_%d%d%d%d_ridge%f_drate%f_lr%f_seed%d.h5" %(
            lat1,lon1,ly1,ly2,n_layers,HIDDENS[0],HIDDENS[1],HIDDENS[2],HIDDENS[3],ridgepenL2,drate,lr,random_seed)  
          print('warning layer limit reached in string')
             
    return modelstrout

def makeYdata(Y_full,ilat,ilon,output_nodes):
    
    nzeros=output_nodes-1
    shape = Y_full.shape[0]
    Y_train_grab = Y_full[:,ilat,ilon]    
    Y_zeros = np.concatenate((np.expand_dims(Y_train_grab, axis=1),np.zeros((shape,nzeros))),axis=1)
    
    return Y_zeros

def RegressLossExpSigma(y_true, y_pred):
    
    mu = tf.cast(y_pred[:,0], tf.float64)
    std = tf.math.exp(tf.cast(y_pred[:,1], tf.float64))
    norm_dist = tfp.distributions.Normal(mu,std)

    loss = -norm_dist.log_prob(tf.cast(y_true[:,0],tf.float64))
    loss = tf.cast(loss,tf.float64)

    return tf.reduce_mean(loss, axis=-1)


def makedata_heatx3_SSTonly(ly1=1,ly2=5):

    sststr = glob.glob("sst-output_detrended*spinup*%d-%d*heatx3.nc" %(ly1,ly2))[0]

    sstoutputset = xr.open_dataset(sststr)

    sst = sstoutputset.sst
    lat = sstoutputset.lat
    lon = sstoutputset.lon

    sstsel = np.asarray(sst)
    latsel = np.asarray(lat)
    lonsel = np.asarray(lon)
    
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

def makedata_heatx3_OHConly(ly1=1,ly2=5,runin=60):
    
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

#%% experiment data

X_train,X_val,X_test = makedata_heatx3_OHConly(ly1,ly2,runin)
Y_train_full,Y_val_full,Y_test_full,latsel,lonsel = makedata_heatx3_SSTonly(ly1,ly2)

#%% experiment variables

patience = 100
n_epochs = 1000
batchsize = 64
lr = 1e-4
ridgepenL2 = 0
drate=0.8
hiddens = [60,4]#,50,10]

# drate=0.8
# hiddens = [60,4]#,50,10]

seeds = np.arange(0,10)
loss_function = RegressLossExpSigma

    
#%% output specifies

nlat = latsel.shape[0]
nlon = lonsel.shape[0]
nseeds = seeds.shape[0]

latsel = np.asarray(latsel)
lonsel = np.asarray(lonsel)

testloss = np.empty((nlat,nlon,nseeds))+np.nan
testMAE = np.empty((nlat,nlon,nseeds))+np.nan
testsign = np.empty((nlat,nlon,nseeds))+np.nan
testspread = np.empty((nlat,nlon,nseeds))+np.nan
testpercentiles = np.empty((nlat,nlon,nseeds,10))+np.nan
test_muvar = np.empty((nlat,nlon,nseeds))+np.nan
testpCC = np.empty((nlat,nlon,nseeds,10))+np.nan
testsCC = np.empty((nlat,nlon,nseeds,10))+np.nan
testpvalp = np.empty((nlat,nlon,nseeds,10))+np.nan
testpvals = np.empty((nlat,nlon,nseeds,10))+np.nan
testaccuracy = np.empty((nlat,nlon,nseeds,10))+np.nan
testMAEpercentiles_scaled = np.empty((nlat,nlon,nseeds,10))+np.nan

# testtercilesacc = np.empty((nlat,nlon,nseeds,10))+np.nan

folderstr = 'SSTfromOHC_ASHA'

#%%

output_nodes = 2
for ilat,lat in enumerate(latsel):
    for ilon,lon in enumerate(lonsel):
        # ilon = np.where(lon==lonsel)[0][0] # picking from original latxlon grid
        # ilat = np.where(lat==latsel)[0][0]   
        if np.isnan(Y_train_full[0,ilat,ilon]):
            # testloss[ilat,ilon,:] = np.nan
            print("lat="+str(lat)+ " lon=" +str(lon)+" over land")
        else:
            # Y_train = makeYdata(Y_train_full,ilat,ilon,output_nodes)
            
            Y_test = makeYdata(Y_test_full,ilat,ilon,output_nodes)
        
            for iseed,seed in enumerate(seeds):
                Y_val = makeYdata(Y_val_full,ilat,ilon,output_nodes)
                Y_test = makeYdata(Y_test_full,ilat,ilon,output_nodes)
                modelstrout = modelstrfunc(folderstr,lat,lon,ly1,ly2,hiddens,ridgepenL2,lr,seed,drate)
                print(modelstrout)
                print("lat="+str(lat)+ " lon=" +str(lon)+"seed="+str(seed)) 
                np.random.seed(seed)
                tf.random.set_seed(seed)
                random.seed(int(seed))         
                
                # load and train
                model = loadmodel(hiddens,seed,ridgepenL2,lr,drate)
                model.load_weights(modelstrout)
                
                y_pred = model.predict(X_val)
                
                testloss[ilat,ilon,iseed]=ANNmetrics.logp_GAUSS(y_pred,Y_val)
                testMAE[ilat,ilon,iseed]=ANNmetrics.MAE(y_pred,Y_val)
                testsign[ilat,ilon,iseed]=ANNmetrics.halfNhalf_GAUSS(y_pred,Y_val)
                testspread[ilat,ilon,iseed]=ANNmetrics.predspread_GAUSS(y_pred,Y_val)
                testpercentiles[ilat,ilon,iseed,:],_,_=ANNmetrics.MAEpercentiles_GAUSS(y_pred,Y_val)
                test_muvar[ilat,ilon,iseed]=np.nanstd(y_pred[:,0])
                
                testpCC[ilat,ilon,iseed,:],testpvalp[ilat,ilon,iseed,:]=ANNmetrics.Pearsonpercentile(y_pred,Y_val)
                testsCC[ilat,ilon,iseed,:],testpvals[ilat,ilon,iseed,:]=ANNmetrics.Spearmanpercentile(y_pred,Y_val)
                testaccuracy[ilat,ilon,iseed,:]=ANNmetrics.ClassificationAccuracy(y_pred,Y_val)
                
                # testMAEpercentiles_scaled[ilat,ilon,iseed,:] = ANNmetrics.MAEpercentile_scaledvar(y_pred, Y_val, y_valpred)
                # testtercilesacc[ilat,ilon,iseed,:] = ANNmetrics.tercileaccuracy(y_pred, Y_test)
                # testACC

#%%

metricstr = 'ExperimentMetricsAll_validation_ly'+str(ly1)+'-'+str(ly2)+'_hiddens'+str(hiddens[0])+str(hiddens[1])+'_lr'+str(lr)+'_drate'+str(drate)+'.nc'
percentiles = np.flipud(np.arange(10,110,10))
# percentiles2 = np.flipud(np.arange(10,110,10))
#%%
ds = xr.Dataset(
    {"testloss":(("lat","lon","seed"), testloss),
    "testMAE":(("lat","lon","seed"), testMAE),
    "testsign":(("lat","lon","seed"), testsign),
    "testspread":(("lat","lon","seed"), testspread),
    "test_muvar":(("lat","lon","seed"), test_muvar),
    "testpercentiles":(("lat","lon","seed","percentile"), testpercentiles),
    "testpCC":(("lat","lon","seed","percentile"), testpCC),
    "testpvalp":(("lat","lon","seed","percentile"), testpvalp),
    "testsCC":(("lat","lon","seed","percentile"), testsCC),
    "testpvals":(("lat","lon","seed","percentile"), testpvals),    
    "testaccuracy":(("lat","lon","seed","percentile"), testaccuracy),
    "testMAEscaled":(("lat","lon","seed","percentile"), testMAEpercentiles_scaled),
    },
    coords={
        "lat": latsel,
        "lon": lonsel,
        "seed": seeds,
        "percentile": percentiles
        },
    )
ds.to_netcdf(metricstr)


# ds = xr.Dataset(
#     {"testACC":(("lat","lon","seed"), testACC),
#     "testpval":(("lat","lon","seed"), testpval)},
#     coords={
#         "lat": latsel,
#         "lon": lonsel,
#         "seed": seeds,
#         },
#     )
# ds.to_netcdf(metricstr)

# metricstrterciles = 'ExperimentMetricsTerciles_ly'+str(ly1)+'-'+str(ly2)+'_hiddens'+str(hiddens[0])+str(hiddens[1])+'_lr'+str(lr)+'_drate'+str(drate)+'.nc'

# ds = xr.Dataset(
#     {"testtercileacc":(("lat","lon","seed","percentile"), testtercilesacc),
#     },
#     coords={
#         "lat": latsel,
#         "lon": lonsel,
#         "seed": seeds,
#         "percentile": percentiles
#         },
#     )
# ds.to_netcdf(metricstrterciles)

#%%

# bestACC = np.max(np.abs(testaccuracy[:,:,:,0]),axis=2)
# bestACC[bestACC<0.4] = np.nan
# import matplotlib.pyplot as plt
# import cmasher as cmr
# plt.contourf(bestACC,levels=np.arange(0.5,1.05,0.05),cmap=cmr.flamingo)
# plt.colorbar()

















