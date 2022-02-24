#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:40:40 2021

@author: emgordy
"""

# Train NNs
# this script was run on a cluster and takes days. Proceed with caution

import xarray as xr
import glob
import numpy as np

import math
import tensorflow as tf
import random
import tensorflow_probability as tfp
from scipy.stats import norm

ly1 = 3 # lead years (1 and 5, or 3 and 7)
ly2 = 7
runin = 60

#%% define some funcs

# function outputs model object
def loadmodel(HIDDENS,random_seed,ridgepenL2,lr,drate):
    
    n_layers = np.shape(HIDDENS)[0]
    
    # define the model
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dropout(rate = drate,seed=random_seed))
    
    model.add(tf.keras.layers.Dense(HIDDENS[0], activation='relu',
                bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                kernel_regularizer=tf.keras.regularizers.L2(l2=ridgepenL2)))#,

    # add hidden layers
    for layer in range(1,n_layers):
        model.add(tf.keras.layers.Dense(HIDDENS[layer], activation='relu',
                        bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed)))
                        #kernel_regularizer=tf.keras.regularizers.L1L2(l1=lassopenL1,l2=ridgepenL2)))#,
                        #kernel_regularizer=keras.regularizers.L2(ridgepenL2)))  
    
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
    

# function to generate model string to save under
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

# grab specific point from SST map
def makeYdata(Y_full,ilat,ilon,output_nodes):
    
    nzeros=output_nodes-1
    shape = Y_full.shape[0]
    Y_train_grab = Y_full[:,ilat,ilon]    
    Y_zeros = np.concatenate((np.expand_dims(Y_train_grab, axis=1),np.zeros((shape,nzeros))),axis=1) # add column of zeros to output for correct prediction format
    
    return Y_zeros

# custom loss function
def RegressLossExpSigma(y_true, y_pred):
    
    mu = tf.cast(y_pred[:,0], tf.float64)
    std = tf.math.exp(tf.cast(y_pred[:,1], tf.float64))
    norm_dist = tfp.distributions.Normal(mu,std)

    loss = -norm_dist.log_prob(tf.cast(y_true[:,0],tf.float64))
    loss = tf.cast(loss,tf.float64)

    return tf.reduce_mean(loss, axis=-1)

# split SST to train,val,testing data, then standardize by the training data
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

# split OHC into train,val,test, standardize by the training set
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
    
    X_train = np.divide(X_train,X_std) # make nans zeros
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

patience = 100 # early stopping patience
n_epochs = 1000 # training epochs
batchsize = 64
lr = 1e-4 # learning rate
ridgepenL2 = 0 #l2/ridge penalty, zero in this experiment
drate=0.8 # dropout rate in training
hiddens = [60,4] # hidden layer architecture, 2 layers, 60 nodes then 4 nodes

seeds = np.arange(0,10) # random seeds
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True) #early stopping

callbacks = [es_callback,]
loss_function = RegressLossExpSigma

#%% output specifics
nlat = latsel.shape[0]
nlon = lonsel.shape[0]
nseeds = seeds.shape[0]

latsel = np.asarray(latsel)
lonsel = np.asarray(lonsel)

folderstr = 'SSTfromOHC_regression' # folder to save models

#%% train 10 models at each point on the globe

output_nodes = 2
for ilat,lat in enumerate(latsel):
    for ilon,lon in enumerate(lonsel):   
        if np.isnan(Y_train_full[0,ilat,ilon]):
            print("lat="+str(lat)+ " lon=" +str(lon)+" over land")
        else:
            Y_train = makeYdata(Y_train_full,ilat,ilon,output_nodes)
            Y_val = makeYdata(Y_val_full,ilat,ilon,output_nodes)
        
            for iseed,seed in enumerate(seeds):
                              
                modelstrout = modelstrfunc(folderstr,lat,lon,ly1,ly2,hiddens,ridgepenL2,lr,seed,drate)
                strcheck = glob.glob(modelstrout)
                if len(strcheck) == 0:
                    print(modelstrout)
                    print("lat="+str(lat)+ " lon=" +str(lon)+"seed="+str(seed)) 
                    np.random.seed(seed)
                    tf.random.set_seed(seed)
                    random.seed(int(seed))         
                    
                    # load and train
                    model = loadmodel(hiddens,seed,ridgepenL2,lr,drate)
                    
                    print('training model')
                    history = model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batchsize, validation_data=(X_val, Y_val), 
                                    shuffle=True, verbose=0, callbacks=es_callback)
                    print('done training')
                    model.save_weights(modelstrout)
                
                else:
                    print('model exists')

