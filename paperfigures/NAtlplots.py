#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:33:40 2022

@author: emgordy
"""

#PAPER CODE
# plot input output maps for NAtl point and scatter and MAE vs predictions and AMV/IPO histograms


import xarray as xr
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys 
import matplotlib as mpl

sys.path.append('functions/')
import ANNplots
import ANNmetrics
import math
import tensorflow as tf
import random
import tensorflow_probability as tfp
import nndata
from scipy.stats import norm, linregress, ttest_ind
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmasher as cmr

import pickle
from scipy.stats import spearmanr, pearsonr

mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi']= 150
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['font.size'] = 12

params = {"ytick.color" : "k",
          "xtick.color" : "k",
          "axes.labelcolor" : "k",
          "axes.edgecolor" : "k"}
plt.rcParams.update(params)

ly1 = 1
ly2 = 5
runin = 60

X_train,X_val,X_test = nndata.makedata_heatx3_OHConly(ly1,ly2,runin)
Y_train_full,Y_val_full,Y_test_full,latsel,lonsel = nndata.makedata_heatx3_SSTonly_ALL(ly1,ly2)

#%% define some funcs

def loadmodel(HIDDENS,random_seed,ridgepenL2,lr,drate):
    
    n_layers = np.shape(HIDDENS)[0]
    
    # define the model
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(HIDDENS[0], activation='relu',input_shape=(7947,),
                bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                kernel_regularizer=tf.keras.regularizers.L2(l2=ridgepenL2)))#,

    # model.add(tf.keras.layers.Dropout(rate = 0.2,seed=random_seed))
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
    


def modelstrfunc(folderstr,lat1,lon1,ly1,ly2,HIDDENS,ridgepenL2,lr,random_seed,drate):
    n_layers = np.shape(HIDDENS)[0]
    if n_layers == 1:
        modelstrout = "../models/"+ folderstr +"/polydetrendlat%d_lon%d_ly%d-%d_layers%d_%d_ridge%f_drate%f_lr%f_seed%d.h5" %(
            lat1,lon1,ly1,ly2,n_layers,HIDDENS[0],ridgepenL2,lr,drate,random_seed)
    elif n_layers == 2:
         modelstrout = "../models/"+ folderstr +"/polydetrendlat%d_lon%d_ly%d-%d_layers%d_%d%d_ridge%f_drate%f_lr%f__seed%d.h5" %(
            lat1,lon1,ly1,ly2,n_layers,HIDDENS[0],HIDDENS[1],ridgepenL2,lr,drate,random_seed)  
    elif n_layers == 3:
         modelstrout = "../models/"+ folderstr +"/polydetrendlat%d_lon%d_ly%d-%d_layers%d_%d%d%d_ridge%f_drate%f_lr%f_seed%d.h5" %(
            lat1,lon1,ly1,ly2,n_layers,HIDDENS[0],HIDDENS[1],HIDDENS[2],ridgepenL2,drate,lr,random_seed)  
    else:
          modelstrout = "../models/"+ folderstr +"/polydetrendlat%d_lon%d_ly%d-%d_layers%d_%d%d%d%d_ridge%f_drate%f_lr%f_seed%d.h5" %(
            lat1,lon1,ly1,ly2,n_layers,HIDDENS[0],HIDDENS[1],HIDDENS[2],HIDDENS[3],ridgepenL2,drate,lr,random_seed)  
          print('warning layer limit reached in string')
             
    return modelstrout

def RegressLossExpSigma(y_true, y_pred):
    
    mu = tf.cast(y_pred[:,0], tf.float64)
    std = tf.math.exp(tf.cast(y_pred[:,1], tf.float64))
    norm_dist = tfp.distributions.Normal(mu,std)

    loss = -norm_dist.log_prob(tf.cast(y_true[:,0],tf.float64))
    loss = tf.cast(loss,tf.float64)

    return tf.reduce_mean(loss, axis=-1)

def makeYdata(Y_full,ilat,ilon):
    
    nzeros = output_nodes-1
    shape = Y_full.shape[0]
    Y_train_grab = Y_full[:,ilat,ilon]    
    Y_zeros = np.concatenate((np.expand_dims(Y_train_grab, axis=1),np.zeros((shape,nzeros))),axis=1)
    
    return Y_zeros

#%% experiment variables

patience = 100
batchsize = 64
lr = 1e-4
ridgepenL2 = 0
drate=0.8
hiddens = [60,4]

metricstr = 'mean_absolute_error'
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
lrate_callback = tf.keras.callbacks.LearningRateScheduler(step_decay)

loss_function = RegressLossExpSigma
output_nodes=2

callbacks = [es_callback,]

folderstr = "SSTfromOHC_ASHA"


#%% picking lats and lons
latsel = np.asarray(latsel)
lonsel = np.asarray(lonsel)

# Subpolar Gyre, North Atlantic
regionstr = 'Subpolar Gyre, North Atlantic'
lat1 = 52.5
lon1 = 325

# first grab best seed from each region

with open('bestind_val_ly'+str(ly1)+'-'+str(ly2)+'_array.pkl', 'rb') as f:
    bestind = pickle.load(f)


latind = np.argwhere(latsel==lat1)[0,0]
lonind = np.argwhere(lonsel==lon1)[0,0]

Y_train = makeYdata(Y_train_full,latind,lonind)
Y_val = makeYdata(Y_val_full,latind,lonind)
Y_test = makeYdata(Y_test_full,latind,lonind)
seed = int(bestind[latind,lonind])\
    
modelstrout = modelstrfunc(folderstr,lat1,lon1,ly1,ly2,hiddens,ridgepenL2,lr,seed,drate)

# load model and add weights
model = loadmodel(hiddens,seed,ridgepenL2,lr,drate)                            
model.load_weights(modelstrout)


y_pred = model.predict(X_test) 

mu = y_pred[:,0]
sigma = np.exp(y_pred[:,1])
timeseries_mu = mu
timeseries_sigma = sigma
timeseries_true = Y_test[:,0]

X_input = X_test
X_output = X_test[72:,:]

MAEbins,_,_ = ANNmetrics.MAEpercentiles_GAUSS(y_pred,Y_test)
MAE = MAEbins
spearman,pvals = ANNmetrics.Pearsonpercentile(y_pred, Y_test)

#%% load in indices

_,_,AMVindex = nndata.AMVindex(runin,ly1)
_,_,IPOindex = nndata.IPOindex(runin,ly1)


#%% plot composite of most confident inputs for each models place

projection = ccrs.EqualEarth(central_longitude=225)
transform = ccrs.PlateCarree()

latplot,lonplot = nndata.latlon90x45()
lonplot = np.arange(0,364,4)

contours = np.arange(-1.5,1.6,0.1)
cmap = cmr.fusion_r
cmapSST = 'RdBu_r'

#%%
Y_true = Y_test[:,0]
Y_mu = y_pred[:,0]
Y_sigma = np.exp(y_pred[:,1])

perc_low = Y_mu - Y_sigma
perc_high = Y_mu + Y_sigma

# Y_inrange = (Y_true < perc_high) & (Y_true > perc_low)
abserr = np.abs(Y_true-Y_mu)    

plt.figure(figsize=(5,4))
ax1 = plt.subplot(1,1,1)
# plt.subplot(2,1,1)
ax1.vlines(0,-3,3,colors='gray',linestyle='--',linewidth=0.5,zorder=1)
ax1.hlines(0,-3,3,colors='gray',linestyle='--',linewidth=0.5,zorder=3)
# plt.errorbar(YTnR[::10],YMnR[::10],YSnR[::10],marker='.',linewidth=0.4,color='xkcd:raspberry',ls='None',label='miss')
# plt.errorbar(YTR[::10],YMR[::10],YSR[::10],marker='.',linewidth=0.4,color='xkcd:maroon',ls='None',label='hit')
ax1.errorbar(Y_true[::10],Y_mu[::10],Y_sigma[::10],marker='.',linewidth=0.4,color='xkcd:maroon',ls='None',label='miss',zorder=5)
ax1.plot(np.arange(-3,4),np.arange(-3,4),linewidth=0.8,color='gray',zorder=7)
ax1.errorbar(truegrab,mugrab,sigmagrab,color='xkcd:fuchsia',zorder=9,marker='o')
plt.xlabel('True SST anomaly (standardized)')
plt.ylabel('Predicted SST anomaly (standardized)')
plt.title(r' '+str(lat1)+'$^{\circ}$N, '+str(lon1)+'$^{\circ}$E')
plt.title(r'b. Truth vs Predicted')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.tight_layout()
# plt.savefig('figures/nicescatter.png',dpi=300)
plt.show()

#%%
ticklabs = ['100','90','80','70','60','50','40','30','20','10']

plt.figure(figsize=(4,3))
# plt.subplot(2,1,2)
plt.plot(np.arange(10),MAEbins,color='xkcd:raspberry',linewidth=2)
plt.plot(np.arange(10),MAEbins,color='xkcd:maroon',marker='o')
plt.xlim(-0.1,9.1)
plt.ylim(0.35,0.55)
plt.xticks(ticks=np.arange(10),labels=ticklabs)
plt.yticks(ticks=np.arange(0.35,0.55,0.05))
plt.xlabel('Percent most confident')
plt.ylabel('Mean Absolute Error')
plt.title('c. Confidence vs MAE')
plt.grid(axis='y')
plt.tight_layout()

# plt.savefig('figures/niceMAEconfidence_paper.png',dpi=300)
plt.show()

#%% 
text1 = ['OHC100','OHC300','OHC700']
text2 = ['OHC at input','OHC at output']
text3 = ['a.','b.','c.']

contours2 = np.arange(-0.7,0.75,0.05)
        
sigmaboo = timeseries_sigma<np.percentile(sigmaloop,20) # choose data in lowest 20 percentile
muboo =  timeseries_mu0 # choose only positivetive predictions

trueboo = Y_true>0

sigmaboo_output = sigmaboo[:-72]
muboo_output = muboo[:-72]
trueboo_output = trueboo[:-72]

X_inputloop = np.nanmean(X_input[(sigmaboo & muboo & trueboo),:],axis=0)       
ohclm = nndata.getmask()        
X_inputplot = np.empty(12150)+np.nan
X_inputplot[ohclm] = X_inputloop

plt.figure(figsize=(7,7))

for iplot in range(3):
    X_plot = np.reshape(X_inputplot[iplot*4050:(iplot+1)*4050],(45,90))
    X_plot,_ = nndata.addcyclicpoint(X_plot, lonplot)
    a1 = plt.subplot(3,1,iplot+1,projection=projection)
    c1=a1.contourf(lonplot,latplot,X_plot,contours,cmap=cmap,transform=transform,extend='both')
    a1.coastlines(color='gray')
    a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', facecolor='gray'))
    a1.scatter(lon1,lat1,color='xkcd:crimson',transform=transform,marker='o')
    a1.text(-0.1,0.3,text1[iplot],rotation='vertical',fontsize=14,transform=a1.transAxes)
    a1.text(0.05,0.9,text3[iplot],fontsize=14,transform=a1.transAxes)
    if iplot == 0:
        plt.title(text2[0],fontsize=16)
cax=plt.axes((0.87,0.3,0.02,0.4))
cbar1=plt.colorbar(c1,cax=cax,ticks=np.arange(-1.5,2,0.5))
cbar1.ax.set_ylabel('OHC anomaly (standardized)')

plt.tight_layout()
# plt.savefig('figures/OHCinput_lat='+str(lat1)+'lon='+str(lon1)+'.png',dpi=300)
plt.show()

#%%

Y_output = np.nanmean(Y_test_full[(sigmaboo & muboo & trueboo),:,:],axis=0)
Y_output,_ = nndata.addcyclicpoint(Y_output,lonsel)
lon2 = np.arange(0,365,5)
a2=plt.subplot(1,1,1,projection=projection)
c2=a2.contourf(lon2,latsel,Y_output,contours2,cmap=cmapSST,transform=transform,extend='both')
a2.coastlines(color='gray')
a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', facecolor='gray'))
a2.scatter(lon1,lat1,color='xkcd:neon green',transform=transform,marker='o')
plt.title('f. SST at output')
cbar2=plt.colorbar(c2,shrink=0.6,ticks=np.arange(-0.6,0.9,0.3))
cbar2.ax.set_ylabel('SST anomaly (standardized)')
plt.tight_layout()
# plt.savefig('figures/SSToutput_lat='+str(lat1)+'lon='+str(lon1)+'.png',dpi=300)
plt.show()

#%% and the histograms of AMV/IPO for these predictions

AMVbest20 = AMVindex[muboo & sigmaboo & trueboo]
IPObest20 = IPOindex[muboo & sigmaboo & trueboo]

Ntotal = AMVindex.shape[0]
Nbest = AMVbest20.shape[0]

totalweights = np.zeros(Ntotal)+(1/Ntotal)
bestweights = np.zeros(Nbest)+(1/Nbest)

binwidth = 0.4
bins = np.arange(-3.6,3.6+binwidth,binwidth)

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.hist(AMVindex,bins,weights=totalweights,color='xkcd:maroon',label='all testing')
plt.hist(AMVbest20,bins,weights=bestweights,color='xkcd:neon pink',alpha=0.6,label='20% most confident')
plt.legend(framealpha=0.2)
plt.xticks(np.arange(-3,4))
plt.yticks(np.arange(0,0.7,0.1))
plt.xlim(-3.6,3.6)
plt.ylabel('Frequency')
plt.title('d. AMV index')
plt.xlabel('index value')

plt.subplot(1,2,2)
plt.hist(IPOindex,bins,weights=totalweights,color='xkcd:dark green',label='all testing')
plt.hist(IPObest20,bins,weights=bestweights,color='xkcd:neon green',alpha=0.6,label='20% most confident')
plt.legend(framealpha=0.2)
plt.xticks(np.arange(-3,4))
plt.yticks(np.arange(0,0.3,0.05))
plt.xlim(-3.6,3.6)
plt.title('e. IPO index')
plt.xlabel('index value')

plt.tight_layout()

# plt.savefig('figures/IndexHists_lat='+str(lat1)+'lon='+str(lon1)+'.png',dpi=300)
plt.show()

#%% bootstrap distribution

nboots = 10000

AMVboots = np.empty(nboots)
IPOboots = np.empty(nboots)

for iboot in range(nboots):
    AMVboots[iboot] = np.mean(np.random.choice(AMVindex,Nbest,replace=False))
    IPOboots[iboot] = np.mean(np.random.choice(IPOindex,Nbest,replace=False))    
    
AMVpers = np.percentile(AMVboots,[1,99])
IPOpers = np.percentile(IPOboots,[1,99])

AMVbestmean = np.mean(AMVbest20)
IPObestmean = np.mean(IPObest20)

AMVsig = (AMVbestmean<AMVpers[0]) | (AMVbestmean>AMVpers[1])
IPOsig = (IPObestmean<IPOpers[0]) | (IPObestmean>IPOpers[1])

if AMVsig:
    print(AMVbestmean)
    print(AMVpers)
    print('AMV mean significant')
if IPOsig:
    print(IPObestmean)
    print(IPOpers)
    print('IPO mean significant')

#%% and give us some numbers

numconf = AMVindex[sigmaboo].shape[0]
numconfandpos = AMVindex[muboo & sigmaboo].shape[0]
numconfandposandcorr = AMVindex[muboo & sigmaboo & trueboo].shape[0]
numconfandnegandcorr = AMVindex[~muboo & sigmaboo & ~trueboo].shape[0]

print(numconf)
print(numconfandpos)
print(numconfandposandcorr)

