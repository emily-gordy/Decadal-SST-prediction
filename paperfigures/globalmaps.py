#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:32:20 2022

@author: emgordy
"""

# make nice figure of 

# top row MAE all predictions
# middle row MAE on 20% most confident predictions
# bottom row MAE difference from persistence

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cmasher as cmr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import sys
sys.path.append('functions/')
import nndata
import matplotlib.colors as colors
import pickle

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

runin = 60
hiddens = [60,4]
lr = 1e-4
drate = 0.8

lys = [[1,5],[3,7]]

metricstrs = 'ExperimentMetricsAll_ly*_hiddens'+str(hiddens[0])+str(hiddens[1])+'_lr'+str(lr)+'_drate'+str(drate)+'.nc'
metricstrs = glob.glob(metricstrs)
metricstrs = sorted(metricstrs)

#%% load in metric ds

metricds1 = xr.open_dataset(metricstrs[0])
testloss = np.asarray(metricds1.testloss)
arrdims = np.shape(testloss)

bestMAEall = np.empty((arrdims[0],arrdims[1],2))+np.nan
bestMAE20 = np.empty((arrdims[0],arrdims[1],2))+np.nan

persistenceMAE = np.empty((arrdims[0],arrdims[1],2))+np.nan

# loop through the two lead year ranges
for ily in range(2):
    
    metricstr = metricstrs[ily]
    print(metricstr)
    ly1 = lys[ily][0]
    ly2 = lys[ily][1]
    
    metricsds = xr.open_dataset(metricstr)
    
    testloss = np.asarray(metricsds.testloss)
    testpercentiles = np.asarray(metricsds.testpercentiles)
    test_muvar = np.asarray(metricsds.test_muvar)
    testMAEscaled = np.asarray(metricsds.testMAEscaled)
    sCC = np.asarray(metricsds.testsCC)
    pvals = np.asarray(metricsds.testpvals)
    classacc = np.asarray(metricsds.testaccuracy)
    
    MAE = np.asarray(metricsds.testMAE)
    
    lat = np.asarray(metricsds.lat)
    lon = np.asarray(metricsds.lon)
    seeds = metricsds.seed
    percentiles = metricsds.percentile
    
    
    #calculate variance in Y_test
    Y_train, Y_val, Y_test, _, _ = nndata.makedata_heatx3_SSTonly_ALL(ly1,ly2)
    Y_testvar = np.expand_dims(np.std(Y_test,axis=0),axis=2)
    
    #get rid of land + flatliners
    test_muvar = test_muvar/Y_testvar
    
    nomodel = np.isnan(testloss)
    flatliner = test_muvar<0.1
    
    testloss[nomodel | flatliner] = np.nan
    testpercentiles[nomodel | flatliner,:] = np.nan
    test_muvar[nomodel | flatliner] = np.nan
    MAE[nomodel | flatliner] = np.nan
    sCC[nomodel | flatliner,:] = np.nan
    pvals[nomodel | flatliner,:] = np.nan
    classacc[nomodel | flatliner,:] = np.nan
    testMAEscaled[nomodel | flatliner,:] = np.nan
    
    with open('bestind_val_ly'+str(ly1)+'-'+str(ly2)+'_array.pkl', 'rb') as f: # load in best model seed numbers
        bestind = pickle.load(f)
    
    for ilat,latloop in enumerate(lat):
        for ilon,lonloop in enumerate(lon):
            if ~np.isnan(bestind[ilat,ilon]):
                bestMAEall[ilat,ilon,ily] = testpercentiles[ilat,ilon,int(bestind[ilat,ilon]),0]
                bestMAE20[ilat,ilon,ily] = testpercentiles[ilat,ilon,int(bestind[ilat,ilon]),8]
                
    # persistence model
    per_train, per_val, per_test, latsel, lonsel = nndata.sstpersistence_ALL(ly1,ly2)
    persistenceMAE[:,:,ily] = np.nanmean(np.abs(per_test-Y_test),axis=0)

errdiff = bestMAEall-persistenceMAE    

#%% plotting

bestMAEall[np.isnan(bestMAEall)] = 36
bestMAE20[np.isnan(bestMAE20)] = 36
errdiff[np.isnan(errdiff)] = 0

bounds1 = np.arange(0.2,1.05,0.05)
bounds2 = np.arange(-0.7,0.8,0.1)

cmapMAE = cmr.torch_r
cmapdiffMAE = cmr.redshift_r

norm1 = colors.BoundaryNorm(boundaries=bounds1, ncolors=cmapMAE.N)
norm2 = colors.BoundaryNorm(boundaries=bounds2, ncolors=cmapdiffMAE.N)

projection=ccrs.EqualEarth(central_longitude=255)
transform=ccrs.PlateCarree()

plt.figure(figsize=(12,9))

a1=plt.subplot(3,2,1,projection=projection)
c1=a1.pcolormesh(lon,lat,bestMAEall[:,:,0],vmin=bounds1[0],vmax=bounds1[-1],cmap=cmapMAE,norm=norm1,transform=transform)
a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', facecolor='gray'))
plt.title('lead year 1-5')
plt.text(-0.07,0.25,'all predictions',transform=a1.transAxes,rotation='vertical')
plt.text(0.02,0.97,'a',transform=a1.transAxes)

a2=plt.subplot(3,2,2,projection=projection)
a2.pcolormesh(lon,lat,bestMAEall[:,:,1],vmin=bounds1[0],vmax=bounds1[-1],cmap=cmapMAE,norm=norm1,transform=transform)
a2.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', facecolor='gray'))
plt.title('lead year 3-7')
plt.text(0.02,0.97,'b',transform=a2.transAxes)

a3=plt.subplot(3,2,3,projection=projection)
c3=a3.pcolormesh(lon,lat,bestMAE20[:,:,0],vmin=bounds1[0],vmax=bounds1[-1],cmap=cmapMAE,norm=norm1,transform=transform)
a3.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', facecolor='gray'))
plt.text(-0.07,0.15,'20% most confident',transform=a3.transAxes,rotation='vertical')
plt.text(0.02,0.97,'c',transform=a3.transAxes)

a4=plt.subplot(3,2,4,projection=projection)
a4.pcolormesh(lon,lat,bestMAE20[:,:,1],vmin=bounds1[0],vmax=bounds1[-1],cmap=cmapMAE,norm=norm1,transform=transform)
a4.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', facecolor='gray'))
plt.text(0.02,0.97,'d',transform=a4.transAxes)

cax1 = plt.axes((0.91,0.46,0.015,0.35))
cbar1=plt.colorbar(c1,cax=cax1,ticks=np.arange(0.2,1.2,0.2))
cbar1.ax.set_ylabel('MAE')

a5=plt.subplot(3,2,5,projection=projection)
c5=a5.pcolormesh(lon,lat,errdiff[:,:,0],vmin=bounds2[0],vmax=bounds2[-1],cmap=cmapdiffMAE,norm=norm2,transform=transform)
a5.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', facecolor='gray'))
plt.text(-0.07,0.2,'MAE difference',transform=a5.transAxes,rotation='vertical')
plt.text(0.02,0.97,'e',transform=a5.transAxes)

a6=plt.subplot(3,2,6,projection=projection)
a6.pcolormesh(lon,lat,errdiff[:,:,1],vmin=bounds2[0],vmax=bounds2[-1],cmap=cmapdiffMAE,norm=norm2,transform=transform)
a6.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', facecolor='gray'))
plt.text(0.02,0.97,'f',transform=a6.transAxes)

cax2 = plt.axes((0.91,0.14,0.015,0.2))
cbar2 = plt.colorbar(c5,cax=cax2,ticks=np.arange(-0.6,0.9,0.3),extend='both')
cbar2.ax.set_ylabel(r'$\Delta$MAE ')

plt.tight_layout()

plt.savefig('figures/nicemaps.png',dpi=300)

plt.show()



