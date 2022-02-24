#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:06:33 2021

@author: emgordy
"""
# functions for evaluating various metrics

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
import tensorflow_probability as tfp

def logp_GAUSS(y_pred,Y_test): # -log likelihood of y_true given conditional distribution predicted by ANN

    mu = y_pred[:,0]
    std = np.exp(y_pred[:,1])
    logp = np.mean(-norm.logpdf(Y_test[:,0],loc=mu,scale=std))
    
    return logp

def MAE(y_pred,Y_test): # mean absolute error, |truth-mu|
    
    mu = y_pred[:,0] 
    y_true = Y_test[:,0]
    
    AE = np.abs(mu-y_true)
    MAE = np.mean(AE)
    
    return MAE

def halfNhalf_GAUSS(y_pred,Y_test): # guesses too high vs guesses too low, should be approximately 50/50
    
    mu = y_pred[:,0] 
    y_true = Y_test[:,0]
    
    hilo = mu-y_true
    numhi = hilo[hilo>0].shape[0]
    numtotal = hilo.shape[0]
    
    prophi = numhi/numtotal
    
    return prophi


def predspread_GAUSS(y_pred,Y_test): # guesses within 1 sigma vs guesses outside 1 sigma, should be 68/32
    
    mu = y_pred[:,0]
    std = np.exp(y_pred[:,1])
    Y_true = Y_test[:,0]
    
    upper = mu+std
    lower = mu-std

    hitboo = (Y_true>lower) & (Y_true<upper)
    numhits = std[hitboo].shape[0]
    numtotal = std.shape[0]
    
    prophits = numhits/numtotal
    
    return prophits

def MAEpercentiles_GAUSS(y_pred,Y_test): # MAE as function of network confidence, each step thresholds higher confidence (lower sigma)
    mu = y_pred[:,0]
    sigma = np.exp(y_pred[:,1])
    Y_true = Y_test[:,0]
    percentiles = np.arange(0,100,10)
    numbins = percentiles.shape[0]
    
    inv_sigma = 1/sigma
    MAEbinned = np.empty(numbins)
    MAEbinned[:] = np.nan
    
    sigmacutoff = np.empty(numbins)
    sigmacutoff[:] = np.nan

    for iper, per in enumerate(percentiles):
        thres = np.percentile(inv_sigma,per)
        mu_choose = mu[inv_sigma>thres]
        true_choose = Y_true[inv_sigma>thres]
        MAEloop = np.abs(mu_choose-true_choose)
        MAEbinned[iper] = np.mean(MAEloop)
        sigmacutoff[iper] = 1/thres
    
    return MAEbinned, percentiles, sigmacutoff

def ClassificationAccuracy(y_pred,Y_test): # classification accuracy: how often is the sign correct
    
    mu = y_pred[:,0]
    sigma = np.exp(y_pred[:,1])
        
    y_true = Y_test[:,0]
    y_true[y_true>0] = 1
    y_true[y_true<=0] = 0
    mu_class = np.copy(mu)
    mu_class[mu_class>0] = 1
    mu_class[mu_class<=0] = 0
    
    mu_correct = (mu_class == y_true)    
    percentilebins = np.arange(0,100,10)
    confidence = 1/sigma
    accout = np.empty(10)
    # x=[]
    for iper,per in enumerate(percentilebins):
        thres = np.percentile(confidence,per)    
        thresboo = confidence>=thres
        mu_corrconf = mu_correct[thresboo]
        # print(mu_corrconf.shape[0])
        # x.append(mu_corrconf.shape[0])
        acc = mu_corrconf[mu_corrconf].shape[0]
        accacc = acc/mu_corrconf.shape[0]
        accout[iper] = accacc
    
    return accout
            
def Pearsonpercentile(y_pred,Y_test): # pearson correlation coefficient as a function of network confidence
    mu = y_pred[:,0]
    sigma = np.exp(y_pred[:,1])
    Y_true = Y_test[:,0]
    
    percentiles = np.arange(0,100,10)
    numbins = percentiles.shape[0]
    
    inv_sigma = 1/sigma
    pearsonbinned = np.empty(numbins)+np.nan
    pval = np.empty(numbins)+np.nan    
    for iper, per in enumerate(percentiles):
        thres = np.percentile(inv_sigma,per)
        mu_choose = mu[inv_sigma>thres]
        if mu_choose.shape[0]>0:
            true_choose = Y_true[inv_sigma>thres]
            pearsonbinned[iper],pval[iper] = pearsonr(mu_choose,true_choose)

    return pearsonbinned,pval 


def Spearmanpercentile(y_pred,Y_test): # spearman rank coefficient as a function of network confidence
    mu = y_pred[:,0]
    sigma = np.exp(y_pred[:,1])
    Y_true = Y_test[:,0]
    
    percentiles = np.arange(0,100,10)
    numbins = percentiles.shape[0]
    
    inv_sigma = 1/sigma
    spearmanbinned = np.empty(numbins)+np.nan
    pval = np.empty(numbins)+np.nan    
    for iper, per in enumerate(percentiles):
        thres = np.percentile(inv_sigma,per)
        mu_choose = mu[inv_sigma>thres]
        if mu_choose.shape[0]>0:
            true_choose = Y_true[inv_sigma>thres]
            spearmanbinned[iper],pval[iper] = spearmanr(mu_choose,true_choose)
        
    return spearmanbinned,pval 

