#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# This subroutine script generates:
    # optimal fingerprinting results 


#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
import numpy as np
from funcs import *


#%%============================================================================

# mod data
def of_subroutine(models,
                  nx,
                  analysis,
                  exp_list,
                  obs_types,
                  pi,
                  obs_data,
                  obs_data_continental,
                  fp,
                  fp_continental,
                  omega_samples,
                  ctl_data,
                  ctl_data_continental,
                  bs_reps,
                  ns,
                  nt,
                  reg,
                  cons_test,
                  formule_ic_tls,
                  trunc,
                  ci_bnds,
                  continents):

    var_sfs = {}
    bhi = {}
    b = {}
    blow = {}
    pval = {}
    var_fin = {}
    var_ctlruns = {}
    U = {}
    yc = {}
    Z1c = {}
    Z2c = {}
    Xc = {}
    Cf1 = {}
    Ft = {}
    beta_hat = {}

    models.append('mmm')   
    
    for obs in obs_types:
        
        var_sfs[obs] = {}
        bhi[obs] = {}
        b[obs] = {}
        blow[obs] = {}
        pval[obs] = {}
        var_fin[obs] = {}
        var_ctlruns[obs] = {}
        U[obs] = {}
        yc[obs] = {}
        Z1c[obs] = {}
        Z2c[obs] = {}
        Xc[obs] = {}
        Cf1[obs] = {}
        Ft[obs] = {}
        beta_hat[obs] = {}
            
        for mod in models:

            #==============================================================================
            
            # global analysis
            if analysis == "global":
                
                bhi[obs][mod] = {}
                b[obs][mod] = {}
                blow[obs][mod] = {}
                pval[obs][mod] = {}
                var_fin[obs][mod] = {}
                
                for exp in exp_list:
                    
                    bhi[obs][mod][exp] = []
                    b[obs][mod][exp] = []
                    blow[obs][mod][exp] = []
                    pval[obs][mod][exp] = []
                
                y = obs_data[obs][mod]
                X = fp[mod]
                if pi == 'model':
                    ctl = ctl_data[mod]
                elif pi == 'allpi':
                    ctl = ctl_data
                    
                nbts = nt
                n_spa = ns
                n_st = n_spa * nbts               
                # Spatio-temporal dimension after reduction
                if nbts > 1:
                    n_red = n_st - n_spa
                    U = projfullrank(nbts, n_spa)
                elif nbts == 1: # case where I regress with collapsed time dimension and each point is a signal
                    n_red = n_st - nbts # therefore treating ns as nt
                    U = projfullrank(n_spa, nbts)                                   
                
                cov_omega = {}
                booties = {}
                
                for exp in exp_list:
                    
                    if exp == "historical":
                        runs = nx[mod][0]
                    elif exp == "hist-noLu":
                        runs = nx[mod][1]
                    
                    omega = omega_samples[mod][exp]
                    omega = np.transpose(np.matrix(mod))
                    omega = np.dot(U, omega)
                    booties[exp] = []
                
                    for i in np.arange(1000):
                        booty = []
                        for r in np.arange(len(runs)):
                            index = np.random.randint(0,len(omega.shape[1]))
                            booty.append(omega[:,index])
                        booties[exp].append(np.mean(booty))
                    
                    omega = np.stack(booties[exp],axis=0)
                    omega = np.transpose(np.matrix(omega))
                    cov_omega[exp] = np.dot(omega,omega.T) / omega.shape[1]
                                    
                
                
                nb_runs_x= nx[mod]
                
                if bs_reps == 0: # for no bs, run ROF once
                
                    bs_reps += 1
                
                for i in np.arange(0,bs_reps):
                    
                    # shuffle rows of ctl
                    ctl = np.take(ctl,
                                    np.random.permutation(ctl.shape[0]),
                                    axis=0)
                    
                    # run detection and attribution
                    var_sfs[obs][mod],\
                    var_ctlruns[obs][mod],\
                    proj,\
                    U[obs][mod],\
                    yc[obs][mod],\
                    Z1c[obs][mod],\
                    Z2c[obs][mod],\
                    Xc[obs][mod],\
                    Cf1[obs][mod],\
                    Ft[obs][mod],\
                    beta_hat[obs][mod] = da_run(
                        nx,
                        bs_reps,
                        y,
                        X,
                        ctl,
                        omega,
                        nb_runs_x,
                        ns,
                        nt,
                        reg,
                        cons_test,
                        formule_ic_tls,
                        trunc,
                        ci_bnds,
                        bhi,
                        b,
                        blow,
                        pval,
                        var_sfs)
                    
                    # [yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat]
                    for i,exp in enumerate(exp_list):
                        
                        bhi[obs][mod][exp].append(var_sfs[obs][mod][2,i])
                        b[obs][mod][exp].append(var_sfs[obs][mod][1,i])
                        blow[obs][mod][exp].append(var_sfs[obs][mod][0,i])
                        pval[obs][mod][exp].append(var_sfs[obs][mod][3,i])
                
                for exp in exp_list:
                    
                    bhi_med = np.median(bhi[obs][mod][exp])
                    b_med = np.median(b[obs][mod][exp])
                    blow_med = np.median(blow[obs][mod][exp])
                    pval_med = np.median(pval[obs][mod][exp])
                    var_fin[obs][mod][exp] = [bhi_med,
                                                b_med,
                                                blow_med,
                                                pval_med]
            
            #==============================================================================
                
            # continental analysis
            elif analysis == 'continental':
                
                bhi[obs][mod] = {}
                b[obs][mod] = {}
                blow[obs][mod] = {}
                pval[obs][mod] = {}
                var_fin[obs][mod] = {}
                var_sfs[obs][mod] = {}
                var_ctlruns[obs][mod] = {}
                U[obs][mod] = {}
                yc[obs][mod] = {}
                Z1c[obs][mod] = {}
                Z2c[obs][mod] = {}
                Xc[obs][mod] = {}
                Cf1[obs][mod] = {}
                Ft[obs][mod] = {}
                beta_hat[obs][mod] = {}
                
                for exp in exp_list:
                    
                    bhi[obs][mod][exp] = {}
                    b[obs][mod][exp] = {}
                    blow[obs][mod][exp] = {}
                    pval[obs][mod][exp] = {}
                    var_fin[obs][mod][exp] = {}
                    
                    for c in continents.keys():
                        
                        bhi[obs][mod][exp][c] = []
                        b[obs][mod][exp][c] = []
                        blow[obs][mod][exp][c] = []
                        pval[obs][mod][exp][c] = []
                    
                for c in continents.keys():
                
                    y = obs_data_continental[obs][mod][c]
                    X = fp_continental[mod][c]
                    if pi == 'model':
                        ctl = ctl_data_continental[mod][c]
                    elif pi == 'allpi':
                        ctl = ctl_data_continental[c]
                    nb_runs_x= nx[mod]
                    ns = len(continents[c])
                
                    if bs_reps == 0: # for no bs, run ROF once
                    
                        bs_reps += 1
                    
                    for i in np.arange(0,bs_reps):
                        
                        # shuffle rows of ctl
                        ctl = np.take(ctl,
                                        np.random.permutation(ctl.shape[0]),
                                        axis=0)
                        
                        # run detection and attribution
                        var_sfs[obs][mod][c],\
                        var_ctlruns[obs][mod][c],\
                        proj,\
                        U[obs][mod][c],\
                        yc[obs][mod][c],\
                        Z1c[obs][mod][c],\
                        Z2c[obs][mod][c],\
                        Xc[obs][mod][c],\
                        Cf1[obs][mod][c],\
                        Ft[obs][mod][c],\
                        beta_hat[obs][mod][c] = da_run(y,
                                                        X,
                                                        ctl,
                                                        nb_runs_x,
                                                        ns,
                                                        nt,
                                                        reg,
                                                        cons_test,
                                                        formule_ic_tls,
                                                        trunc,
                                                        ci_bnds)
                        
                        # [yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat]
                        for i,exp in enumerate(exp_list):
                            
                            bhi[obs][mod][exp][c].append(var_sfs[obs][mod][c][2,i])
                            b[obs][mod][exp][c].append(var_sfs[obs][mod][c][1,i])
                            blow[obs][mod][exp][c].append(var_sfs[obs][mod][c][0,i])
                            pval[obs][mod][exp][c].append(var_sfs[obs][mod][c][3,i])
                    
                for exp in exp_list:
                    
                    for c in continents.keys():
                        
                        bhi_med = np.median(bhi[obs][mod][exp][c])
                        b_med = np.median(b[obs][mod][exp][c])
                        blow_med = np.median(blow[obs][mod][exp][c])
                        pval_med = np.median(pval[obs][mod][exp][c])
                        var_fin[obs][mod][exp][c] = [bhi_med,
                                                        b_med,
                                                        blow_med,
                                                        pval_med]
                            
    return var_sfs,var_ctlruns,proj,U,yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat,var_fin,models

# %%
