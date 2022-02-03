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
    # model ensemble means for OF (mod_ens)
    # model t-series as ar6-weighted matrices of rows for tsteps and columns for ar6 regions (mod_ts_ens)
        # axis 0 for realisations (realisations x tstep_rows x ar6_columns)
        # these t-series are for box plot data; not used for OF 
        
# To check:
    # check that maps for all cases have 1's for desired locations and 0 otherwise:
        # absoute change, area change and all pixels versions
        # make sure it is working for different observation types


#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
from copy import deepcopy
from da_funcs import *


#%%============================================================================

# mod data
def ensemble_subroutine(modDIR,
                        maps,
                        models,
                        exps,
                        var,
                        lu_techn,
                        measure,
                        lulcc_type,
                        y1,
                        grid,
                        agg,
                        weight,
                        freq,
                        obs_types,
                        continents,
                        ns,
                        fp_files,
                        ar6_regs,
                        ar6_wts,
                        cnt_regs,
                        cnt_wts):

    os.chdir(modDIR)
    mod_data = {}
    mod_ens = {}
    mod_ts_ens = {}
    mod_ts_ens['mmm'] = {}
    omega_samples = {}
        
    # observation resolution data
    if grid == 'obs':
        
        for exp in exps:
            
            mod_ts_ens['mmm'][exp] = {}
            
            for obs in obs_types:
                
                mod_ts_ens['mmm'][exp][obs] = []
        
        i = 0
        
        for mod in models:
            
            mod_data[mod] = {}
            mod_ens[mod] = {}
            mod_ts_ens[mod] = {}
            
            for exp in exps:
                
                mod_data[mod][exp] = {}    
                mod_ens[mod][exp] = {}
                mod_ts_ens[mod][exp] = {}
                
                for obs in obs_types:
                    
                    mod_data[mod][exp][obs] = []    
                    mod_ts_ens[mod][exp][obs] = []
                    
                    if measure == 'area_change':
                        
                        lc = maps[obs][lulcc_type]
                        
                    elif measure == 'all_pixels':
                        
                        lc = maps[obs]
                
                    for file in fp_files[mod][exp][obs]:
                    
                        da = nc_read(file,
                                     y1,
                                     var,
                                     obs=obs,
                                     freq=freq)
                        da = da.where(lc == 1)
                        if i == 0:
                            nt = len(da.time.values)
                        i += 1
                        mod_ar6 = ar6_weighted_mean(continents,
                                                da,
                                                ar6_regs[obs],
                                                nt,
                                                ns)
                        mod_ar6 = del_rows(mod_ar6)
                        input_mod_ar6 = deepcopy(mod_ar6)
                        mod_ar6_center = temp_center(ns,
                                                     input_mod_ar6)
                        mod_ts_ens[mod][exp][obs].append(mod_ar6_center)
                        mod_ts_ens['mmm'][exp][obs].append(mod_ar6_center)
                        mod_data[mod][exp][obs].append(da)
                    
                    mod_ts_ens[mod][exp][obs] = np.stack(mod_ts_ens[mod][exp][obs],axis=0)
                    mod_ens[mod][exp][obs] = da_ensembler(mod_data[mod][exp][obs])
            
            if lu_techn == 'mean': 
            
                for obs in obs_types:
                    
                    mod_ens[mod]['lu'][obs] = mod_ens[mod]['historical'][obs] - mod_ens[mod]['hist-noLu'][obs]
                
        for exp in exps:
            
            for obs in obs_types:
            
                mod_ts_ens['mmm'][exp][obs] = np.stack(mod_ts_ens['mmm'][exp][obs],axis=0)
    
    # model resolution data
    elif grid == 'model':
        
        for exp in exps:
            
            mod_ts_ens['mmm'][exp] = []
        
        i = 0
        
        for mod in models:
            
            mod_data[mod] = {}
            mod_ens[mod] = {}
            mod_ts_ens[mod] = {}
            omega_samples[mod] = {}
            
            if measure == 'area_change':
                
                lc = maps[mod][lulcc_type]
                
            elif measure == 'all_pixels':
                
                lc = maps[mod]
            
            for exp in exps:
                
                mod_data[mod][exp] = []    
                mod_ts_ens[mod][exp] = []
                omega_samples[mod][exp] = []
                
                for file in fp_files[mod][exp]:
                
                    da = nc_read(file,
                                 y1,
                                 var,
                                 freq=freq)
                    da = da.where(lc == 1)
                    if i == 0:
                        try:
                            nt = len(da.time.values)
                        except:
                            nt = 1
                    i += 1
                    
                    if agg == 'ar6':
                        
                        mod_ar6 = ar6_weighted_mean(continents,
                                                    da,
                                                    ar6_regs[mod],
                                                    nt,
                                                    ns,
                                                    weight,
                                                    ar6_wts[mod])
                        mod_ar6 = del_rows(mod_ar6)
                        input_mod_ar6 = deepcopy(mod_ar6)
                        mod_ar6_center = temp_center(
                            nt,
                            ns,
                            input_mod_ar6)
                        mod_ts_ens[mod][exp].append(mod_ar6_center)
                        omega_samples[mod][exp].append(mod_ar6_center.flatten())
                        mod_ts_ens['mmm'][exp].append(mod_ar6_center)
                        
                    elif agg == 'continental':
                        
                        mod_cnt = cnt_weighted_mean(continents,
                                                    da,
                                                    cnt_regs[mod],
                                                    nt,
                                                    ns,
                                                    weight,
                                                    cnt_wts[mod])
                        mod_cnt = del_rows(mod_cnt)
                        input_mod_cnt = deepcopy(mod_cnt)
                        mod_cnt_center = temp_center(
                            nt,
                            ns,
                            input_mod_cnt)
                        mod_ts_ens[mod][exp].append(mod_cnt_center)
                        omega_samples[mod][exp].append(mod_cnt_center.flatten())
                        mod_ts_ens['mmm'][exp].append(mod_cnt_center)      
                                          
                    mod_data[mod][exp].append(da)
                
                mod_ts_ens[mod][exp] = np.stack(mod_ts_ens[mod][exp],axis=0)
                omega_samples[mod][exp] = np.stack(omega_samples[mod][exp],axis=0)
                mod_ens[mod][exp] = da_ensembler(mod_data[mod][exp])
            
            if lu_techn == 'mean':
            
                mod_ens[mod]['lu'] = mod_ens[mod]['historical'] - mod_ens[mod]['hist-noLu']
                
        for exp in exps:
            
            mod_ts_ens['mmm'][exp] = np.stack(mod_ts_ens['mmm'][exp],axis=0)
        
    return mod_ens,omega_samples,nt
# %%
