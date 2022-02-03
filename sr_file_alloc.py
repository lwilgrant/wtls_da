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
    # dictionaries containing keyed ties to files for models, pichunks and obs
    # based on grid and obs


#%%============================================================================
# import
# =============================================================================


import os
import xarray as xr
import numpy as np
from funcs import *


#%%============================================================================

# file organization
def file_subroutine(mapDIR,
                    modDIR,
                    piDIR,
                    allpiDIR,
                    obsDIR,
                    grid,
                    pi,
                    obs_types,
                    lulcc,
                    y1,
                    y2,
                    t_ext,
                    models,
                    exps,
                    var):
    
    map_files = {}
    grid_files = {}
    fp_files = {}
    pi_files = {}
    obs_files = {}
    
    nx = {}
    # nx['CanESM5'] = np.asarray(([7,7]))
    # nx['CNRM-ESM2-1'] = np.asarray(([3,3]))
    # nx['IPSL-CM6A-LR'] = np.asarray(([4,4]))
    # nx['UKESM1-0-LL'] = np.asarray(([4,4]))
    # nx['mmm'] = np.asarray(([4,4]))    

    #==============================================================================
            
    # map files
    os.chdir(mapDIR)
    
    if grid == 'obs':
        
        for obs in obs_types:
            
            map_files[obs] = {}
            grid_files[obs] = obs + '_gridarea.nc' # won't need this since map files at obs res are already in area
            
            for lu in lulcc:
            
                for file in [file for file in sorted(os.listdir(mapDIR))
                            if obs in file\
                            and lu in file\
                            and str(y1) in file\
                            and str(y2) in file\
                            and 'absolute_change' in file]:
                    
                    map_files[obs][lu] = file   
    
    elif grid == 'model':
    
        for mod in models:
        
            map_files[mod] = {}
            grid_files[mod] = mod+'_gridarea.nc'
            
            for lu in lulcc:
                
                for file in [file for file in sorted(os.listdir(mapDIR))
                            if mod in file\
                            and lu in file\
                            and str(y1) in file\
                            and str(y2) in file\
                            and 'absolute_change' in file]:

                        map_files[mod][lu] = file

    #==============================================================================
    
    # model files
    os.chdir(modDIR)
    
    for mod in models:
        
        fp_files[mod] = {}
        nx[mod] = {}
        
        for exp in exps:
            
            e_i = 0
            
            if grid == 'obs':
                
                fp_files[mod][exp] = {}
                
                for obs in obs_types:
                
                    fp_files[mod][exp][obs] = []
                
                    for file in [file for file in sorted(os.listdir(modDIR))\
                                if var in file\
                                and mod in file\
                                and exp in file\
                                and t_ext in file\
                                and obs in file\
                                and 'unmasked' in file\
                                and not 'ensmean' in file]:
                        
                        fp_files[mod][exp][obs].append(file)
                    
            elif grid == 'model':
                
                fp_files[mod][exp] = []
                
                for file in [file for file in sorted(os.listdir(modDIR))\
                            if var in file\
                            and mod in file\
                            and exp in file\
                            and t_ext in file\
                            and not obs_types[0] in file\
                            and not obs_types[1] in file\
                            and 'unmasked' in file\
                            and not 'ensmean' in file]:
                        
                    fp_files[mod][exp].append(file)  
                    e_i += 1
                
                if exp == 'historical' or exp == 'hist-noLu':    
                    nx[mod][exp] = e_i
        
        nx[mod] = np.array([[nx[mod]['historical'],nx[mod]['hist-noLu']]])
                    
    #==============================================================================
    
    # pi files
    os.chdir(piDIR)
    
    if pi == 'model':
        
        for mod in models:
            
            if grid == 'obs':
                
                pi_files[mod] = {}
            
                for obs in obs_types:
                    
                    pi_files[mod][obs] = []
            
                    for file in [file for file in sorted(os.listdir(piDIR))\
                                if var in file\
                                and mod in file\
                                and t_ext in file\
                                and obs in file\
                                and 'unmasked' in file]:
                        
                        pi_files[mod][obs].append(file)
                    
            if grid == 'model':
                
                pi_files[mod] = []
            
                for file in [file for file in sorted(os.listdir(piDIR))\
                            if var in file\
                            and mod in file\
                            and t_ext in file\
                            and not obs_types[0] in file\
                            and not obs_types[1] in file\
                            and 'unmasked' in file]:
                    
                    pi_files[mod].append(file)
    
    # use all available pi chunks
    elif pi == 'allpi':
            
        # this option is not available yet
        if grid == 'obs':
            
            pi_files[obs] = {}
        
            for obs in obs_types:
                
                pi_files[obs] = []
        
                for file in [file for file in sorted(os.listdir(piDIR))\
                            if var in file\
                            and t_ext in file\
                            and obs in file\
                            and 'unmasked' in file]:
                    
                    pi_files[obs].append(file)
                        
        if grid == 'model':
            
            pi_files = []
        
            for file in [file for file in sorted(os.listdir(allpiDIR))\
                        if var in file\
                        and t_ext in file\
                        and not obs_types[0] in file\
                        and not obs_types[1] in file\
                        and 'unmasked' in file]:
                
                pi_files.append(file)
                
    #==============================================================================
    
    # obs files
    os.chdir(obsDIR)            
    
    if grid == 'obs':
        
        for obs in obs_types:
        
            for file in [file for file in sorted(os.listdir(obsDIR))\
                        if var in file\
                        and 'obs' in file\
                        and obs in file\
                        and not '-res' in file\
                        and 'unmasked' in file\
                        and t_ext in file]:
                
                obs_files[obs] = file
            
    elif grid == 'model':
        
        obs_files = {}

        for obs in obs_types:
        
            obs_files[obs] = {}
        
            for mod in models:
                
                for file in [file for file in sorted(os.listdir(obsDIR))\
                            if var in file\
                            and 'obs' in file\
                            and obs in file\
                            and mod + '-res' in file\
                            and 'unmasked' in file\
                            and t_ext in file]:
                    
                    obs_files[obs][mod] = file

    return     map_files,grid_files,fp_files,pi_files,obs_files,nx