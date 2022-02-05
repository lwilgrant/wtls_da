#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 2018
@author: alexander.winkler@mpimet.mpg.de
Title: Optimal Fingerprinting after Ribes et al., 2009
"""

# =============================================================================
# import
# =============================================================================


import numpy as np
import scipy.linalg as spla
import scipy.stats as sps
import xarray as xr
import pandas as pd
import os
import regionmask as rm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import pickle as pk
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask as rm
from random import shuffle
from matplotlib.lines import Line2D
from copy import deepcopy
import pickle as pk
import geopandas as gp
import mapclassify as mc


# =============================================================================
# functions
# =============================================================================

#%%============================================================================

def nc_read(file,
            y1,
            var,
            obs=False,
            freq=False):
    
    """ Read in netcdfs based on variable and set time.
    
    Parameters
    ----------
    file : files in data directory
    
    Returns
    ------- 
    Xarray data array
    """
    
    ds = xr.open_dataset(file,decode_times=False)
    da = ds[var].squeeze()
    units, reference_date = da.time.attrs['units'].split('since')
    reference_date = reference_date.replace(reference_date[1:5],str(y1))[1:]
    new_date = pd.date_range(start=reference_date, periods=da.sizes['time'], freq='YS')
    da['time'] = new_date
    
    if 'height' in da.coords:
        da = da.drop('height')
    if obs == 'berkley_earth':
        da = da.rename({'latitude':'lat','longitude':'lon'})
        
    da = da.resample(time=freq,
                     closed='left',
                     label='left').mean('time') #this mean doesn't make sense for land cover maps
    
    # for freq of 25Y, collapse time dimenson by taking diff of 2 25 yr periods
    if freq == '25Y':
        
        da = da.isel(time=-1) - da.isel(time=0)
    
    return da

#%%============================================================================

def ar6_mask(da):
    
    lat = da.lat.values
    lon = da.lon.values
    ar6_regs = rm.defined_regions.ar6.land.mask(lon,lat)
    landmask = rm.defined_regions.natural_earth.land_110.mask(lon,lat)
    ar6_regs = ar6_regs.where(landmask == 0)
    
    return ar6_regs

#%%============================================================================

def cnt_mask(sfDIR,
             da):
    
    lat = da.lat.values
    lon = da.lon.values
    ar6_regs = rm.defined_regions.ar6.land.mask(lon,lat)
    landmask = rm.defined_regions.natural_earth.land_110.mask(lon,lat)    
    ar6_regs = ar6_regs.where(landmask == 0)
    ar6_land = xr.where(ar6_regs > 0,1,0)
    os.chdir(sfDIR)
    gpd_continents = gp.read_file('IPCC_WGII_continental_regions.shp')
    gpd_continents = gpd_continents[(gpd_continents.Region != 'Antarctica')&(gpd_continents.Region != 'Small Islands')]
    cnt_regs = rm.mask_geopandas(gpd_continents,lon,lat)
    cnt_regs = cnt_regs.where((ar6_regs != 0)&(ar6_regs != 20)&(ar6_land == 1))
    
    return cnt_regs

#%%============================================================================

def ar6_weighted_mean(continents,
                      da,
                      ar6_regs,
                      nt,
                      ns,
                      weight,
                      ar6_wts):
    
    # nt = len(da.time.values)
    matrix = np.zeros(shape=(nt,ns))
    s = 0
    for c in continents.keys():
        for i in continents[c]:
            da_i = da.where(ar6_regs==i,
                            drop=True)   
            if nt > 1:                 
                for t in np.arange(nt):
                    da_i_t = da_i.isel(time=t)
                    weights = np.cos(np.deg2rad(da_i_t.lat))
                    weights.name='weights'
                    if weight == 'no_weights':
                        da_i_t_w = da_i_t.weighted(weights).mean(('lon','lat')).values
                    if weight == 'weights':
                        da_i_t_w = da_i_t.weighted(weights).mean(('lon','lat')).values        
                        da_i_t_w = da_i_t_w * ar6_wts[c][i]            
                    matrix[t,s]= da_i_t_w
            elif nt == 1:
                t = 0
                weights = np.cos(np.deg2rad(da_i.lat))
                weights.name='weights'
                if weight == 'no_weights':
                    da_i_w = da_i.weighted(weights).mean(('lon','lat')).values
                if weight == 'weights':
                    da_i_w = da_i.weighted(weights).mean(('lon','lat')).values        
                    da_i_w = da_i_w * ar6_wts[c][i]            
                matrix[t,s]= da_i_w
            s += 1
                
            
    return matrix

#%%============================================================================

def cnt_weighted_mean(continents,
                      da,
                      cnt_regs,
                      nt,
                      ns,
                      weight,
                      cnt_wts):
    
    # nt = len(da.time.values)
    matrix = np.zeros(shape=(nt,ns))
    s = 0
    for c in continents.keys():
        da_i = da.where(cnt_regs==continents[c],
                        drop=True) 
        if nt > 1:             
            for t in np.arange(nt):
                da_i_t = da_i.isel(time=t)
                weights = np.cos(np.deg2rad(da_i_t.lat))
                weights.name='weights'
                if weight == 'no_weights': # do not weigh based on continent size
                    da_i_t_w = da_i_t.weighted(weights).mean(('lon','lat')).values
                elif weight == 'weights':
                    da_i_t_w = da_i_t.weighted(weights).mean(('lon','lat')).values
                    da_i_t_w = da_i_t_w * cnt_wts[c]
                matrix[t,s]= da_i_t_w
        elif nt == 1:
            t = 0
            weights = np.cos(np.deg2rad(da_i.lat))
            weights.name='weights'
            if weight == 'no_weights': # do not weigh based on continent size
                da_i_w = da_i.weighted(weights).mean(('lon','lat')).values
            elif weight == 'weights':
                da_i_w = da_i.weighted(weights).mean(('lon','lat')).values
                da_i_w = da_i_w * cnt_wts[c]
            matrix[t,s]= da_i_w
        s += 1            
            
    return matrix

#%%============================================================================

def del_rows(matrix):
    
    # remove tsteps with nans (temporal x spatial shaped matrix)
    del_rows = []
    
    for i,row in enumerate(matrix):
        
        nans = np.isnan(row)
        
        if True in nans:
            
            del_rows.append(i)
            
    matrix = np.delete(matrix,
                       del_rows,
                       axis=0)
            
    return matrix

#%%============================================================================

def temp_center(
    nt,
    ns,
    mod_ar6):
    
    if nt > 1:
        for s in np.arange(ns):
            arr = mod_ar6[:,s]
            arr_mean = np.mean(arr)
            arr_center = arr - arr_mean
            mod_ar6[:,s] = arr_center
            
    elif nt == 1:
        arr_mean = np.mean(mod_ar6)
        mod_ar6 = mod_ar6 - arr_mean   
        
    return mod_ar6

#%%============================================================================

def ensembler(data_list,
              ax=False):
    if not ax:
        matrix = np.stack(data_list,axis=1)
        nx = np.shape(matrix)[1]
        ens_mean = np.mean(matrix,axis=1)
    
    if ax:
        matrix = np.stack(data_list,axis=ax)
        nx = np.shape(matrix)[ax]
        ens_mean = np.mean(matrix,axis=ax)
    
    return ens_mean,nx

#%%============================================================================

def da_ensembler(data):
    
    concat_dim = np.arange(len(data))
    aligned = xr.concat(data,dim=concat_dim)
    mean = aligned.mean(dim='concat_dim')
    
    return mean

#%%============================================================================

def eigvalvec(C):
    """
    Eigenvalue / Eigenvector calculation
    """
    ## Compute eigenvalues and eigenvectors
    eigval0, eigvec0 = spla.eigh(C) 
    ## Take real part (to avoid numeric noise, eg small complex numbers)
    if np.max(np.imag(eigval0))/np.max(np.real(eigval0)) > 1e-12:
        print("Matrix is not symmetric")
        return
    ## Check that C is symetric (<=> real eigen-values/-vectors)
    eigval1 = np.real(eigval0)
    eigvec1 = np.real(eigvec0)
    ## Sort in a descending order
    dorder = np.argsort(eigval1)[::-1]
    eigvec = np.flipud(eigvec1)[:, dorder]
    eigval = eigval1[dorder]
    return eigval, eigvec

#%%============================================================================

def projfullrank(t, 
                 s):
    """
    Projection on full rank matrix
    """
    ## M: the matrix corresponding to the temporal centering
    M = np.eye(t, t) - np.ones([t, t])/float(t)
    ## Eigen-values/-vectors of M; note that rk(M)=T-1, so M has one eigenvalue equal to 0.
    eigval, eigvec = eigvalvec(M)
    ## (T-1) first eigenvectors (ie the ones corresponding to non-zero eigenvalues)
    eigvec = eigvec[:, :t-1].T
    ## The projection matrix P, which consists in S replications of U.
    P = np.zeros([(t-1)*s, t*s])
    for i in range(s):
        P[i:(t-1)*s:s, i:t*s:s] = eigvec
    return P

#%%============================================================================

def regC(X):
    """
    Calculation regularized CoVariance matrix
    """
    # just to be sure it is a matrix object
    X = np.matrix(X)
    n, p = np.shape(X)
    # Sample covariance
    CE = X.T * X / float(n)		

    Ip = np.eye(p, p)
    # First estimate in L&W
    m = np.trace(CE * Ip) / float(p)	
    XP = CE - m * Ip
    # Second estimate in L&W
    d2 = np.trace(XP * XP.T) / float(p) 	

    bt = []
    for i in range(n):
        Mi = X[i, :].T * X[i, :]
        bt.append(np.trace((Mi - CE) * (Mi - CE).T) / float(p))
    bb2 = 1. / n**2 * np.sum(bt)
    # Third estimate in L&W
    b2 = np.min([bb2, d2])	
    # Fourth estimate in L&W
    a2 = d2 - b2		
    Cr = b2 * m / d2 * Ip + a2 / d2 * CE
    return Cr

#%%============================================================================

def beta_calc(
    X,
    cov,
    y,
    ):
    
    cov_inv = np.real(spla.inv(cov))
    Ft = np.transpose(np.dot(np.dot(spla.inv(np.dot(np.dot(X.T,cov_inv), X)), X.T), cov_inv))
    b = np.dot(y.T, Ft)
    
    return b

#%%============================================================================

def it_step1(
    cov_mod, # om
    beta_t, # beta_t_i
    cov_pic, # cov_z1
    y_bar_t, # y_bar_i
    x_t, # x_t_i
    comp='s1',
    ):
    
    if comp == 's1': # original eqn for step 1
        
        cov_mod_inv = np.real(spla.inv(cov_mod))
        cov_pic_inv = np.real(spla.inv(cov_pic))
        m1 = np.real(
            spla.inv(
                np.matrix(
                    (cov_mod_inv + np.dot(beta_t**2,cov_pic_inv))
                    )
                )
            )
        m2 = np.matrix(
            (beta_t * np.dot(cov_pic_inv,y_bar_t) + np.dot(cov_mod_inv,x_t))
            )
        x_t_ast = np.dot(m1,m2)
        
    elif comp == 'd1': # appendix D/ eqn D1 for step 1
        
        cov_mod_sqrt = spla.fractional_matrix_power(cov_mod, -0.5)
        cov_prod = np.dot(np.dot(cov_mod_sqrt,cov_pic),cov_mod_sqrt)
        w,v = spla.eigh(cov_prod)
        delta = np.diag(w)
        # m1 = np.real(
        #     spla.inv(
        #         np.matrix(
        #             (delta + beta_t**2 * np.identity(len(w)))
        #             )
        #         )
        #     )
        m1 = spla.inv(
                np.matrix(
                    (delta + beta_t**2 * np.identity(len(w)))
                    )
                )
                
        y_bar_tlde = np.dot(np.dot(v,cov_mod_sqrt),y_bar_t)
        x_t_tlde = np.dot(np.dot(v,cov_mod_sqrt),x_t)
        m2 = beta_t * y_bar_tlde + np.dot(delta,x_t_tlde)
        x_t_tlde = np.dot(m1,m2) # test for i=1 (lu) against normal eqn to see if this needs further transformation to undo tlde
        x_t_ast = np.dot(
            spla.inv(cov_mod_sqrt),np.dot(spla.inv(v),x_t_tlde)
            )
        
    return x_t_ast

#%%============================================================================

def extract_Z2(NZ, 
               frac_Z2, 
               sampling_name):
    """
    Z1 and Z2 based on control
    """
    Ind_Z2 = np.zeros((int(NZ), 1))
    NZ2 = int(np.floor(NZ * frac_Z2))
    if sampling_name == 'segment':
        Ind_Z2[0:NZ2] = 1
        print('Z2 : segment 1-'+str(NZ2)+', fraction ~ '+str(NZ2/NZ))
    elif sampling_name == 'regular':
        ix = []
        a = 0
        while a <= NZ - 1./frac_Z2:
            a += 1./frac_Z2
            ix.append(np.int(np.floor(a)) - 1)
        Ind_Z2[ix] = 1
        print('Z2 : regular, fraction ~ '+str(sum(Ind_Z2)/NZ))
    elif sampling_name == 'random':
        u = np.random.normal(0, 1, size=(NZ, 1))
        z = np.argsort(u, axis=0)[::-1]
        Ind_Z2[z[0:NZ2]] = 1
        print('Z2 : random, fraction ~ '+str(sum(Ind_Z2)/NZ))
    else:
        print('Unknown sampling_name.')
    return Ind_Z2

#%%============================================================================

def gke(d_H0, 
        d):
    """
    Silverman's rule of Thumb
    """
    N = len(d_H0)
    h = 1.06 * np.std(d_H0, ddof=1) * N ** (-1./5)	# Silverman's rule of Thumb
    onem = sps.norm.cdf(d, d_H0, h)
    pvi = 1 - onem
    pv = np.sum(pvi)/N

    return pv

#%%============================================================================

def tls(X, 
        Y, 
        Z2, 
        nX, 
        PROJ, 
        Formule_IC_TLS, 
        ci_bnds):
    """
    TLS routine
    """
    n = Y.shape[1]
    m = X.shape[0]
    # Check sizes of X and Y
    if Y.shape[1] != X.shape[1]:
        print('Error in TLS: size of inputs X, Y.')
        return
    # Normalise the variance of X
    X = np.multiply(X, (np.sqrt(nX).T * np.ones((1, n))))
    if X.shape[0] == 1: # adjusted
        DnX = np.sqrt(nX).squeeze()
    else:
        DnX = np.diag(np.sqrt(nX).A1)
    # Computation of beta_hat
    #--------------------------
    # TLS fit via svd...
    M = np.vstack([X, Y])
    U, D, V = np.linalg.svd(M)
    V = V.T
    
    # Consider the "smallest" singular vector
    Uk = U[:, -1]
    Uk_proj = np.vstack([PROJ * DnX * Uk[:-1], Uk[-1]])
    # Computes beta_hat
    beta_hat = - Uk_proj[:-1] / Uk_proj[-1]
    # instantiate array for beta uncertainty estimates
    beta_hat_inf = np.zeros(beta_hat.shape)
    beta_hat_sup = np.zeros(beta_hat.shape)
    # Reconstructed data
    D_tilde = np.matrix(np.zeros(M.shape))
    np.fill_diagonal(D_tilde, D)
    D_tilde[m, m] = 0
    Z_tilde = U * D_tilde * V.T
    X_tilde = Z_tilde[0:m, :] / (np.dot(np.sqrt(nX).T, np.ones((1, n))))
    Y_tilde = Z_tilde[m, :]
    # Computation of Confidence Intervals
    #--------------------------------------
    # The square singular values (denoted by lambda in AS03)
    d = D**2
    # Computation of corrected singular value (cf Eq 34 in AS03)
    d_hat = np.zeros(d.shape)
    NAZv = Z2.shape[0]
    for i in range(len(d)):
        vi = V[:, i].T
        if Formule_IC_TLS == "AS03":
            # Formule Allen & Stott (2003)
            d_hat[i] = d[i] / np.dot(np.dot(np.dot(vi, Z2.T), Z2 / NAZv), vi.T)
        elif Formule_IC_TLS == "ODP": 
             # Formule ODP (Allen, Stone, etc)
            d_hat[i] = d[i] / np.dot(np.power(vi, 2), np.sum(np.power(Z2, 2), axis=0).T / NAZv)
        else:
            print('tls_v1.sci : unknown formula for computation of TLS CI.')
    # The "last" corrected singular value will be used in the Residual Consistency Check
    d_cons = d_hat[-1]
    # Threshold of the Fisher distribution, used for CI computation (cf Eq 36-37 in AS03)
    seuil_1d = np.sqrt(sps.f.ppf(ci_bnds, 1, NAZv))
    # In order to compute CI, we need to run through the m-sphere (cf Eq 30 in AS03)
    # Nb of pts on the (m-)sphère...
    npt = 1000	
    if m == 1:
        Pts = np.array([[1], [-1]])
    else:
        Pts_R = np.random.normal(0, 1, size=(npt, m))
        # The points on the sphere
        Pts = Pts_R / (np.sqrt(np.sum(Pts_R ** 2, axis=1).reshape((npt, 1)) * np.ones((1, m))))
    # delta_d_hat provides the diagonal of the matrix used in Eq 36 in AS03
    delta_d_hat = d_hat - np.min(d_hat)
    # following notation of Eq 30 in AS03
    a = seuil_1d * Pts			
    arg_min = np.nan
    arg_max = np.nan
    # Check that 0 is not reached before the last index of delta_d_hat:
    if True not in (delta_d_hat[:-1] == 0):
        b_m1 = a / np.dot(np.ones((Pts.shape[0], 1)), np.sqrt(delta_d_hat[:-1]).reshape((1, delta_d_hat[:-1].shape[0])))
        # following notation of Eq 31 in AS03
        #b_m2 = np.sqrt(1 - np.sum(b_m1**2, axis=1))	
        b_m2 = np.matrix(np.sqrt(1 - np.sum(b_m1**2, axis=1))).T
        # b_m2 need to be strctly positive, otherwise the CI will be unbounded
        if (False in np.isreal(b_m2)) | (True in (b_m2 == 0)) | (True in np.isnan(b_m2)):
            print('Unbounded CI (2)', np.max(np.imag(b_m2)))
            beta_hat_inf += np.nan
            beta_hat_sup += np.nan
        else:
            # Then in order to CI that include +/- infinity, the computation are made in terms of angles,
            # based on complex numbers (this is a descrepancy with ODP)
            V_pts = np.dot(np.column_stack([b_m1, b_m2]), U.T)
            V_pts_proj = np.column_stack([np.dot(np.dot(V_pts[:, :-1], DnX), PROJ.T), V_pts[:, -1]])
            for i in range(m):
                Vc_2d_pts = V_pts_proj[:, i] + V_pts_proj[:, -1] * 1j 
                Vc_2d_ref = Uk_proj[i] + Uk_proj[-1] * 1j
                Vprod_2d = Vc_2d_pts / Vc_2d_ref
                arg = np.sort(np.imag(np.log(Vprod_2d)), axis=0)
                delta_arg_min = arg[0]
                delta_arg_max = arg[-1]
                Delta_max_1 = np.max(arg[1:] - arg[:-1])
                k1 = np.argmax(arg[1:] - arg[:-1])
                Delta_max = np.max([Delta_max_1, arg[0] - arg[-1] + 2 * np.pi])
                k2 = np.argmax([Delta_max_1, arg[0] - arg[-1] + 2 * np.pi])
                if Delta_max < np.pi:
                    beta_hat_inf[i] = np.nan
                    beta_hat_sup[i] = np.nan
                else:
                    if k2 != 1:
                        print("Warning k2")
                    arg_ref = np.imag(np.log(Vc_2d_ref))
                    arg_min = delta_arg_min + arg_ref
                    arg_max = delta_arg_max + arg_ref
                    beta_hat_inf[i] = -1 / np.tan(arg_min)
                    beta_hat_sup[i] = -1 / np.tan(arg_max)
    else:    
        # If 0 is reached before last index of delta_d_hat, the CI will be unbounded
        print('Unbounded CI (1)')
        beta_hat_inf += np.nan
        beta_hat_sup += np.nan
    return beta_hat, beta_hat_inf, beta_hat_sup, d_cons, X_tilde, Y_tilde

#%%============================================================================

def consist_mc_tls(Sigma, 
                   X0, 
                   nb_runs_X, 
                   n1, 
                   n2, 
                   N, 
                   Formula):
    """
    Consistency check TLS
    """
    # Check that Sigma is a square matrix
    n = Sigma.shape[0]
    if (Sigma.shape[1] != n) | (X0.shape[0] != n):
        print("Error of size in consist_mc_tls.sci")
        return
    # Number of external forcings considered
    k = X0.shape[1]
    # Initial value of beta for the Monte Carlo simulations
    beta0 = np.ones((k,1))
    # Monte Carlo simulations
    #-------------------------
    Sigma12 = spla.sqrtm(Sigma)
    d_cons_H0 = np.zeros((N,1))
    for i in range(N):
        # Virtual observations Y
        Yt = np.dot(X0, beta0)
        Y = Yt + np.dot(Sigma12, np.random.normal(0, 1, size=(n, 1)))
        # Virtual noised response patterns X
        X = X0 + np.dot(Sigma12, np.random.normal(0, 1, size=(n, k)) / (np.ones(Yt.shape) * np.sqrt(nb_runs_X)))
        # Variance normalised X
        Xc = np.multiply((np.dot(np.ones(Yt.shape), np.sqrt(nb_runs_X))), X)
        # Virtual independent samples of pure internal variability, Z1 and Z2
        Z1 = np.dot(Sigma12, np.random.normal(0, 1, size=(n, n1)))
        Z2 = np.dot(Sigma12, np.random.normal(0, 1, size=(n, n2)))
        # Virtual estimated covariance matrix (based on Z1 only)
        C1_hat = regC(Z1.T)
        C12 = spla.inv(spla.sqrtm(C1_hat))
        # The following emulates the TLS algorithm and computes the variable used in the RCC (which is written in d_cons_H0). See also tls_v1.sci.
        # Xc and Y are prewhitened
        M = np.dot(C12, np.column_stack([Xc, Y]))
        U, D, V = np.linalg.svd(M.T)
        V = V.T
        d = D**2
        nd = len(d)
        vi = V[:,nd].T

        if Formula == "AS03":
            # Z2 is prewhitened
            Z2w = np.dot(C12, Z2).T
            # Formule Allen & Stott (2003)
            d_cons_H0[i,0] = d[nd-1] / np.dot(np.dot(np.dot(vi, Z2w.T), Z2w / n2), vi.T)
        elif Formula == "ODP":
            # Z2 is prewhitened
            Z2w = np.dot(C12, Z2).T
            # Formule ODP (Allen, Stone, etc)
            d_cons_H0[i,0] = d[nd-1] / np.dot(np.power(vi,2), np.sum(np.power(Z2w, 2), axis=0).T / n2)
        else:
            print("consist_mc_tls.sci : unknown formula for computation of RCC.")

    return d_cons_H0

#%%============================================================================

def da_run(
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
    var_sfs):
    
    # =============================================================================
    # y = obs
    # X = fp
    # ctl = ctl
    # nb_runs_x= nx
    # reg = 'TLS' # regression type (total least squares algorithm)
    # cons_test = 'MC' # choice of residual consistency test (MC for monte carlo)
    # formule_ic_tls = 'ODP' # formula for calculating confidence intervals in TLS (ODP from optimal detection package)
    # trunc = 0 # spherical harmonics truncation
    # =============================================================================

    # Input parameters
    y = np.matrix(y).T
    X = np.matrix(X).T
    Z = np.transpose(np.matrix(ctl))
    nb_runs_x = np.matrix(nb_runs_x)
    # Number of Time steps
    nbts = nt
    # Spatial dimension
    n_spa = ns
    # Spatio_temporal dimension (ie dimension of y)
    n_st = n_spa * nbts
    # number of different forcings
    I = X.shape[1]
    
    # Create even sample numbered Z1 and Z2 samples of internal vari from cntrl runs
    nb_runs_ctl = np.shape(Z)[1]
    half_1_end = int(np.floor(nb_runs_ctl / 2))
    Z1 = Z[:,:half_1_end]
    if nb_runs_ctl % 2 == 0:
        Z2 = Z[:,half_1_end:]
    else:
        Z2 = Z[:,half_1_end+1:]
        
    
    
    # Spatio-temporal dimension after reduction
    if nbts > 1:
        n_red = n_st - n_spa
        U = projfullrank(nbts, n_spa)
    elif nbts == 1: # case where I regress with collapsed time dimension and each point is a signal
        n_red = n_st - nbts # therefore treating ns as nt
        U = projfullrank(n_spa, nbts)
    
    # Project all input data 
    yc = np.dot(U, y)
    Z1c = np.dot(U, Z1)
    Z2c = np.dot(U, Z2)
    Xc = np.dot(U, X)
    proj = np.identity(X.shape[1]) # already took factorial LU, so no need for proj to use dot to decipher histnl + lu from hist, histnl
    
    # Statistical estimation
    ## Regularised covariance matrix
    Cf = regC(Z1c.T)
    
    # normal covariance matrices from split samples
    cov_Z1 = np.dot(Z1c,Z1c.T) / Z1c.shape[1]
    cov_Z1_inv = np.real(spla.inv(cov_Z1))
    cov_Z2 = np.dot(Z2c,Z2c.T) / Z2c.shape[1]
    cov_Z2_inv = np.real(spla.inv(cov_Z2))
    
    
    
    x_0 = Xc
    
    
    Cf1 = np.real(spla.inv(Cf))
    
    if reg == 'OLS':
        ## OLS algorithm
        pv_consist = np.nan
        Ft = np.transpose(np.dot(np.dot(spla.inv(np.dot(np.dot(Xc.T, Cf1), Xc)), Xc.T), Cf1))
        beta_hat = np.dot(np.dot(yc.T, Ft), proj.T)
        ## 1-D confidence intervals
        NZ2 = Z2c.shape[1]
        var_valid = np.dot(Z2c, Z2c.T) / NZ2
        var_beta_hat = np.dot(np.dot(np.dot(np.dot(proj, Ft.T), var_valid), Ft), proj.T)
        beta_hat_inf = beta_hat - sps.t.ppf(ci_bnds, NZ2) * np.sqrt(np.diag(var_beta_hat))
        beta_hat_sup = beta_hat + sps.t.ppf(ci_bnds, NZ2) * np.sqrt(np.diag(var_beta_hat))
        ## Consistency check
        # print('Residual Consistency Check')
        epsilon = yc - np.dot(np.dot(Xc, proj.T), beta_hat.T)
        if  cons_test == "OLS_AT99":
            # Formula provided by Allen & Tett (1999)
            d_cons = np.dot(np.dot(epsilon.T, np.linalg.pinv(var_valid)), epsilon) / (n_red - I)
            pv_cons = 1 - sps.f.cdf(d_cons, n_red - I, NZ2)
        elif cons_test == "OLS_Corr":
            # Hotelling Formula
            d_cons = np.dot(np.dot(epsilon.T, np.linalg.pinv(var_valid)), epsilon)/(NZ2*(n_red-I))*(NZ2-n_red+1)
            if NZ2-n_red + 1 > 0:
                pv_cons = 1 - sps.f.cdf(d_cons, n_red - I, NZ2 - n_red + 1)
            else:
                pv_cons = np.nan
        else:
            print('Unknown Cons_test : ', cons_test)
    
    elif reg == 'TLS':
        ## TLS algorithm
        c0, c1, c2, d_cons, x_tilde_white, y_tilde_white = tls(np.dot(Xc.T, Cf12), np.dot(yc.T, Cf12), np.dot(Z2c.T, Cf12),\
                                                               nb_runs_x, proj, formule_ic_tls, ci_bnds)
        x_tilde = np.dot(Cfp12, x_tilde_white.T)
        y_tilde = np.dot(Cfp12, y_tilde_white.T)
        beta_hat = c0.T
        beta_hat_inf = c1.T
        beta_hat_sup = c2.T
    
        # Consistency check
        print("Residual Consistency Check")
        NZ1 = Z1c.shape[1]
        NZ2 = Z2c.shape[1]
    
        if  cons_test == 'MC':
            ## Null-distribution sampled via Monte-Carlo simulations
            ## Note: input data here need to be pre-processed, centered, etc.
            ## First, simulate random variables following the null-distribution
            N_cons_mc = 1000
            d_H0_cons = consist_mc_tls(Cf, Xc, nb_runs_x, NZ1, NZ2, N_cons_mc, formule_ic_tls)
            ## Evaluate the p-value from the H0 sample (this uses gke = gaussian kernel estimate)
            pv_cons = gke(d_H0_cons, d_cons)
        elif cons_test == "AS03":
            ## Formula provided by Allen & Stott (2003)
            pv_cons = 1 - sps.f.cdf(d_cons / (n_red-I), n_red-I, NZ2)
        else:
            pv_cons = np.nan
        Ft = np.nan # added because existing code wanted this out of function, so TLS didn't work
    
    beta = np.zeros((4, I))
    beta[:-1, :] = np.concatenate((beta_hat_inf, beta_hat, beta_hat_sup))
    beta[-1, 0] = pv_cons
    
    return beta,nb_runs_ctl,proj,U,yc,Z1c,Z2c,Xc,Cf1,Ft,beta_hat


#%%============================================================================

def input_pickler(pklDIR,
                  flag_factor,
                  dictionary,
                  grid,
                  pi,
                  agg,
                  weight,
                  freq,
                  t_ext,
                  reg,
                  obs_mod):
    
    os.chdir(pklDIR)
    if obs_mod == 'mod':
        pkl_file = open('mod_inputs_{}-flagfactor_{}-grid_{}-pi_{}-agg_{}-weight_{}_{}_{}.pkl'.format(flag_factor,grid,pi,agg,weight,freq,t_ext,reg),'wb')
    elif obs_mod == 'obs':
        pkl_file = open('obs_inputs_{}-grid_{}-pi_{}-agg_{}-weight_{}_{}_{}.pkl'.format(grid,pi,agg,weight,freq,t_ext,reg),'wb')
    elif obs_mod == 'pic':
        pkl_file = open('pic_inputs_{}-grid_{}-pi_{}-agg_{}-weight_{}_{}_{}.pkl'.format(grid,pi,agg,weight,freq,t_ext,reg),'wb')
    elif obs_mod == 'omega':
        pkl_file = open('omega_inputs_{}-grid_{}-pi_{}-agg_{}-weight_{}_{}_{}.pkl'.format(grid,pi,agg,weight,freq,t_ext,reg),'wb')        
    pk.dump(dictionary,pkl_file)
    pkl_file.close()

#%%============================================================================

def ts_pickler(pklDIR,
               ts,
               grid,
               pi,
               agg,
               weight,
               t_ext,
               obs_mod):
    
    os.chdir(pklDIR)
    if obs_mod == 'mod':
        pkl_file = open('mod_ts_{}-grid_{}-pi_{}-agg_{}-weight_{}.pkl'.format(grid,pi,agg,weight,t_ext),'wb')
    elif obs_mod == 'obs':
        pkl_file = open('obs_ts_{}-grid_{}-pi_{}-agg_{}-weight_{}.pkl'.format(grid,pi,agg,weight,t_ext),'wb')
    elif obs_mod == 'pic':
        pkl_file = open('pic_ts_{}-grid_{}-pi_{}-agg_{}-weight_{}.pkl'.format(grid,pi,agg,weight,t_ext),'wb')
    pk.dump(ts,pkl_file)
    pkl_file.close()
    

#%%============================================================================

def pickler(curDIR,
            var_fin,
            analysis,
            grid,
            agg,
            weight,
            freq,
            t_ext,
            bs_reps,
            exp_list,
            pi):
    
    os.chdir(curDIR)
    if len(exp_list) == 2:
        pkl_file = open('var_fin_2-factor_{}-grid_{}_{}-pi_{}-agg_{}-weight_{}-bsreps_{}_{}.pkl'.format(grid,analysis,pi,agg,weight,bs_reps,freq,t_ext),'wb')
    elif len(exp_list) == 1:
        exp = exp_list[0]
        pkl_file = open('var_fin_1-factor_{}_{}-grid_{}_{}-pi_{}-agg_{}-weight_{}-bsreps_{}_{}.pkl'.format(exp,grid,analysis,pi,agg,weight,bs_reps,freq,t_ext),'wb')
    pk.dump(var_fin,pkl_file)
    pkl_file.close()

#%%============================================================================

def det_finder(list):
    hi = list[0]
    low = list[2]
    if all(i > 0 for i in list[:3]):
    
        k = 1 # detection
        
        if ((list[0] > 1) and (list[2] < 1)):
            
            k = 2 # attribution
            
    elif any(i < 0 for i in list[:3]):
        
        k = 3 # none
        
    return k

#%%============================================================================

def scale_take(array): #must take diff between b and sup/inf, store in separate lists
    b = array[1]
    b_inf = b - array[0]
    b_sup = array[2] - b
    p = array[3]
    return b,b_inf,b_sup,p

#%%============================================================================

def plot_scaling_global(
    models,
    grid,
    obs_types,
    pi,
    agg,
    weight,
    exp_list,
    var_fin,
    freq,
    reg,
    flag_svplt,
    outDIR):
    
    if exp_list == ['hist-noLu','lu']:
        
        cmap_whole = plt.cm.get_cmap('PRGn')
        cols={}
        median_cols={}
        cols['hist-noLu'] = cmap_whole(0.25)
        cols['lu'] = cmap_whole(0.75)
        median_cols['hist-noLu'] = cmap_whole(0.05)
        median_cols['lu'] = cmap_whole(0.95)
        
    elif exp_list == ['historical','hist-noLu']:
    
        cmap_whole = plt.cm.get_cmap('BrBG')
        cols={}
        median_cols={}
        cols['historical'] = cmap_whole(0.25)
        cols['hist-noLu'] = cmap_whole(0.75)
        median_cols['historical'] = cmap_whole(0.05)
        median_cols['hist-noLu'] = cmap_whole(0.95)
    
    letters=['a','b','c','d','e','f','g']
        
    x=25
    y=4
    
    x0 = 0.1
    y0 = 0.95
    xlen = 0.4
    ylen = 0.5
    inset_font = 18
    
    # y ticks temporal OF insets
    yticks_OF = np.arange(-0.5,2.5,0.5)
    ytick_labels_OF = [None, '0', None, '1', None, '2']
    
    # OF data prep
    b = {}
    b_inf = {}
    b_sup = {}
    p = {}
    err = {}
    
    for obs in obs_types:
    
        f,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=1,
                                        ncols=5,
                                        figsize=(x,y),
                                        sharey=True)
        
        # OF data prep
        b[obs] = {}
        b_inf[obs] = {}
        b_sup[obs] = {}
        p[obs] = {}
        err[obs] = {}
        
        for mod in models:
            b[obs][mod] = {}
            b_inf[obs][mod] = {}
            b_sup[obs][mod] = {}
            p[obs][mod] = {}
            err[obs][mod] = {}
            for exp in exp_list:
                b[obs][mod][exp],\
                b_inf[obs][mod][exp],\
                b_sup[obs][mod][exp],\
                p[obs][mod][exp] = scale_take(var_fin[obs][mod][exp])
                err[obs][mod][exp] = np.stack([[b_inf[obs][mod][exp]],
                                              [b_sup[obs][mod][exp]]],
                                              axis=0)
        
        
        count = 0
        for ax,mod in zip((ax1,ax2,ax3,ax4,ax5),models):
            
            if exp_list == ['hist-noLu','lu']:
        
                ax.errorbar(x=b[obs][mod]['hist-noLu'],
                            y=b[obs][mod]['lu'],
                            xerr=err[obs][mod]['hist-noLu'],
                            fmt='o',
                            markersize=3,
                            ecolor=cols['hist-noLu'],
                            markerfacecolor=median_cols['hist-noLu'],
                            mec=cols['lu'],
                            capsize=5,
                            elinewidth=4,
                            markeredgewidth=1)
                
                ax.errorbar(x=b[obs][mod]['hist-noLu'],
                            y=b[obs][mod]['lu'],
                            yerr=err[obs][mod]['lu'],
                            fmt='o',
                            markersize=3,
                            ecolor=cols['lu'],
                            markerfacecolor=median_cols['lu'],
                            mec=cols['lu'],
                            capsize=5,
                            elinewidth=4,
                            markeredgewidth=1)
                
            elif exp_list == ['historical','hist-noLu']:
                
                ax.errorbar(x=b[obs][mod]['hist-noLu'],
                            y=b[obs][mod]['historical'],
                            xerr=err[obs][mod]['hist-noLu'],
                            fmt='o',
                            markersize=3,
                            ecolor=cols['hist-noLu'],
                            markerfacecolor=median_cols['hist-noLu'],
                            mec=cols['historical'],
                            capsize=5,
                            elinewidth=4,
                            markeredgewidth=1)
                
                ax.errorbar(x=b[obs][mod]['hist-noLu'],
                            y=b[obs][mod]['historical'],
                            yerr=err[obs][mod]['historical'],
                            fmt='o',
                            markersize=3,
                            ecolor=cols['historical'],
                            markerfacecolor=median_cols['historical'],
                            mec=cols['historical'],
                            capsize=5,
                            elinewidth=4,
                            markeredgewidth=1)
            
            ax.set_title(letters[count],
                        loc='left',
                        fontweight='bold',
                        fontsize=inset_font)
            
            ax.set_title(mod,
                        loc='center',
                        fontweight='bold',
                        fontsize=inset_font)
            ax.set_xticks([])
            ax.set_xlim((-1,2))
            ax.set_ylim((-1,2))
            
            count += 1
            
            ax.hlines(y=1,
                    xmin=-1,
                    xmax=2,
                    colors='k',
                    linestyle='dashed',
                    linewidth=1)
            ax.hlines(y=0,
                    xmin=-1,
                    xmax=2,
                    colors='k',
                    linestyle='solid',
                    linewidth=0.25)
            
            ax.vlines(x=1,
                    ymin=-1,
                    ymax=2,
                    colors='k',
                    linestyle='dashed',
                    linewidth=1)
            ax.vlines(x=0,
                    ymin=-1,
                    ymax=2,
                    colors='k',
                    linestyle='solid',
                    linewidth=0.25)
            
            ax.tick_params(axis="x",
                        direction="in", 
                        left="off",
                        labelleft="on")
            ax.tick_params(axis="y",
                        direction="in")
            
            ax.set_xlabel(r'$\beta_{HISTNL}$',
                        fontsize=20,
                        color=median_cols['hist-noLu'],
                        fontweight='bold')
            
        if exp_list == ['hist-noLu','lu']:
        
            ax1.set_ylabel(r'$\beta_{LU}$',
                        fontsize=20,
                        color=median_cols['lu'],
                        fontweight='bold')
            
        elif exp_list == ['historical','hist-noLu']:
            
            ax1.set_ylabel(r'$\beta_{HIST}$',
                        fontsize=20,
                        color=median_cols['historical'],
                        fontweight='bold')
        
        if flag_svplt == 0:
            pass
        elif flag_svplt == 1:
            
            if exp_list == ['hist-noLu','lu']:
                
                f.savefig(outDIR+'/global_attribution_2-factor_{}_{}_{}-grid_{}-pi_{}-agg_{}-weight_{}.png'.format(reg,obs,grid,pi,agg,weight,freq),bbox_inches='tight',dpi=200)     
                
            elif exp_list == ['historical','hist-noLu']:
                
                f.savefig(outDIR+'/global_attribution_1-factor_{}_{}_{}-grid_{}-pi_{}-agg_{}-weight_{}.png'.format(reg,obs,grid,pi,agg,weight,freq),bbox_inches='tight',dpi=200)     
  
#%%============================================================================
                
def plot_scaling_map_continental(
    sfDIR,
    obs_types,
    pi,
    agg,
    weight,
    models,
    exp_list,
    continents,
    var_fin,
    grid,
    letters,
    freq,
    flag_svplt,
    outDIR):
    
    data = var_fin
    os.chdir(sfDIR)
    
    #test for merging ar6 into desired continent extents
    regions = gp.read_file('IPCC-WGI-reference-regions-v4.shp')
    gpd_continents = gp.read_file('IPCC_WGII_continental_regions.shp')
    gpd_continents = gpd_continents[(gpd_continents.Region != 'Antarctica')&(gpd_continents.Region != 'Small Islands')]
    regions = gp.clip(regions,gpd_continents)
    regions['keep'] = [0]*len(regions.Acronym)
    
    for c in continents.keys():
        for ar6 in continents[c]:
            regions.at[ar6,'Continent'] = c
            regions.at[ar6,'keep'] = 1
    
    regions = regions[regions.keep!=0]  
    regions = regions.drop(columns='keep')
    ar6_continents = regions.dissolve(by='Continent')

    for obs in obs_types:
        for mod in models:
            for exp in exp_list:
                ar6_continents['{}-grid {} {}'.format(obs,mod,exp)] = [1] * len(ar6_continents.Acronym)

    for obs in obs_types:
        for mod in models:
            for exp in exp_list:
                for c in continents.keys():
                    ar6_continents.loc[c,'{}-grid {} {}'.format(obs,mod,exp)] = det_finder(data[obs][mod][exp][c])

    if exp_list == ['hist-noLu', 'lu']:
        
        cmap_whole = plt.cm.get_cmap('PRGn')
        hnolu_det = cmap_whole(0.40)
        hnolu_att = cmap_whole(0.20)
        lu_det = cmap_whole(0.65)
        lu_att = cmap_whole(0.85)     
        color_mapping = {}
        color_mapping['lu'] = {1:lu_det,2:lu_att,3:'lightgrey'}
        color_mapping['hist-noLu'] = {1:hnolu_det,2:hnolu_att,3:'lightgrey'}
        
    elif exp_list == ['historical','hist-noLu']:
        
        cmap_whole = plt.cm.get_cmap('BrBG')
        hist_det = cmap_whole(0.40)
        hist_att = cmap_whole(0.20)
        hnolu_det = cmap_whole(0.65)
        hnolu_att = cmap_whole(0.85)     
        color_mapping = {}
        color_mapping['historical'] = {1:hist_det,2:hist_att,3:'lightgrey'}
        color_mapping['hist-noLu'] = {1:hnolu_det,2:hnolu_att,3:'lightgrey'}

    for obs in obs_types:
        f, axes = plt.subplots(nrows=len(models),ncols=len(exp_list),figsize=(8,10))
        j = 0
        l = 0
        for row,mod in zip(axes,models):
            i = 0
            for ax,exp in zip(row,exp_list):
                ar6_continents.plot(ax=ax,
                                    color=ar6_continents['{}-grid {} {}'.format(obs,mod,exp)].map(color_mapping[exp]),
                                    edgecolor='black',
                                    linewidth=0.3)
                gpd_continents.boundary.plot(ax=ax,
                                             edgecolor='black',
                                             linewidth=0.3)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title(letters[l],
                             loc='left',
                             fontweight='bold',
                             fontsize=10)
                if j == 0:
                    ax.set_title(exp,
                                 loc='center',
                                 fontweight='bold',
                                 fontsize=10)
                if i == 0:
                    ax.text(-0.07, 0.55, 
                            mod, 
                            va='bottom', 
                            ha='center',# # create legend with patche for hsitnolu and lu det/att levels
                            fontweight='bold',
                            rotation='vertical', 
                            rotation_mode='anchor',
                            transform=ax.transAxes)
                i += 1
                l += 1
            j += 1
            
        if flag_svplt == 0:
            pass
        elif flag_svplt == 1:
            
            if exp_list == ['hist-noLu', 'lu']:
                
                f.savefig(outDIR+'/continental_attribution_2-factor_{}_{}-grid_{}-pi_{}-agg_{}-weight_{}.png'.format(obs,grid,pi,agg,weight,freq),bbox_inches='tight',dpi=500)
                
            elif exp_list == ['historical','hist-noLu']:
                
                f.savefig(outDIR+'/continental_attribution_1-factor_{}_{}-grid_{}-pi_{}-agg_{}-weight_{}.png'.format(obs,grid,pi,agg,weight,freq),bbox_inches='tight',dpi=500)
                

#%%============================================================================
                
def plot_scaling_map_combined(
    sfDIR,
    obs_types,
    pi,
    weight,
    models,
    exp_list,
    var_fin_cnt,
    var_fin_glb,
    grid,
    letters,
    freq,
    flag_svplt,
    outDIR):
    
    continents = {}
    continents['North America'] = [1,2,3,4,5,6,7]
    continents['South America'] = [9,10,11,12,13,14,15]
    continents['Europe'] = [16,17,18,19]
    continents['Asia'] = [28,29,30,31,32,33,34,35,37,38]
    continents['Africa'] = [21,22,23,24,25,26]
    continents['Australia'] = [39,40,41,42]    
    
    # x ticks temporal OF insets
    xticks_OF = np.arange(-0.5,2.5,0.5)
    xtick_labels_OF = [None, '0', None, '1', None, '2']        
    
    # y ticks temporal OF insets
    yticks_OF = np.arange(-0.5,2.5,0.5)
    ytick_labels_OF = [None, '0', None, '1', None, '2']    
    
    # inset_font = 18
    data_cnt = var_fin_cnt
    data_glb = var_fin_glb
    os.chdir(sfDIR)
    
    #test for merging ar6 into desired continent extents
    regions = gp.read_file('IPCC-WGI-reference-regions-v4.shp')
    gpd_continents = gp.read_file('IPCC_WGII_continental_regions.shp')
    gpd_continents = gpd_continents[(gpd_continents.Region != 'Antarctica')&(gpd_continents.Region != 'Small Islands')]
    regions = gp.clip(regions,gpd_continents)
    regions['keep'] = [0]*len(regions.Acronym)
    
    for c in continents.keys():
        for ar6 in continents[c]:
            regions.at[ar6,'Continent'] = c
            regions.at[ar6,'keep'] = 1
    
    regions = regions[regions.keep!=0]  
    regions = regions.drop(columns='keep')
    ar6_continents = regions.dissolve(by='Continent')

    for obs in obs_types:
        for mod in models:
            for exp in exp_list:
                ar6_continents['{}-grid {} {}'.format(obs,mod,exp)] = [1] * len(ar6_continents.Acronym)

    for obs in obs_types:
        for mod in models:
            for exp in exp_list:
                for c in continents.keys():
                    ar6_continents.loc[c,'{}-grid {} {}'.format(obs,mod,exp)] = det_finder(data_cnt[obs][mod][exp][c])

    if exp_list == ['hist-noLu', 'lu']:
        
        cmap_whole = plt.cm.get_cmap('PRGn')
        hnolu_det = cmap_whole(0.40)
        hnolu_att = cmap_whole(0.20)
        lu_det = cmap_whole(0.65)
        lu_att = cmap_whole(0.85)     
        color_mapping = {}
        color_mapping['lu'] = {1:lu_det,2:lu_att,3:'lightgrey'}
        color_mapping['hist-noLu'] = {1:hnolu_det,2:hnolu_att,3:'lightgrey'}
        
        cols={}
        median_cols={}
        cols['hist-noLu'] = cmap_whole(0.25)
        cols['lu'] = cmap_whole(0.75)
        median_cols['hist-noLu'] = cmap_whole(0.05)
        median_cols['lu'] = cmap_whole(0.95)        
        
    elif exp_list == ['historical','hist-noLu']:
        
        cmap_whole = plt.cm.get_cmap('BrBG')
        hist_det = cmap_whole(0.40)
        hist_att = cmap_whole(0.20)
        hnolu_det = cmap_whole(0.65)
        hnolu_att = cmap_whole(0.85)     
        color_mapping = {}
        color_mapping['historical'] = {1:hist_det,2:hist_att,3:'lightgrey'}
        color_mapping['hist-noLu'] = {1:hnolu_det,2:hnolu_att,3:'lightgrey'}
        
        cols={}
        median_cols={}
        cols['historical'] = cmap_whole(0.25)
        cols['hist-noLu'] = cmap_whole(0.75)
        median_cols['historical'] = cmap_whole(0.05)
        median_cols['hist-noLu'] = cmap_whole(0.95)          
        
    # OF data prep
    b = {}
    b_inf = {}
    b_sup = {}
    p = {}
    err = {}

    for obs in obs_types:
        
        # OF data prep
        b[obs] = {}
        b_inf[obs] = {}
        b_sup[obs] = {}
        p[obs] = {}
        err[obs] = {}    
        
        for mod in models:
            b[obs][mod] = {}
            b_inf[obs][mod] = {}
            b_sup[obs][mod] = {}
            p[obs][mod] = {}
            err[obs][mod] = {}
            for exp in exp_list:
                b[obs][mod][exp],\
                b_inf[obs][mod][exp],\
                b_sup[obs][mod][exp],\
                p[obs][mod][exp] = scale_take(data_glb[obs][mod][exp])
                err[obs][mod][exp] = np.stack([[b_inf[obs][mod][exp]],
                                              [b_sup[obs][mod][exp]]],
                                              axis=0)            
        
        f, axes = plt.subplots(
            nrows=len(models),
            ncols=len(exp_list)+1,
            gridspec_kw={
                'width_ratios':[2,2,1],
                'height_ratios':[1,1,1,1,1]},
            
            figsize=(12,10))
        
        j = 0
        
        for row,mod in zip(axes[:,:2],models):
            i = 0
            for ax,exp in zip(row,exp_list):
                ar6_continents.plot(ax=ax,
                                    color=ar6_continents['{}-grid {} {}'.format(obs,mod,exp)].map(color_mapping[exp]),
                                    edgecolor='black',
                                    linewidth=0.3)
                gpd_continents.boundary.plot(ax=ax,
                                             edgecolor='black',
                                             linewidth=0.3)
                ax.set_yticks([])
                ax.set_xticks([])
                if j == 0:
                    if exp == 'hist-noLu':
                        exp_label = 'HISTNL'
                    elif exp == 'lu':
                        exp_label = 'LU'
                    elif exp == 'historical':
                        exp_label = 'HIST'
                    ax.set_title(
                        exp_label,
                        loc='center',
                        fontweight='bold',
                        fontsize=10)
                if i == 0:
                    ax.text(-0.07, 0.55, 
                            mod, 
                            va='bottom', 
                            ha='center',# # create legend with patche for hsitnolu and lu det/att levels
                            fontweight='bold',
                            rotation='vertical', 
                            rotation_mode='anchor',
                            transform=ax.transAxes)
                i += 1
            j += 1
        
        g = 0
        for ax,mod in zip(axes[:,2],models):
            
            if exp_list == ['hist-noLu','lu']:

                ax.errorbar(x=b[obs][mod]['hist-noLu'],
                            y=b[obs][mod]['lu'],
                            xerr=err[obs][mod]['hist-noLu'],
                            fmt='o',
                            markersize=3,
                            ecolor=cols['hist-noLu'],
                            markerfacecolor=median_cols['hist-noLu'],
                            mec=cols['lu'],
                            capsize=5,
                            elinewidth=4,
                            markeredgewidth=1)
                
                ax.errorbar(x=b[obs][mod]['hist-noLu'],
                            y=b[obs][mod]['lu'],
                            yerr=err[obs][mod]['lu'],
                            fmt='o',
                            markersize=3,
                            ecolor=cols['lu'],
                            markerfacecolor=median_cols['lu'],
                            mec=cols['lu'],
                            capsize=5,
                            elinewidth=4,
                            markeredgewidth=1)
                
            elif exp_list == ['historical','hist-noLu']:
                
                ax.errorbar(x=b[obs][mod]['hist-noLu'],
                            y=b[obs][mod]['historical'],
                            xerr=err[obs][mod]['hist-noLu'],
                            fmt='o',
                            markersize=3,
                            ecolor=cols['hist-noLu'],
                            markerfacecolor=median_cols['hist-noLu'],
                            mec=cols['historical'],
                            capsize=5,
                            elinewidth=4,
                            markeredgewidth=1)
                
                ax.errorbar(x=b[obs][mod]['hist-noLu'],
                            y=b[obs][mod]['historical'],
                            yerr=err[obs][mod]['historical'],
                            fmt='o',
                            markersize=3,
                            ecolor=cols['historical'],
                            markerfacecolor=median_cols['historical'],
                            mec=cols['historical'],
                            capsize=5,
                            elinewidth=4,
                            markeredgewidth=1)            
            
            ax.set_xticks([])
            ax.set_xlim((-1,2))
            ax.set_ylim((-1,2))
            
            ax.hlines(y=1,
                    xmin=-1,
                    xmax=2,
                    colors='k',
                    linestyle='dashed',
                    linewidth=1)
            ax.hlines(y=0,
                    xmin=-1,
                    xmax=2,
                    colors='k',
                    linestyle='solid',
                    linewidth=0.25)
            
            ax.vlines(x=1,
                    ymin=-1,
                    ymax=2,
                    colors='k',
                    linestyle='dashed',
                    linewidth=1)
            ax.vlines(x=0,
                    ymin=-1,
                    ymax=2,
                    colors='k',
                    linestyle='solid',
                    linewidth=0.25)
            
            ax.tick_params(axis="x",
                        direction="in", 
                        left="off",
                        labelleft="on")
            ax.tick_params(axis="y",
                        direction="in")
            
            ax.yaxis.tick_right()
            ax.set_yticks(yticks_OF)
            ax.set_yticklabels([None, '0', None, '1', None, '2']) 
            
            if ax == axes.flatten()[-1]:
                
                ax.set_xlabel(r'$\beta_{HISTNL}$',
                            fontsize=12,
                            color=median_cols['hist-noLu'],
                            fontweight='bold')
                ax.set_xticks(xticks_OF)
                ax.set_xticklabels([None, '0', None, '1', None, '2'])
                
            if exp_list == ['hist-noLu','lu']:
            
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(r'$\beta_{LU}$',
                            fontsize=12,
                            color=median_cols['lu'],
                            fontweight='bold')
                
            elif exp_list == ['historical','hist-noLu']:
                
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(r'$\beta_{HIST}$',
                            fontsize=12,
                            color=median_cols['historical'],
                            fontweight='bold')
                
            g += 1
                   
        
        for l,ax in enumerate(axes.flatten()):
            ax.set_title(
                letters[l],
                loc='left',
                fontweight='bold',
                fontsize=10)
            
        if flag_svplt == 0:
            pass
        elif flag_svplt == 1:
            
            if exp_list == ['hist-noLu', 'lu']:
                
                f.savefig(outDIR+'/combined_attribution_2-factor_{}_{}-grid_{}-pi_{}-weight_{}.png'.format(obs,grid,pi,weight,freq),bbox_inches='tight',dpi=500)
                
            elif exp_list == ['historical','hist-noLu']:
                
                f.savefig(outDIR+'/combined_attribution_1-factor_{}_{}-grid_{}-pi_{}-weight_{}.png'.format(obs,grid,pi,weight,freq),bbox_inches='tight',dpi=500)
# %%
