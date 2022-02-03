#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:52:49 2020

@author: luke
"""


# =============================================================================
# SUMMARY
# =============================================================================


# This script uses 5-year mean, weighted, model-grid all-pi optioned input data
# from 'da_*.py' for HISTNL + LU at continental and global levels to test for
# building Hannart's maximum likelihood estimate


#%%============================================================================
# import
# =============================================================================

# import sys
import os
import numpy as np
import pickle as pk
import scipy.linalg as spla
from funcs import *

#%%============================================================================
# path
#==============================================================================

curDIR = r'C:/Users/lgrant/Documents/repos/wtls_da'
os.chdir(curDIR)

# data input directories
sfDIR = os.path.join(curDIR, 'shapefiles')
pklDIR = os.path.join(curDIR, 'pickle')
outDIR = os.path.join(curDIR, 'figures')


#%%============================================================================
# options - analysis
#==============================================================================

# adjust these flag settings for analysis choices only change '<< SELECT >>' lines

# << SELECT >>
flag_svplt=0      # 0: do not save plot
                  # 1: save plot in picDIR

# << SELECT >>
flag_analysis=0   # 0: d&a on global scale (all chosen ar6 regions)
                  # 1: d&a on continental scale (scaling factor per continent; continent represented by AR6 weighted means)
                  # 2: d&a on ar6 scale (scaling factor per ar6 region)
                  # 3: d&a already proc'd for global + continental, read in pickle and plot together
                  
# << SELECT >>
flag_data_agg=1   # 0: global d&a (via flag_analysis) w/ ar6 scale input points (want this for continental scale via flag_analysis)
                  # 1: global d&a w/ continental scale input points             
                  
# << SELECT >>
flag_lulcc=0      # 0: forest loss
                  # 1: crop expansion
                  
# << SELECT >>
flag_grid=0       # 0: model grid resolution (decided on always using this; many options don't work otherwise)
                  # 1: uniform obs grid resolution
                  
# << SELECT >>
flag_pi=1         # 0: only use pi from chosen models
                  # 1: use all available pi
                  
# << SELECT >>
flag_factor=0     # 0: 2-factor -> hist-noLu and lu
                  # 1: 1-factor -> historical
                  # 2: 1-factor -> hist-noLu
                        
# << SELECT >>
flag_weight=1           # 0: no weights on spatial means (not per pixel, which is automatic, but for overall area diff across continents when flag_data_agg == 1)
                        # 1: weights (asia weight of 1, australia weight of ~0.18; same for ar6 within continents)    
                        
# << SELECT >>
flag_lu_technique=1     # 0: lu as mean of individual (historical - hist-nolu)
                        # 1: lu as mean(historical) - mean(hist-nolu)

# << SELECT >>
flag_y1=1         # 0: 1915
                  # 1: 1965

# << SELECT >>
flag_len=0        # 0: 50
                  # 1: 100

# << SELECT >>
flag_resample=0    # 0: 5 year block means
                   # 1: 10 year block means
                   # 2: 2, 25 yr block means which are subtracted to collapse time dim

# << SELECT >>
flag_var=0   # 0: tasmax

# << SELECT >> 
flag_bs=1         # 0: No bootstrapping of covariance matrix build
                  # 1: 50 (e.g. 50 reps of ROF, each with shuffled pichunks for Cov_matrices)
                  # 2: 100
                  # 3: 500
                  # 4: 1000

# << SELECT >>  # confidence intervals on scaling factors
ci_bnds = 0.95    # means  beta - 0.95 cummulative quantile and + 0.95 cummulative quantile, 
  
# << SELECT >> 
flag_reg=0        # 0: OLS
                  # 1: TLS
                  # 2: WTLS (Hannart)

# << SELECT >>
flag_constest=3   # 0: OLS_AT99 
                  # 1: OLS_Corr
                  # 2: AS03 (TLS only)
                  # 3: MC (TLS only)

# << SELECT >> # confidence internval calculation in case that TLS regression chosen 
flag_ci_tls=0     # 0: AS03
                  # 1: ODP

# << SELECT >>
trunc=0

analyses = ['global',
            'continental',
            'ar6',
            'combined']
agg_opts = ['ar6',
            'continental']
grids = ['model',
         'obs']
pi_opts = ['model',
           'allpi']
factors = [['hist-noLu','lu'],
           ['historical'],
           ['hist-noLu']]
obs_types = ['cru',
             'berkley_earth']
weight_opts = ['no_weights',
               'weights']
start_years = [1915,
               1965]
lengths = [50,
           100]
resample=['5Y',
          '10Y',
          '25Y']
variables = ['tasmax']
regressions = ['OLS',
               'TLS']
consistency_tests = ['OLS_AT99',
                     'OLS_Corr',
                     'AS03',
                     'MC']
tls_cis = ['AS03',
           'ODP']
shuffle_opts = ['no', 
                'yes']
bootstrap_reps = [0,50,100,500,1000]

analysis = analyses[flag_analysis]
agg = agg_opts[flag_data_agg]
grid = grids[flag_grid]
pi = pi_opts[flag_pi]
exp_list = factors[flag_factor]
weight = weight_opts[flag_weight]
y1 = start_years[flag_y1]
length = lengths[flag_len]
freq = resample[flag_resample]
var = variables[flag_var]
bs_reps = bootstrap_reps[flag_bs]
reg = regressions[flag_reg]
cons_test = consistency_tests[flag_constest]
formule_ic_tls = tls_cis[flag_ci_tls]

# temporal extent of analysis data
strt_dt = str(y1) + '01'
y2 = y1+length-1
end_dt = str(y2) + '12'
t_ext = strt_dt+'-'+end_dt

models = ['CanESM5',
          'CNRM-ESM2-1',
          'IPSL-CM6A-LR',
          'UKESM1-0-LL']

exps = ['historical',
        'hist-noLu',
        'lu']

if (analysis == 'global' and agg == 'ar6') or analysis == 'continental' or analysis == 'ar6':
    
    continents = {}
    continents['North America'] = [1,2,3,4,5,6,7]
    continents['South America'] = [9,10,11,12,13,14,15]
    continents['Europe'] = [16,17,18,19]
    continents['Asia'] = [28,29,30,31,32,33,34,35,37,38]
    continents['Africa'] = [21,22,23,24,25,26]
    continents['Australia'] = [39,40,41,42]

    continent_names = []
    for c in continents.keys():
        continent_names.append(c)

    labels = {}
    labels['North America'] = ['WNA','CNA','ENA','SCA']
    labels['South America'] = ['NWS','NSA','NES','SAM','SWS','SES']
    labels['Europe'] = ['NEU','WCE','EEU','MED']
    labels['Asia'] = ['WSB','ESB','TIB','EAS','SAS','SEA']
    labels['Africa'] = ['WAF','CAF','NEAF','SEAF','ESAF']

    ns = 0
    for c in continents.keys():
        for i in continents[c]:
            ns += 1
            
elif analysis == 'global' and agg == 'continental':
    
    continents = {}
    continents['North America'] = 7
    continents['South America'] = 4
    continents['Europe'] = 1
    continents['Asia'] = 6
    continents['Africa'] = 3
    continents['Australia'] = 5

    continent_names = []
    for c in continents.keys():
        continent_names.append(c)

    ns = 0
    for c in continents.keys():
        ns += 1    
        
letters = ['a', 'b', 'c',
           'd', 'e', 'f',
           'g', 'h', 'i',
           'j', 'k', 'l',
           'm', 'n', 'o',
           'p', 'q', 'r',
           's', 't', 'u',
           'v', 'w', 'x',
           'y', 'z']


#==============================================================================
# get data 
#==============================================================================

#%%============================================================================

# mod ensembles
os.chdir(pklDIR)
pkl_file = open('omega_inputs_{}-flagfactor_{}-grid_{}-pi_{}-agg_{}-weight_{}_{}_{}.pkl'.format(
    flag_factor,grid,pi,agg,weight,freq,t_ext,reg),'rb')
omega_samples = pk.load(pkl_file)
pkl_file.close()

# get ensemble metadata
nx = {}
mmmh = 0
mmmhnl = 0
for mod in models:
    nx[mod] = np.array([[omega_samples[mod]['historical'].shape[0],
                         omega_samples[mod]['hist-noLu'].shape[0]]])
    mmmh += omega_samples[mod]['historical'].shape[0]
    mmmhnl += omega_samples[mod]['hist-noLu'].shape[0]
nx['mmm'] = np.array([[mmmh,mmmhnl]])
nt = 10

#%%============================================================================

# mod fingerprint
pkl_file = open('mod_inputs_{}-flagfactor_{}-grid_{}-pi_{}-agg_{}-weight_{}_{}_{}.pkl'.format(
    flag_factor,grid,pi,agg,weight,freq,t_ext,reg),'rb')
dictionary = pk.load(pkl_file)
pkl_file.close()
fp = dictionary['global']
fp_continental = dictionary['continental']


#%%============================================================================

# pi data
pkl_file = open('pic_inputs_{}-grid_{}-pi_{}-agg_{}-weight_{}_{}_{}.pkl'.format(
    grid,pi,agg,weight,freq,t_ext,reg),'rb')
dictionary = pk.load(pkl_file)
pkl_file.close()
ctl_data = dictionary['global']
ctl_data_continental = dictionary['continental']


#%%============================================================================

# obs data
pkl_file = open('obs_inputs_{}-grid_{}-pi_{}-agg_{}-weight_{}_{}_{}.pkl'.format(
    grid,pi,agg,weight,freq,t_ext,reg),'rb')
dictionary = pk.load(pkl_file)
pkl_file.close()
obs_data = dictionary['global']
obs_data_continental = dictionary['continental']


#%%============================================================================
# detection & attribution 
#==============================================================================

# optimal fingerprinting
# os.chdir(curDIR)
# from sr_of import *
# var_sfs,\
# var_ctlruns,\
# proj,\
# U,\
# yc,\
# Z1c,\
# Z2c,\
# Xc,\
# Cf1,\
# Ft,\
# beta_hat,\
# var_fin,\
# models = of_subroutine(models,
#                        nx,
#                        analysis,
#                        exp_list,
#                        obs_types,
#                        pi,
#                        obs_data,
#                        obs_data_continental,
#                        fp,
#                        fp_continental,
#                        omega_samples,
#                        ctl_data,
#                        ctl_data_continental,
#                        bs_reps,
#                        ns,
#                        nt,
#                        reg,
#                        cons_test,
#                        formule_ic_tls,
#                        trunc,
#                        ci_bnds,
#                        continents)


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

obs = 'berkley_earth'
mod = 'CNRM-ESM2-1'
gamma = 0.05 # precision level
om_bs=100
nx = nx[mod]
    
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
    
# input vectors
y = np.matrix(y).T
X = np.matrix(X).T
nb_runs_x = np.matrix(nx)
nbts = nt # number t-steps
n_spa = ns # spatial dim
n_st = n_spa * nbts # spatio_temporal dimension (ie dimension of y)
I = X.shape[1] # number of different forcings

# spatio-temporal dimension after reduction
if nbts > 1:
    n_red = n_st - n_spa
    U = projfullrank(nbts, n_spa)
elif nbts == 1: # case where I regress with collapsed time dimension and each point is a signal
    n_red = n_st - nbts # therefore treating ns as nt
    U = projfullrank(n_spa, nbts)

# project all input data 
yc = np.dot(U, y)
Xc = np.dot(U, X)
proj = np.identity(X.shape[1]) # already took factorial LU, so no need for proj to use dot to decipher histnl + lu from hist, histnl

# model based uncertainty estimate                   
cov_omega = {}
cov_omega_sqrt = {}
boots = {}
for exp in exp_list:
    if exp == "historical":
        runs = nb_runs_x[0,0]
    elif exp == "hist-noLu" or "lu":
        runs = nb_runs_x[0,1]
    omega = omega_samples[mod][exp]
    omega = np.transpose(np.matrix(omega))
    omega = np.dot(U, omega)

    for i in np.arange(om_bs):
        boot = np.empty_like(omega)
        for r in range(runs):
            index = np.random.randint(0,omega.shape[1])
            boot[:,r] = np.array(omega[:,index])
        if i == 0:
            bs_smp = np.mean(boot,axis=1)
        else:
            bs_smp = np.hstack((bs_smp,np.mean(boot,axis=1)))
    
    omega = bs_smp
    cov_omega[exp] = np.dot(omega,omega.T) / omega.shape[1]
    cov_omega_sqrt[exp] = spla.fractional_matrix_power(cov_omega[exp], -0.5)


# pic based IV estimates
for r in np.arange(0,bs_reps):
    
    # shuffle rows of ctl
    ctl = np.take(ctl,
                    np.random.permutation(ctl.shape[0]),
                    axis=0)
    
    z = np.transpose(np.matrix(ctl))
    
    # create even sample numbered z1 and z2 samples of IV
    nb_runs_ctl = np.shape(z)[1]
    half_1_end = int(np.floor(nb_runs_ctl / 2))
    z1 = z[:,:half_1_end]
    if nb_runs_ctl % 2 == 0:
        z2 = z[:,half_1_end:]
    else:
        z2 = z[:,half_1_end+1:]    
        
    ## Regularised covariance matrix
    z1c = np.dot(U, z1)
    z2c = np.dot(U, z2)
    # Cf = regC(Z1c.T) # will test hannart algo later with this regularized estimate

    # normal covariance matrices from split samples
    cov_z1 = np.dot(z1c,z1c.T) / z1c.shape[1]
    # cov_z1_inv = np.real(spla.inv(cov_z1))
    cov_z2 = np.dot(z2c,z2c.T) / z2c.shape[1]
    
    # Hannart scheme:
    x_t = Xc # initial x
    y_t = yc # y
    beta_t = beta_calc( # initial beta
        x_t,
        cov_z1,
        y_t,
    )
    gamma_i = 1
    while gamma_i >= gamma:
        for i in range(x_t.shape[1]): # step 1; "s1"
            if i == 0:
                om = cov_omega_sqrt['hist-noLu'] # choosing sqrt is for conforming to appendix D; can switch for normal and avoid these steps
                j = 1
            else:
                om = cov_omega_sqrt['lu']
                j = 0
            cov_prod = np.dot(np.dot(om,cov_z1),om)
            w,v = spla.eigh(cov_prod)
            delta = np.diag(w)
            # m1 = delta + beta_t[0,i]**2 * np.identity(len(w))
            m1 = np.real(spla.inv(np.matrix((delta + beta_t[0,i]**2 * np.identity(len(w))))))
            y_bar = y_t - beta_t[0,j]*x_t[:,i]
            m2 = beta_t[0,i]*y_bar + np.dot(delta,x_t[:,i])
            x_t[:,i] = np.dot(m1,m2)
            
            x_t =
            
        
        gamma_i = ... # redefine gamma score for while loop
     
    
    
    

           
#%%============================================================================
# plotting scaling factors
#==============================================================================    

if analysis == 'global':
    
    plot_scaling_global(models,
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
                        outDIR)

elif analysis == 'continental':
    
    
    plot_scaling_map_continental(sfDIR,
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
                                 flag_svplt,
                                 outDIR)

elif analysis == 'ar6':
    
    plot_scaling_map_ar6(sfDIR,
                         obs_types,
                         pi,
                         models,
                         exp_list,
                         continents,
                         var_fin,
                         grid,
                         letters,
                         flag_svplt,
                         outDIR)              
    
elif analysis == 'combined':
    
    os.chdir(curDIR)    
    models.append('mmm')
    if len(exp_list) == 2:
        
        os.chdir(pklDIR)
        
        pkl_file_cnt = open('var_fin_2-factor_{}-grid_{}_{}-pi_{}-agg_{}-weight_{}-bsreps_{}_{}.pkl'.format(grid,'continental',pi,'ar6',weight,bs_reps,freq,t_ext),'rb')
        var_fin_cnt = pk.load(pkl_file_cnt)
        pkl_file_cnt.close()
        
        pkl_file_glb = open('var_fin_2-factor_{}-grid_{}_{}-pi_{}-agg_{}-weight_{}-bsreps_{}_{}.pkl'.format(grid,'global',pi,'continental',weight,bs_reps,freq,t_ext),'rb')
        var_fin_glb = pk.load(pkl_file_glb) 
        pkl_file_glb.close()       

    elif len(exp_list) == 1:
        
        start_exp = deepcopy(exp_list[0])
        if start_exp == 'historical':
            second_exp = 'hist-noLu'
        elif start_exp == 'hist-noLu':
            second_exp = 'historical'
            
        os.chdir(pklDIR)
        pkl_file_cnt = open('var_fin_1-factor_{}_{}-grid_{}_{}-pi_{}-agg_{}-weight_{}-bsreps_{}_{}.pkl'.format(start_exp,grid,'continental',pi,'ar6',weight,bs_reps,freq,t_ext),'rb')
        var_fin_cnt = pk.load(pkl_file_cnt)
        pkl_file_cnt.close()
        
        pkl_file_cnt_2 = open('var_fin_1-factor_{}_{}-grid_{}_{}-pi_{}-agg_{}-weight_{}-bsreps_{}_{}.pkl'.format(second_exp,grid,'continental',pi,'ar6',weight,bs_reps,freq,t_ext),'rb')
        var_fin_cnt_2 = pk.load(pkl_file_cnt_2)
        pkl_file_cnt_2.close()        
        
        pkl_file_glb = open('var_fin_1-factor_{}_{}-grid_{}_{}-pi_{}-agg_{}-weight_{}-bsreps_{}_{}.pkl'.format(start_exp,grid,'global',pi,'continental',weight,bs_reps,freq,t_ext),'rb')
        var_fin_glb = pk.load(pkl_file_glb)
        pkl_file_glb.close()
        
        pkl_file_glb_2 = open('var_fin_1-factor_{}_{}-grid_{}_{}-pi_{}-agg_{}-weight_{}-bsreps_{}_{}.pkl'.format(second_exp,grid,'global',pi,'continental',weight,bs_reps,freq,t_ext),'rb')
        var_fin_glb_2 = pk.load(pkl_file_glb_2)
        pkl_file_glb_2.close()        
        
        for obs in obs_types:
            for mod in models:
                var_fin_cnt[obs][mod][second_exp] = var_fin_cnt_2[obs][mod].pop(second_exp)        
        
        for obs in obs_types:
            for mod in models:
                var_fin_glb[obs][mod][second_exp] = var_fin_glb_2[obs][mod].pop(second_exp)
                
        exp_list = ['historical', 'hist-noLu']    
    
    plot_scaling_map_combined(
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
        outDIR
        )    
         
    
    

# %%
