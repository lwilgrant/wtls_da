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
from re import L
import numpy as np
import pickle as pk
import scipy.linalg as spla
import math
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
mod = 'IPSL-CM6A-LR'
cov1_comp = 'regC' # for ledoit and wolf regularized estimate or 'C' for normal covariance
gamma = 0.3 # precision level
om_bs = 100 # boot straps for omega as model covariance
om_comp = 'H2014' # or 'G2020' or 'H2014_modtr' or 'H2014tr'; 
    # in G2020, only bootstrapped model cov is used for omega; in H2014, bootstrapped model cov is added to PIC cov
    # in H2014_modtr, do H2014 but instead compute trace of mod cov and add tr(mod_cov)*identity to pic_cov/runs
    # in H2014, full hannart routine; no use of covariance estimates from fingerprints to get delta
s1_comp = 's1' # computation type for step 1 of iteration; "s1" for original, and "d1" for appendix
n_runs = nx[mod]
    
# var_sfs[obs] = {}
# bhi[obs] = {}
# b[obs] = {}
# blow[obs] = {}
# pval[obs] = {}
# var_fin[obs] = {}
# var_ctlruns[obs] = {}
# U[obs] = {}
# yc[obs] = {}
# Z1c[obs] = {}
# Z2c[obs] = {}
# Xc[obs] = {}
# Cf1[obs] = {}
# Ft[obs] = {}
# beta_hat[obs] = {}
        
# bhi[obs][mod] = {}
# b[obs][mod] = {}
# blow[obs][mod] = {}
# pval[obs][mod] = {}
# var_fin[obs][mod] = {}
            
# for exp in exp_list:
    
#     bhi[obs][mod][exp] = []
#     b[obs][mod][exp] = []
#     blow[obs][mod][exp] = []
#     pval[obs][mod][exp] = []
    
y = obs_data[obs][mod]
X = fp[mod]
if pi == 'model':
    ctl = ctl_data[mod]
elif pi == 'allpi':
    ctl = ctl_data     
    
# input vectors
y = np.matrix(y).T
X = np.matrix(X).T
nb_runs_x = np.matrix(n_runs)
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
if om_comp != 'H2014':
    cov_omega = omega_calc(
        exp_list,
        nb_runs_x,
        omega_samples,
        mod,
        U,
        om_bs
    )
else:
    cov_omega = {}

# pic based IV estimates
# for r in np.arange(0,bs_reps):
    
    # # shuffle rows of ctl
    # ctl = np.take(ctl,
    #                 np.random.permutation(ctl.shape[0]),
    #                 axis=0)
    
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
if cov1_comp == 'regC':
    cov_z1 = regC(z1c.T)
if om_comp != 'H2014':
    for exp in exp_list:
        if om_comp == 'H2014_modtr':
            cov_omega[exp] = np.trace(cov_omega[exp]) * np.eye(cov_omega[exp].shape[0]) / cov_omega[exp].shape[0] 
        cov_omega[exp] = cov_z1 / nb_runs_x[0,1] + cov_omega[exp] # nb_runs_x only configured for 2-way with histnolu and lu
elif om_comp == 'H2014':
    for exp in exp_list:
        cov_omega[exp] = cov_z1 / nb_runs_x[0,1] + np.trace(cov_z1) * np.eye(cov_z1.shape[0]) / cov_z1.shape[0]
cov_z2 = np.dot(z2c,z2c.T) / z2c.shape[1]

#%%============================================================================
# Hannart scheme - virtual
#==============================================================================

# input data; "knowns"
n1 = z1c.shape[1]
n2 = z2c.shape[1]
x_start = deepcopy(Xc) # "reference" x
k = x_start.shape[1] 
beta_1 = np.ones((k,1)) # "true" beta
sigma12 = spla.sqrtm(cov_z1)
n = sigma12.shape[0]
runs = np.matrix([[omega_samples[mod]['hist-noLu'].shape[0],omega_samples[mod]['lu'].shape[0]]]) # fudging this here for now even tho inconsistent with above

# virtual data; estimated 
yv_t = np.dot(x_start, beta_1) # "true" y
yv_n = yv_t + np.dot(sigma12,np.random.normal(0,1,size=(n,1))) # "observed"/noised y
xv = x_start + np.dot(sigma12, np.random.normal(0, 1, size=(n,k)) / (np.ones(yv_t.shape) * np.sqrt(runs))) # "observed"/noised x
xv_n = np.multiply((np.dot(np.ones(yv_t.shape), np.sqrt(runs))), xv) # normalizing variance in x DONT UNDERSTAND THIS
sigma_v = np.dot(sigma12, np.random.normal(0, 1, size=(n, n1))) # virtual Z samples
if cov1_comp == 'regC':
    sigma_v = regC(sigma_v.T)
else:
    sigma_v =  np.dot(sigma_v,sigma_v.T) / sigma_v.shape[1]
    
beta_list = []
l_list = []
gamma_list = []

# rename to fit algo
x_t = deepcopy(xv_n)
y_t = deepcopy(yv_n)
cov_z1 = deepcopy(sigma_v)

# hannart algo
beta_0 = beta_calc( # initial beta
    x_t,
    cov_z1,
    y_t,
)
gamma_i = 1
it = 0
while np.all(gamma_i) >= gamma: 
    for i,exp in zip(range(x_t.shape[1]),exp_list): # step 1; "s1"
        if i == 0:
            om = cov_omega[exp] # choosing sqrt is for conforming to appendix D; can switch for normal and avoid these steps
            j = 1
        else:
            om = cov_omega[exp]
            j = 0
        # y_bar_i = y_t - beta_t[0,j]*x_t[:,i]
        x_t_i = x_t[:,i]
        if it == 0:
            y_bar_i = y_t - beta_0[0,j]*x_t[:,i] # check the output of this beta*x order against that inside the log likelihood; does either screw order up?
            beta_t_i = beta_0[0,i] # comment line above answer is no because this is scalar mult and other is dot
        else:
            y_bar_i = y_t - beta_t[0,j]*x_t[:,i]
            beta_t_i = beta_t[0,i]
        x_t[:,i] = it_step1( # it_step1
            om,
            beta_t_i,
            cov_z1,
            y_bar_i,
            x_t_i,
            s1_comp
            )
    if it == 0:
        beta_t = beta_calc( # it_step2
            x_t,
            cov_z1,
            y_t,
            )
        gamma_i = np.abs((beta_t - beta_0)) / np.abs(beta_0) # target gamma
    else:
        beta_t1 = beta_calc( # it_step2
            x_t,
            cov_z1,
            y_t,
            )
        gamma_i = np.abs((beta_t1 - beta_t)) / np.abs(beta_t) # target gamma
        graph_beta,graph_l = log_likelihood(
            y_t,
            x_t,
            beta_t1,
            cov_z1,
            Xc,
            cov_omega,
            exp_list
        )
        beta_list.append(graph_beta)
        l_list.append(float(graph_l))
        gamma_list.append(gamma_i)
        # l_list.append(math.exp(float(graph_l)))
        
    it += 1

# PRESERVE BELOW FOR RETURNING TO ACTUAL DATA
# x_t = deepcopy(Xc) # initial x
# y_t = deepcopy(yc) # y
# beta_0 = beta_calc( # initial beta
#     x_t,
#     cov_z1,
#     y_t,
# )

# beta_list = []
# l_list = []

# gamma_i = 1
# it = 0
# while np.all(gamma_i) >= gamma: 
#     for i,exp in zip(range(x_t.shape[1]),exp_list): # step 1; "s1"
#         if i == 0:
#             om = cov_omega[exp] # choosing sqrt is for conforming to appendix D; can switch for normal and avoid these steps
#             j = 1
#         else:
#             om = cov_omega[exp]
#             j = 0
#         # y_bar_i = y_t - beta_t[0,j]*x_t[:,i]
#         x_t_i = x_t[:,i]
#         if it == 0:
#             y_bar_i = y_t - beta_0[0,j]*x_t[:,i]
#             beta_t_i = beta_0[0,i]
#         else:
#             y_bar_i = y_t - beta_t[0,j]*x_t[:,i]
#             beta_t_i = beta_t[0,i]
#         x_t[:,i] = it_step1( # it_step1
#             om,
#             beta_t_i,
#             cov_z1,
#             y_bar_i,
#             x_t_i,
#             s1_comp
#             )
#     if it == 0:
#         beta_t = beta_calc( # it_step2
#             x_t,
#             cov_z1,
#             y_t,
#             )
#         gamma_i = np.abs((beta_t - beta_0)) / np.abs(beta_0) # target gamma
#     else:
#         beta_t1 = beta_calc( # it_step2
#             x_t,
#             cov_z1,
#             y_t,
#             )
#         gamma_i = np.abs((beta_t1 - beta_t)) / np.abs(beta_t) # target gamma
#         graph_beta,graph_l = log_likelihood(
#             y_t,
#             x_t,
#             beta_t1,
#             cov_z1,
#             Xc,
#             cov_omega,
#             exp_list
#         )
#         beta_list.append(graph_beta)
#         l_list.append(float(graph_l))
#         # l_list.append(math.exp(float(graph_l)))
        
#     it += 1
    
    
    

    
    
    
# for c in ['s1','d1']:
#     test[exp][c] = it_step1( # it_step1
#         om,
#         beta_t_i,
#         cov_z1,
#         y_bar_i,
#         x_t_i,
#         c,
#     )         
# rexpressing vars for testing in func it_step1 between s1 and d1            
# cov_mod = om
# bt = beta_t_i
# cov_pic = cov_z1
# y_bar_t = y_bar_i          
# x_t = x_t_i     
# np.linalg.cholesky(cov_omega['hist-noLu'])


            
        
    
     
    
    
    
