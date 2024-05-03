
# bran2020_stats.py
"""
Module Name: bran2020_stats.py
Description: This module provides statistical functions for analyzing BRAN2020 data.
Author: 
Date: 3 May 2024
"""

# Import statements
import xarray as xr
import numpy as np
import flox
import bottleneck
import numba
import numbagg


# Function definitions
def mean_monthclim_flox(ds,var_name,time_dim='time',method_str='cohorts',skipna_flag=False):
    """
    currently written for single variable datasets
    """
    if skipna_flag == False:
        mean = ds.groupby(time_dim+'.month').mean(dim=time_dim,engine='flox',method=method_str,skipna=False).rename({var_name:'mean_'+var_name})
        std = ds.groupby(time_dim+'.month').std(dim=time_dim,engine='flox',method=method_str,skipna=False).rename({var_name:'std_'+var_name})
    else:
        mean = ds.groupby(time_dim+'.month').mean(dim=time_dim,engine='numbagg').rename({var_name:'mean_'+var_name})
        std = ds.groupby(time_dim+'.month').std(dim=time_dim,engine='numbagg').rename({var_name:'std_'+var_name})
    max = ds.groupby(time_dim+'.month').max(dim=time_dim,engine='flox',method=method_str).rename({var_name:'max_'+var_name})
    min = ds.groupby(time_dim+'.month').min(dim=time_dim,engine='flox',method=method_str).rename({var_name:'min_'+var_name})
    stats_ds = xr.merge([mean,std,max,min])
    return stats_ds

def median_monthclim(ds,var_name,skipna_flag=False,time_dim='time'):
    """
    current practice is to provide a {'time':-1} chunked ds for median calculation
    """
    median_ds = ds.groupby(time_dim+'.month').median(skipna=skipna_flag,engine='flox').rename({var_name:'median_'+var_name})
    return median_ds

def quantile_monthclim(ds,var_name,skipna_flag=False,time_dim='time',q_list=[0.05,0.95]):
    """
    current practice is to provide a {'time':-1} chunked ds for quantile calculation
    """
    quant = ds.groupby(time_dim+'.month').quantile(q_list,skipna=skipna_flag,dim=time_dim,engine='flox').astype(np.float32)
    quant_ds = xr.merge([quant.isel(quantile=0).reset_coords(drop=True).rename({var_name:'quantile_05_'+var_name}),quant.isel(quantile=1).reset_coords(drop=True).rename({var_name:'quantile_95_'+var_name})])
    return quant_ds