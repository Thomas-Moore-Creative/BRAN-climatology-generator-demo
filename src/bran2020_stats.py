
# bran2020_stats.py
"""
Module Name: bran2020_stats.py
Description: This module provides statistical functions for analyzing BRAN2020 data.
Author: 
Date: 24 April 2024
"""

# Import statements
import xarray as xr
import numpy as np
from flox.xarray import xarray_reduce
import flox
import flox.xarray

# Function definitions
def mean_monthclim_flox(ds,var_name,time_dim='time',method='map-reduce',skipna=False):
    if skipna == False:
        mean = xarray_reduce(ds, time_dim+'.month', func="mean",method=method).rename({var_name:'mean_'+var_name})
        std = xarray_reduce(ds, time_dim+'.month', func="std",method=method).rename({var_name:'std_'+var_name})
    else:
        mean = xarray_reduce(ds, time_dim+'.month', func="nanmean",method=method).rename({var_name:'mean_'+var_name})
        std = xarray_reduce(ds, time_dim+'.month', func="nanstd",method=method).rename({var_name:'std_'+var_name})
    max = xarray_reduce(ds, time_dim+'.month', func="argmax",method=method).rename({var_name:'max_'+var_name})
    min = xarray_reduce(ds, time_dim+'.month', func="argmin",method=method).rename({var_name:'min_'+var_name})
    stats_ds = xr.merge([mean,std,max,min])
    return stats_ds

def median_monthclim(ds,var_name,skipna_flag=False,time_dim='time'):
    median_ds = ds.groupby(time_dim+'.month').median(skipna=skipna_flag).rename({var_name:'median_'+var_name})
    return median_ds

def quantile_monthclim(ds,var_name,skipna_flag=False,time_dim='time',q_list=[0.05,0.95]):
    quant = ds.groupby(time_dim+'.month').quantile(q_list,skipna=skipna_flag,dim=time_dim).astype(np.float32)
    quant_ds = xr.merge([quant.isel(quantile=0).reset_coords(drop=True).rename({var_name:'quantile_05_'+var_name}),quant.isel(quantile=1).reset_coords(drop=True).rename({var_name:'quantile_95_'+var_name})])
    return quant_ds