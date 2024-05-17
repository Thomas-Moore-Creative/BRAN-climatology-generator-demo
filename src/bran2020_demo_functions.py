# bran2020_demo_functions.py
"""
Module Name: bran2020_demo_functions.py
Description: This module provides statistical functions for analyzing BRAN2020 data.
Author: 
Date: 3 May 2024
"""

# Import statements
import xarray as xr
import numpy as np
import pandas as pd
# aggregations
import flox
import bottleneck
import numba
import numbagg
# utils
import gc
import os
import sys
import subprocess
from tabulate import tabulate
import json




# Function definitions
def print_chunks(data_array):
    chunks = data_array.chunks
    dim_names = data_array.dims
    readable_chunks = {dim: chunks[i] for i, dim in enumerate(dim_names)}
    for dim, sizes in readable_chunks.items():
        print(f"{dim} chunks: {sizes}")
    return readable_chunks

def stats_monthclim(ds,var_name,time_dim='time',method_str='cohorts',skipna_flag=False):
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
    currently written for single variable datasets
    """
    median_ds = ds.groupby(time_dim+'.month').median(skipna=skipna_flag,engine='flox').rename({var_name:'median_'+var_name})
    return median_ds

def quantile_monthclim(ds,var_name,skipna_flag=False,time_dim='time',q_list=[0.05,0.5,0.95],flox_flag=True):
    """
    currently written for single variable datasets
    """
    with xr.set_options(use_flox=flox_flag):
        quant = ds.groupby(time_dim+'.month').quantile(q_list,skipna=skipna_flag,dim=time_dim)
        quant_ds = xr.merge([quant.isel(quantile=0).reset_coords(drop=True).rename({var_name:'quantile_05_'+var_name}),
                            quant.isel(quantile=1).reset_coords(drop=True).rename({var_name:'median_'+var_name}),
                            quant.isel(quantile=2).reset_coords(drop=True).rename({var_name:'quantile_95_'+var_name}),])
    return quant_ds

def get_package_version(package):
    # Command to get package version using pip
    command = [sys.executable, '-m', 'pip', 'show', package]
    try:
        # Run the command and capture the output
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
        # Parse the result to find the version line
        for line in result.stdout.split('\n'):
            if 'Version:' in line:
                return line.split(': ')[1].strip()
    except subprocess.CalledProcessError:
        # Return None if the package is not found
        return None
        
def version_table(packages = None ):
    if packages is None:
        packages = []
    # Now you can safely modify list as needed
    packages.extend(['numpy', 'xarray','dask','scipy','numba','numbagg','flox','bottleneck'])
    # Prepare data for tabulation
    table = []
    for package in packages:
        version = get_package_version(package)
        if version is None:
            version = "Not installed"
        table.append([package, version])
    
    # Adjust the table to fit a max of 4 records per column set
    max_rows = 4
    num_columns = (len(table) + max_rows - 1) // max_rows  # Calculate number of columns needed
    multi_column_table = []
    
    for i in range(max_rows):
        row = []
        for j in range(num_columns):
            index = j * max_rows + i
            if index < len(table):
                row.extend(table[index])
            else:
                row.extend(['', ''])  # Fill empty spaces if no more packages
        multi_column_table.append(row)

    # Define headers for the multi-column table
    headers = []
    for i in range(num_columns):
        headers.extend(['Package', 'Version'])

    # Print the table using tabulate
    print(tabulate(multi_column_table, headers=headers, tablefmt='grid'))

def keep_only_selected_vars(ds, vars_to_keep=None):
    if vars_to_keep is None:
        vars_to_keep = ['temp','Time','st_ocean','yt_ocean','xt_ocean']
    # Calculate which variables to drop by finding the difference
    # between all variables in the dataset and the ones you want to keep
    vars_to_drop = set(ds.variables) - set(vars_to_keep)
    return ds.drop_vars(list(vars_to_drop))

def print_chunks(data_array):
    chunks = data_array.chunks
    dim_names = data_array.dims
    readable_chunks = {dim: chunks[i] for i, dim in enumerate(dim_names)}
    for dim, sizes in readable_chunks.items():
        print(f"{dim} chunks: {sizes}")
    return readable_chunks

def remove_zarr_encoding(DS):
    for var in DS:
        DS[var].encoding = {}

    for coord in DS.coords:
        DS[coord].encoding = {}
    return DS

def rechunk_each_st_ocean(ds, level_index,chunking_dict,base_write_dir,var):
    # Select the specific level
    ds_level = ds.isel(st_ocean=level_index)

    # Rechunk the dataset for this level to include all time points
    # Adjust lat and lon to fit within memory constraints
    print('--- chunking dict ---',chunking_dict)
    ds_level_rechunked = ds_level.chunk(chunking_dict)
    print('>>> depth index: ',level_index,' START')
    ds_level_rechunked = remove_zarr_encoding(ds_level_rechunked)
    print(ds_level_rechunked)
    print(ds_level_rechunked.encoding)
    print_chunks(ds_level_rechunked[var])

    # Save or return the result
    workspace = base_write_dir+var+'/'
    encoding_values = tuple(chunking_dict.values())
    encoding = {var:{'chunks':encoding_values}}
    print('encoding: '+ str(encoding))
    ds_level_rechunked.to_zarr(workspace+f'st_ocean_{level_index}_'+var+'_rechunked.zarr',mode='w', encoding=encoding, consolidated=True)
    print('depth index: ',level_index,' FINISH <<< ')
    
    
    
def concatinate_st_ocean_zarrs(zarr_dir_path,var):
    # Assuming all Zarr collections are in the same folder
    zarr_stores = [os.path.join(zarr_dir_path, d) for d in os.listdir(zarr_dir_path) if os.path.isdir(os.path.join(zarr_dir_path, d))]

    # Load all Zarr stores as a list of xarray datasets
    datasets = [xr.open_zarr(store, consolidated=True) for store in zarr_stores]

    # Concatenate all datasets along the level dimension
    all_depths_ds = xr.concat(datasets, dim='st_ocean')
    all_depths_ds = all_depths_ds.sortby('st_ocean')

    # Save the combined dataset to a new Zarr store
    all_depths_ds.to_zarr(zarr_dir_path+'/'+var+'_combined_output.zarr', consolidated=True)

def clear_and_restart(variables, client):
    """
    Clear specified variables from memory, collect garbage, and restart the Dask cluster.

    Args:
        variables (list): List of string names of the variables to clear from the namespace.
        client (dask.distributed.Client): The Dask client associated with the cluster to restart.

    Returns:
        None
    """

    # Clear specified variables
    for var in variables:
        if var in globals():
            del globals()[var]
    
    # Collect garbage
    gc.collect()
    
    # Restart the Dask cluster
    client.restart()

def load_rechunker_config():
    with open('bran_rechunker_config.json', 'r') as file:
        return json.load(file)
    
def load_stats_config():
    with open('/g/data/es60/users/thomas_moore/code/Climatology-generator-demo/src/scripts/bran_stats_config.json', 'r') as file:
        return json.load(file)