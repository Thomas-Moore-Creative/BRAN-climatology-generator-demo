# ///////////////////////
# run_bran_stats.py
# 17 May 2024
#////////////////////////
# --------- packages --------------
import intake
import xarray as xr
import pandas as pd
import numpy as np
from dask.distributed import Client, LocalCluster
import dask
import datetime
import os
import configparser
import sys
from rechunker import rechunk
import zarr 
from datetime import datetime
import json

def main():
    print("importing functions ...")
    # Append the directory of the module to sys.path - import functions
    sys.path.append('/g/data/es60/users/thomas_moore/code/Climatology-generator-demo/src/')
    import bran2020_demo_functions as my_tools
    from bran2020_demo_functions import keep_only_selected_vars, load_stats_config, print_chunks, stats_monthclim, median_monthclim, quantile_monthclim
    # load config
    config = load_stats_config()
    print(">>> config: "+str(config))
    var = config['variable']
    print("variable requested: "+var)
    zarr_path_dict = config['zarr_path_dict']
    print(">>> zarr_path_dict: "+str(zarr_path_dict))
    write_results_base_dir = config['write_results_base_dir']
    print(">>> write_results_base_dir: "+str(write_results_base_dir))
    n_workers = config['n_workers']
    print(">>> n_workers: "+str(n_workers))
    threads_per_worker = config['threads_per_worker']
    print(">>> threads_per_worker: "+str(threads_per_worker))
    memory_limit = config['memory_limit']
    print(">>> memory_limit: "+str(memory_limit))
    run_base_stats = config['run_base_stats']
    print(">>> run_base_stats: "+str(run_base_stats))
    run_quant = config['run_quant']
    print(">>> run_quant: "+str(run_quant))
    run_all_time = config['run_all_time']
    print(">>> run_all_time: "+str(run_all_time))
    run_neutral = config['run_neutral']
    print(">>> run_neutral: "+str(run_neutral))
    run_la_nina = config['run_la_nina']
    print(">>> run_la_nina: "+str(run_la_nina))
    run_el_nino = config['run_el_nino']
    print(">>> run_el_nino: "+str(run_el_nino))
    lat_name_dict = config['lat_name_dict']
    print(">>> lat_name_dict: "+str(lat_name_dict))
    lon_name_dict = config['lon_name_dict']
    print(">>> lon_name_dict: "+str(lon_name_dict))
    time_name = config['time_name']
    print(">>> time_name: "+str(time_name))
    # ------
    print(">>> Spinning up a dask cluster...")
    
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=n_workers,threads_per_worker=threads_per_worker,memory_limit=memory_limit)
    client = Client(cluster)
    print(client)
    # Get current date and time
    now = datetime.now()
    # Format as a string
    timestamp_str = now.strftime("%Y.%m.%d.%H.%M.%S")
    print('timestamp: '+timestamp_str)
    # ----------------
    if run_all_time == True:
        if run_base_stats == True:
            print(">>> running base stats ...")
            #
            lat_name = lat_name_dict[var]
            lon_name = lon_name_dict[var]
            vars_to_keep=[var,time_name,'st_ocean',lat_name,lon_name]
            xarray_open_kwargs = {'chunks': {time_name: -1,'st_ocean':10}}
            ds = xr.open_mfdataset('/g/data/gb6/BRAN/BRAN2020/daily/ocean_'+var+'_*.nc',
                        parallel=True,chunks=xarray_open_kwargs['chunks'],
                        preprocess=lambda ds: keep_only_selected_vars(ds, vars_to_keep=vars_to_keep))
            print_chunks(ds[var])
            stats_monthclim_ds = stats_monthclim(ds,var_name=var,time_dim=time_name,skipna_flag=False,method_str='cohorts')
            print(stats_monthclim_ds.nbytes/1e9)
            # write to netcdf
            results_path = write_results_base_dir
            results_file = 'BRAN2020_base_stats_'+var+'_'+timestamp_str+'.nc'

            print("writing to the base stats netcdf file for : "+var+" ....")

            # Specify chunks
            chunks = {'month':12,'st_ocean': 1, lat_name:1500, lon_name:3600}

            # Specify encoding
            encoding = {}
            for var_name in stats_monthclim_ds.data_vars:
                encoding[var_name] = {'zlib': True, 'complevel': 5, 'dtype': 'float32', 'chunksizes': (12, 1, 1500, 3600)}
            # Save to NetCDF with chunking and encoding
            stats_monthclim_ds.chunk(chunks).to_netcdf(results_path+results_file, engine='netcdf4',encoding=encoding)

            print("netcdf written: "+var)
            print(">>>>> DONE with basic stats calc and write to netcdf for: "+var)
        if run_quant == True:
            print(">>> running quant stats ...")
            #
            lat_name = lat_name_dict[var]
            lon_name = lon_name_dict[var]
            ds = xr.open_zarr(zarr_path_dict[var],consolidated=True)
            ds = ds.sortby('st_ocean')
            print(">>> sorted depths")
            print(ds.st_ocean.values)
            print(">>>> chunks going into quant calculation")
            print_chunks(ds[var])
            quantile_monthclim_ds = quantile_monthclim(ds,var_name=var,time_dim=time_name,skipna_flag=False)
            print(quantile_monthclim_ds.nbytes/1e9)
            #convert float64 quantile to float32
            for data_var in quantile_monthclim_ds.data_vars:
                quantile_monthclim_ds[data_var] = quantile_monthclim_ds[data_var].astype('float32')
            # write to netcdf
            results_path = write_results_base_dir
            results_file = 'BRAN2020_quantile_stats_'+var+'_'+timestamp_str+'.nc'
            print("writing to the quant netcdf file for : "+var+" ....")
            
            # Specify chunks
            chunks = {'month':12,'st_ocean': 1, lat_name:1500, lon_name:3600}

            # Specify encoding
            encoding = {}
            for var_name in quantile_monthclim_ds.data_vars:
                encoding[var_name] = {'zlib': True, 'complevel': 5, 'dtype': 'float32', 'chunksizes': (12, 1, 1500, 3600)}

            # Save to NetCDF with chunking and encoding
            quantile_monthclim_ds.chunk(chunks).to_netcdf(results_path+results_file, engine='netcdf4',encoding=encoding)
            print("quant netcdf written: "+var)
            print(">>>>> DONE with quant stats calc and write to netcdf for: "+var)
    print(">>> all done ...")

if __name__ == "__main__":
    main()