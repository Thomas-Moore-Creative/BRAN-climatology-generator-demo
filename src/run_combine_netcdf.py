# ///////////////////////
# run_combine_netcdf.py
# 31 May 2024
#////////////////////////
# --------- packages --------------
import intake
import xarray as xr
import pandas as pd
import numpy as np
import pandas as pd

from dask.distributed import Client, LocalCluster
import dask
import datetime
import zarr

import gc
import sys
import subprocess
from tabulate import tabulate
import os
import glob


def main():
    # Append the directory of the module to sys.path - import functions
    sys.path.append('/g/data/es60/users/thomas_moore/code/Climatology-generator-demo/src/')
    import bran2020_demo_functions as my_tools
    from bran2020_demo_functions import keep_only_selected_vars, load_rechunker_config, print_chunks, rechunk_each_st_ocean, remove_zarr_encoding, version_table, concatinate_st_ocean_zarrs
#
    # Set configuration options
    dask.config.set({
        'distributed.comm.timeouts.connect': '90s',  # Timeout for connecting to a worker
        'distributed.comm.timeouts.tcp': '90s',  # Timeout for TCP communications
    })

    cluster = LocalCluster(
        n_workers=12,          # Number of workers
        threads_per_worker=1,
        memory_limit='240GB' # Memory limit per each worker
        )
    client = Client(cluster)
    print(client)
    print(">>> cluster running ...")

    # Create an empty dictionary
    dynamic_ds = {}

    # Define your var and phase lists
    var_values = ['temp', 'salt','u','v','eta_t','mld']  # replace with your actual list
    phase_values = ['alltime', 'neutral','la_nina','el_nino']  # replace with your actual list

    #    Iterate over all combinations of var and phase
    for var_name in var_values:
        for phase_name in phase_values:
            # Generate the object name
            ds_name = f'{var_name}_{phase_name}_ds'
        
            # Store the value in the dictionary
            results_path = '/g/data/es60/users/thomas_moore/clim_demo_results/daily/bran2020_intermediate_results/'
            files = glob.glob(results_path+'*_'+var_name+'_*'+phase_name+'*.nc')
            sorted_files = sorted(files, key=os.path.getctime)
            
            dynamic_ds[ds_name] = xr.open_mfdataset(files,parallel=True)  # replace with your actual value

    # Add the phase string to the name of all variables in each dataset
    for ds_name, dataset in dynamic_ds.items():
        # Get the phase name from the dataset name
        phase_name = '_'.join(ds_name.split('_')[1:-1])
        if phase_name not in phase_values:
            phase_name = '_'.join(ds_name.split('_')[2:-1])
        if phase_name in phase_values:
            # Add the phase string to the name of all variables
            for var_name in dataset.data_vars:
                new_var_name = f'{var_name}_{phase_name}'
                dataset = dataset.rename({var_name: new_var_name})
            dynamic_ds[ds_name] = dataset
        else:
            print(f"No match found for phase name: {phase_name}")

    merged_datasets = {}
    for var_name in var_values:
            # Get all datasets with the same var_name
            var_datasets = [dataset for ds_name, dataset in dynamic_ds.items() if var_name+'_' in ds_name]
            
            # Merge the datasets along the time dimension
            merged_dataset = xr.merge(var_datasets)
            
            # Store the merged dataset in the dictionary
            merged_datasets[var_name] = merged_dataset

    # Lazy load each dataset
    lazy_datasets = {}
    for var_name, merged_dataset in merged_datasets.items():
        #print([var_name,merged_dataset])
        lazy_datasets[var_name] = merged_dataset

    # coordinate nomeclature
    coordinate_names = {
        "lat_name_dict": {
            "temp": "yt_ocean",
            "salt": "yt_ocean",
            "u": "yu_ocean",
            "v": "yu_ocean",
            "mld": "yt_ocean",
            "eta_t": "yt_ocean"
        },
        "lon_name_dict": {
            "temp": "xt_ocean",
            "salt": "xt_ocean",
            "u": "xu_ocean",
            "v": "xu_ocean",
            "mld": "xt_ocean",
            "eta_t": "xt_ocean"
        },
        "depth_name_dict": {
            "temp": "st_ocean",
            "salt": "st_ocean",
            "u": "st_ocean",
            "v": "st_ocean"
        }
    }

    # rechunk all the datasets for 1,1,300,300, or 1,300,300

    rechunked_datasets = {}
    for var_name, lazy_dataset in lazy_datasets.items():
        rechunked_dataset = lazy_dataset
        if var_name in coordinate_names['depth_name_dict']:
            depth_coord = coordinate_names['depth_name_dict'][var_name]
            rechunked_dataset = rechunked_dataset.chunk({depth_coord: 1})
        if var_name in coordinate_names['lon_name_dict']:
            lon_coord = coordinate_names['lon_name_dict'][var_name]
            rechunked_dataset = rechunked_dataset.chunk({lon_coord: 300})
        if var_name in coordinate_names['lat_name_dict']:
            lat_coord = coordinate_names['lat_name_dict'][var_name]
            rechunked_dataset = rechunked_dataset.chunk({lat_coord: 300})
        rechunked_dataset = rechunked_dataset.chunk({'month': 1})
        rechunked_datasets[var_name] = rechunked_dataset

    # run the compute and write
    timestamp_str = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    write_vars = ['u']
    print(">>> writing netcdf files ...")
    for write_var in write_vars:
        print(f"Writing NetCDF: {write_var}")
        write_path = '/g/data/es60/users/thomas_moore/clim_demo_results/daily/bran2020_final_results/'
        write_file = f'bran2020_{write_var}_ENSO_stats_{timestamp_str}.nc'
        write_this = rechunked_datasets[write_var].compute()
        # Set up encoding for NetCDF write
        encoding = {} #setup encoding dict
        chunksizes_tuple = (1, 300, 300)  # set default 2D chunksizes for netcdf write
        # Check if var is present in the 'st_ocean' subdictionary
        if write_var in coordinate_names['depth_name_dict']:
            chunksizes_tuple = (1, 1, 300, 300)  # set chunksizes for 3D variable
        for var_name in write_this.data_vars:
            encoding[var_name] = {'zlib': True, 'complevel': 5, 'dtype': 'float32', 'chunksizes': chunksizes_tuple} # encode only the data variables
        # Save to NetCDF with chunking and compression encoding
        write_this.to_netcdf(write_path+write_file, engine='netcdf4',encoding=encoding)
        print(f"NetCDF written: {write_var}")
        client.restart()
    print(">>> all done ...")
 
if __name__ == "__main__":
    main()