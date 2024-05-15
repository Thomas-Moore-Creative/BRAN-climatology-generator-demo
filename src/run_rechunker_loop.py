# ///////////////////////
# run_rechunker_loop.py
# 13 May 2024
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
    """
    spin up cluster & do the work   
    """
    print("importing functions ...")
    # Append the directory of the module to sys.path - import functions
    sys.path.append('/g/data/es60/users/thomas_moore/code/Climatology-generator-demo/src/')
    import bran2020_demo_functions as my_tools
    from bran2020_demo_functions import keep_only_selected_vars, load_rechunker_config, print_chunks, rechunk_each_st_ocean, remove_zarr_encoding, concatinate_st_ocean_zarrs
    # load config
    config = load_rechunker_config()
    print(">>> config: "+str(config))
    var = config['variable']
    print("variable requested: "+var)
    BRAN2020_ard_path = config['BRAN2020_ard_path']
    print("BRAN2020_ard_path: "+BRAN2020_ard_path)
    write_rechunk_base_dir = config['write_rechunk_base_dir']
    print("write_rechunk_base_dir: "+write_rechunk_base_dir)
    n_workers = config['n_workers']
    print("n_workers: "+str(n_workers))
    threads_per_worker = config['threads_per_worker']
    print("threads_per_worker: "+str(threads_per_worker))
    memory_limit = config['memory_limit']
    print("memory_limit: "+memory_limit)
    create_base_zarr = config['create_base_zarr']
    print("create_base_zarr: "+str(create_base_zarr))
    read_base_zarr_path = config['read_base_zarr_path']
    print("read_base_zarr_path: "+read_base_zarr_path)
    run_rechunker_loop = config['run_rechunker_loop']
    print("run_rechunker_loop: "+str(run_rechunker_loop))
    level_start = config['level_start']
    print("level_start: "+str(level_start))
    level_stop = config['level_stop']
    print("level_stop: "+str(level_stop))
    concatinate_st_ocean_zarrs = config['concatinate_st_ocean_zarrs']
    print("concatinate_st_ocean_zarrs: "+str(concatinate_st_ocean_zarrs))
    lat_name = config['lat_name']
    print("lat_name: "+lat_name)
    lon_name = config['lon_name']
    print("lon_name: "+lon_name)

    # -----------  cluster -----------------------
    print(">>> Spinning up a dask cluster...")
    
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=n_workers,threads_per_worker=threads_per_worker,memory_limit=memory_limit)
    client = Client(cluster)
    print(client)
    #
    #
    
    # Get current date and time
    now = datetime.now()

    # Format as a string
    timestamp_str = now.strftime("%Y.%m.%d.%H.%M.%S")
    print('timestamp: '+timestamp_str)
    # base zarr settings
    xarray_open_kwargs = {"Time": 1, "st_ocean": 51, lon_name: 3600, lat_name: 1500}
    chunking_string = 'chunks_' + ''.join(str(key) + str(value)+ '.' for key, value in xarray_open_kwargs.items())
    ard_rcTime_file_ID = 'BRAN2020-'+var+'-'+chunking_string+timestamp_str+'.zarr'
    print(">>> ard_rcTime_file_ID: "+ard_rcTime_file_ID)

    if create_base_zarr == True:
        # write base zarr
        print(">>> writing base zarr for: "+var)

        # load the netcdf and write the base zarr
        vars_to_keep=[var,'Time','st_ocean',lat_name,lon_name]
        ds = xr.open_mfdataset('/g/data/gb6/BRAN/BRAN2020/daily/ocean_'+var+'_*.nc',
                       parallel=True,chunks=xarray_open_kwargs,
                       preprocess=lambda ds: keep_only_selected_vars(ds, vars_to_keep=vars_to_keep))
        ds
        ds32 = ds
        ds32[var] = ds32[var].astype(np.float32)
        ds32.to_zarr(BRAN2020_ard_path+ard_rcTime_file_ID,consolidated=True)
        print(">>> finished writing base zarr for: "+var)
    else:
        print(">>> skipping writing base zarr for: "+var)
        ard_rcTime_file_ID = read_base_zarr_path
    if run_rechunker_loop == True:
        print(">>> running rechunker loop for: "+var)
        # load the base zarr
        DS = xr.open_zarr(ard_rcTime_file_ID,consolidated=True)
        print(DS)
        readable_chunks = print_chunks(DS[var])
        chunking_dict_per_depth = {"Time": 11322, lon_name: 120, lat_name: 120}
        print("chunking_dict_per_depth: "+str(chunking_dict_per_depth))
        # Iterate over each depth and process
        for depth_index in range(level_start,level_stop):
            rechunk_each_st_ocean(DS, depth_index,chunking_dict=chunking_dict_per_depth,base_write_dir=write_rechunk_base_dir,var=var)
            print("finished depth_index: "+str(depth_index))
        # Concatenate the rechunked Zarrs
        print("finished rechunking all depths to 2D zarr")
    if concatinate_st_ocean_zarrs == True:
        print(">>> concatinating zarrs")
        zarr_dir_path = write_rechunk_base_dir+var
        print("zarr_dir_path: "+zarr_dir_path)
        concatinate_st_ocean_zarrs(zarr_dir_path=zarr_dir_path)
        print("finished concatinating zarrs")
    client.shutdown()
if __name__ == "__main__":
    main()
