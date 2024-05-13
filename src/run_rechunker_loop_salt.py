# ///////////////////////
# run_rechunker_loop_salt.py
# 11 May 2024
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

def main():
    """
    spin up cluster & do the work   
    """
    print("importing functions ...")
    # Append the directory of the module to sys.path - import functions
    sys.path.append('/g/data/es60/users/thomas_moore/code/Climatology-generator-demo/src/')
    import bran2020_demo_functions as my_tools
    from bran2020_demo_functions import print_chunks, rechunk_each_st_ocean, remove_zarr_encoding, concatinate_st_ocean_zarrs
    #import my_tools.clear_and_restart as clear_and_restart


    print("Spinning up a dask cluster...")
    # -----------  cluster -----------------------
    from dask.distributed import Client, LocalCluster

    
    cluster = LocalCluster(n_workers=48,threads_per_worker=1,memory_limit='50GB')
    client = Client(cluster)
    print(client)
    #
    #
    # setup 
    var = 'salt'
    print("variable requested: "+var)
    BRAN2020_ard_path = '/scratch/es60/ard/reanalysis/BRAN2020/ARD/'
    # Get current date and time
    now = datetime.now()

    # Format as a string
    timestamp_str = now.strftime("%Y.%m.%d.%H.%M.%S")
    print('timestamp: '+timestamp_str)

    # load the base zarr
    DS = xr.open_zarr(BRAN2020_ard_path+'float32.BRAN2020-salt-chunks_Time1.st_ocean51.xt_ocean3600.yt_ocean1500.v08052024.zarr',
                                                  consolidated=True)
    print(DS)
    readable_chunks = print_chunks(DS[var])
    chunking_dict_per_depth = {"Time": 11322, "xt_ocean": 120, "yt_ocean": 120}
    print("chunking_dict_per_depth: "+str(chunking_dict_per_depth))
    # Iterate over each depth and process
    for depth_index in range(15,51):
        rechunk_each_st_ocean(DS, depth_index,chunking_dict=chunking_dict_per_depth)
        print("finished depth_index: "+str(depth_index))
    # Concatenate the rechunked Zarrs
    print("finished all depths")
    concatinate_st_ocean_zarrs()
    print("finished concatinating zarrs")
    client.shutdown()
if __name__ == "__main__":
    main()
