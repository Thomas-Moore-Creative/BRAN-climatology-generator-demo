# ///////////////////////
# run_rechunker.py
# 6 May 2024
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

def main():
    """
    spin up cluster & do the work   
    """
    print("importing functions ...")
    # Append the directory of the module to sys.path - import functions
    sys.path.append('/g/data/es60/users/thomas_moore/code/Climatology-generator-demo/src/')
    import bran2020_demo_functions as my_tools

    print("Spinning up a dask cluster...")
    # -----------  cluster -----------------------
    import dask
    from dask.distributed import Client, LocalCluster

    
    cluster = LocalCluster(n_workers=24,threads_per_worker=1,processes=True)
    client = Client(cluster)
    print(client)
#
    #
    config = configparser.ConfigParser()
    config.read('/g/data/es60/users/thomas_moore/code/BRAN2020-intake-catalog/config.ini')
    # Get the value of a variable
    catalog_path = config.get('paths', 'catalog_path')
    #
    BRAN2020_catalog = intake.open_esm_datastore(catalog_path+'BRAN2020.json',columns_with_iterables=['variable'])
    var_request_list = ['salt']
    var = var_request_list[0]
    print("variable requested: "+var)
    time_period_request_list = ['daily']
    search = BRAN2020_catalog.search(variable=var,time_period=time_period_request_list)
    xarray_open_kwargs = {"chunks": {"Time": -1,'st_ocean':10}}
    DS=search.to_dask(xarray_open_kwargs=xarray_open_kwargs)
    my_tools.print_chunks(DS[var])
    # rechunk DS for {'Time':-1}
    # -----------  functions ----------------------
    def remove_zarr_encoding(DS):
        for var in DS:
            DS[var].encoding = {}

        for coord in DS.coords:
            DS[coord].encoding = {}
        return DS
    # -------------- setup -------------------
    print("starting rechunker workflow")
    ARD_dir = '/scratch/es60/ard/reanalysis/BRAN2020/ARD/'
    var_request_list = ['salt']
    # -------- run over variables -----
    for var in var_request_list:
        print("variable: "+var)
        print(var+" ARD - CHUNK for time and WRITE zarr")
        ard_rcTime_file_ID = 'BRAN2020-daily-'+var+'-chunk4time-v06052024.zarr'
        # clear zarr encoding
        print('clear zarr encoding')
        DS = DS.chunk({'Time':100})
        input_ds = remove_zarr_encoding(DS)
        my_tools.print_chunks(input_ds[var])    
        print('encoding is reset to: ')
        print(input_ds.encoding)
        # rechunk for time
        print('setup rechunker task')
        target_chunks = {'Time':-1,'st_ocean':-1,'xt_ocean':1,'yt_ocean':100}
        max_mem = "3GB"
        target_store = ARD_dir+ard_rcTime_file_ID
        temp_store = "/scratch/es60/ard/rechunker_scratch/rechunker-tmp.zarr"

        # need to remove the existing stores or it won't work
        os.system("rm -rf /scratch/es60/ard/rechunker_scratch/rechunker-tmp.zarr")

        # rechunk directly from dataset this time
        rechunk_plan = rechunk(
            input_ds, target_chunks, max_mem, target_store, temp_store=temp_store
        )
        print('executing rechunk plan')
        rechunk_plan
        rechunk_plan.execute()
        # consolidate metadata
        zarr.consolidate_metadata(target_store)
        print(var+" ARD - finished with write rechunked zarr")
        
    # -------------
    print("***** done with rechunker batch job ******")
    client.shutdown()
if __name__ == "__main__":
    main()
