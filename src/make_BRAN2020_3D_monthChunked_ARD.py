# ///////////////////////
# make BRAN2020 3D monthChunked ARD.py
# 25 April 2024
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

def main():
    """
    spin up cluster & do the work   
    """

    print("Spinning up a dask cluster...")
    # -----------  cluster -----------------------
    cluster=LocalCluster(n_workers=48,processes=True,threads_per_worker=1)
    client = Client(cluster)
    print(client)
    # -----------  setup ----------------------
    print("setup")
    var_request_list = ['u','v']
    time_period_request_list = ['daily']
    # ----- NRI Catalog ---
    print("opening BRAN2020 intake catalog")
    #
    import configparser

    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the config file
    #########
    #### you will need to specifiy your correct path the the `data-catalogue/config.ini` file 
    #########
    config.read('/g/data/es60/users/thomas_moore/code/BRAN2020-intake-catalog/config.ini')

    # Get the value of a variable
    catalog_path = config.get('paths', 'catalog_path')
    #
    BRAN2020_catalog = intake.open_esm_datastore(catalog_path+'BRAN2020.json',columns_with_iterables=['variable'])
    # -------- run over variables -----
    print("for loop over requested vars")
    for var in var_request_list:
        print("variable: "+var)
        search = BRAN2020_catalog.search(variable=var,time_period=time_period_request_list)
        # load the DS
        print("load DS")
        if var in ['mld','eta_t']:
            xarray_open_kwargs = {"chunks": {"time": 27,  "xt_ocean": 3600, "yt_ocean": 1500}}
        elif var in ['u','v']: 
            xarray_open_kwargs = {"chunks": {"time": 27, "st_ocean": 10, "xu_ocean": 3600, "yu_ocean": 1500}}
        else: 
            xarray_open_kwargs = {"chunks": {"time": 27, "st_ocean": 10, "xt_ocean": 3600, "yt_ocean": 1500}}   
        DS=search.to_dask(xarray_open_kwargs=xarray_open_kwargs)
        # ARD - write zarr & chunk & write zarr
        print(var+" ARD - start write first zarr")
        BRAN2020_ard_path = '/scratch/es60/ard/reanalysis/BRAN2020/ARD/'
        chunking_string = 'chunks.' + ''.join(str(key) + str(value) + '.' for key, value in xarray_open_kwargs['chunks'].items())
        ard_file_ID = 'BRAN2020.daily.'+var+'.'+chunking_string+'v26042024.zarr'
        DS[var]=DS[var].astype('float32')
        DS.to_zarr(BRAN2020_ard_path+ard_file_ID,consolidated=True)
        print(var+" ARD - finished monthChunked zarr collection for "+var)
    # -------------
    print(">>>>>>>>>> all done !!!")
if __name__ == "__main__":
    main()

    