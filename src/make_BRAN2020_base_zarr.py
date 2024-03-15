# ///////////////////////
# make_BRAN2020_base_zarr.py
# 14 March 2024
#////////////////////////
# --------- packages --------------
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import intake
import xarray as xr
import pandas as pd
import numpy as np
import pandas as pd
import os
import shutil

def main():
    """
    spin up cluster & do the work   
    """
    logger = logging.getLogger(__name__)
    logger.info("Spinning up a dask cluster")
    # -----------  cluster -----------------------
    from dask.distributed import Client
    client = Client()
    # -----------  functions ----------------------
    def remove_zarr_encoding(DS):
        for var in DS:
            DS[var].encoding = {}

        for coord in DS.coords:
            DS[coord].encoding = {}
        return DS
    # -------------- setup -------------------
    logger.info("setting up")
    write_dir = '/scratch/es60/ard/reanalysis/BRAN2020/ARD/test_14032024/'
    var_request_list = ['temp','salt']
    time_period_request_list = ['daily']
    # ----- NRI Catalog ---
    logger.info("opening BRAN2020 intake catalog")
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
    logger.info("for loop over requested vars")
    for var in var_request_list:
        logger.info("variable: "+var)
        search = BRAN2020_catalog.search(variable=var,time_period=time_period_request_list)
        # load the DS
        logger.info("load DS")
        if var in ['mld','eta_t']:
            xarray_open_kwargs = {"chunks": {"time": 1,  "xt_ocean": 3600, "yt_ocean": 1500}}
        else: 
            xarray_open_kwargs = {"chunks": {"time": 1, "st_ocean": 51, "xt_ocean": 3600, "yt_ocean": 300}}
        DS=search.to_dask(xarray_open_kwargs=xarray_open_kwargs)
        # ARD - write zarr & chunk & write zarr
        logger.info(var+" ARD - start write base zarr")
        ard_file_ID = 'BRAN2020-daily-'+var+'-v14032024.zarr'
        DS.to_zarr(write_dir+ard_file_ID,consolidated=True)
        logger.info(var+" ARD - done writing base zarr for "+var)
    # -------------
    logger.info("done with base zarr for all vars")
if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
