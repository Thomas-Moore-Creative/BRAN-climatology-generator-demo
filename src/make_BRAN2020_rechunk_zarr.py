# ///////////////////////
# make_BRAN2020_rechunk_zarr.py
# 18 March 2024
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
    ARD_dir = '/scratch/es60/ard/reanalysis/BRAN2020/ARD/test_14032024/'
    var_request_list = ['temp','salt']
    # ------
    logger.info("starting rechunking workflow")
    # -------- run over variables -----
    logger.info("for loop over requested vars")
    for var in var_request_list:
        logger.info("variable: "+var)
        logger.info(var+" ARD - CHUNK for time and WRITE zarr")
        ard_file_ID = 'BRAN2020-daily-'+var+'-v14032024.zarr'
        ard_rcTime_file_ID = 'BRAN2020-daily-'+var+'-chunk4time-v14032024.zarr'
        BRAN2020 = xr.open_zarr(ARD_dir+ard_file_ID,consolidated=True)
        if 'st_ocean' in BRAN2020.coords:
            BRAN2020_rcTime =  BRAN2020.chunk({'Time':-1,'st_ocean':-1,'xt_ocean':1,'yt_ocean':100})
        else:
            BRAN2020_rcTime =  BRAN2020.chunk({'Time':-1,'xt_ocean':100,'yt_ocean':100})
        BRAN2020_rcTime = remove_zarr_encoding(BRAN2020_rcTime)
        BRAN2020_rcTime.to_zarr(ARD_dir+ard_rcTime_file_ID,consolidated=True)
        logger.info(var+" ARD - finished with write rechunked zarr")
        client.restart()
    # -------------
    logger.info("done with all vars")
    client.shutdown()
if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
