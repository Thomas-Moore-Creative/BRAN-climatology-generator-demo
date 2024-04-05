# ///////////////////////
# make_BRAN2020_rechunker_salt_zarr.py
# 5 April 2024
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
from rechunker import rechunk
import zarr 

def main():
    """
    spin up cluster & do the work   
    """
    logger = logging.getLogger(__name__)
    logger.info("Spinning up a dask cluster")
    # -----------  cluster -----------------------
    import dask
    import distributed

    with dask.config.set({"distributed.scheduler.worker-saturation": 1.0,
                      "distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0}):
        client = distributed.Client()
    # -----------  functions ----------------------
    def remove_zarr_encoding(DS):
        for var in DS:
            DS[var].encoding = {}

        for coord in DS.coords:
            DS[coord].encoding = {}
        return DS
    # -------------- setup -------------------
    logger.info("setting up")
    ARD_dir = '/scratch/es60/ard/reanalysis/BRAN2020/ARD/'
    var_request_list = ['salt']
    # ------
    logger.info("starting rechunking workflow")
    # -------- run over variables -----
    logger.info("for loop over requested vars")
    for var in var_request_list:
        logger.info("variable: "+var)
        logger.info(var+" ARD - CHUNK for time and WRITE zarr")
        ard_file_ID = 'BRAN2020-daily-'+var+'-v03042024.zarr'
        ard_rcTime_file_ID = 'BRAN2020-daily-'+var+'-chunk4time-v05042024.zarr'
        BRAN2020 = xr.open_zarr(ARD_dir+ard_file_ID,consolidated=True)
        
        input_ds = BRAN2020
        target_chunks = {'Time':-1,'st_ocean':-1,'xt_ocean':1,'yt_ocean':100}
        max_mem = "8GB"
        target_store = ARD_dir+ard_rcTime_file_ID
        temp_store = "/scratch/es60/ard/rechunker_scratch/rechunker-tmp.zarr"

        # need to remove the existing stores or it won't work
        os.system("rm -rf /scratch/es60/ard/rechunker_scratch/rechunker-tmp.zarr")

        # rechunk directly from dataset this time
        rechunk_plan = rechunk(
            input_ds, target_chunks, max_mem, target_store, temp_store=temp_store
        )
        rechunk_plan.execute()
        # touch the log file to indicate that the process has finished
        os.system("touch /scratch/es60/ard/reanalysis/BRAN2020/ARD/logs/finished_BRAN2020-salt-rechunker-zarr.log")
        zarr.consolidate_metadata(target_store)
        logger.info(var+" ARD - finished with write rechunked zarr")
        
    # -------------
    logger.info("***** done with batch job ******")
    client.shutdown()
if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
