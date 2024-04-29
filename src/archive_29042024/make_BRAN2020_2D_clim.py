# ///////////////////////
# make_BRAN2020_2D_clim.py
# 4 April 2024
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
    write_dir = '/scratch/es60/ard/reanalysis/BRAN2020/ARD/draft_03042024/'
    file_ver = 'v04042024'
    var_request_list = ['mld','eta_t']
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
        ard_file_ID = 'BRAN2020-daily-'+var+'-'+file_ver+'.zarr'
        DS.to_zarr(write_dir+ard_file_ID,consolidated=True)
        logger.info(var+" ARD - done writing base zarr for "+var)
        logger.info(var+" ARD - CHUNK for time and WRITE zarr")
        ard_rcTime_file_ID = 'BRAN2020-daily-'+var+'-chunk4time'+'-'+file_ver+'.zarr'
        BRAN2020 = xr.open_zarr(ARD_dir+ard_file_ID,consolidated=True)
        if 'st_ocean' in BRAN2020.coords:
            BRAN2020_rcTime =  BRAN2020.chunk({'Time':-1,'st_ocean':-1,'xt_ocean':1,'yt_ocean':100})
        else:
            BRAN2020_rcTime =  BRAN2020.chunk({'Time':-1,'xt_ocean':100,'yt_ocean':100})
        BRAN2020_rcTime = remove_zarr_encoding(BRAN2020_rcTime)
        BRAN2020_rcTime.to_zarr(ARD_dir+ard_rcTime_file_ID,consolidated=True)
        logger.info(var+" ARD - finished with write rechunked zarr")
    # ------------- clim workflow step --------
        logger.info("starting clim workflow step for "+var)
        DS_rcTime = xr.open_zarr(ARD_dir+ard_rcTime_file_ID,consolidated=True)
        var_chunked_time = DS_rcTime
    # define El Nino and La Nina using NCAR ONI data
        ONI_DF = pd.read_csv('/g/data/xv83/users/tm4888/data/ENSO/NCAR_ONI.csv')
        ONI_DF.set_index('datetime',inplace=True)
        ONI_DF.index = pd.to_datetime(ONI_DF.index)
        el_nino_threshold = 0.5
        la_nina_threshold = -0.5
        el_nino_threshold_months = ONI_DF["ONI"].ge(el_nino_threshold)
        la_nina_threshold_months = ONI_DF["ONI"].le(la_nina_threshold) 
        ONI_DF = pd.concat([ONI_DF, el_nino_threshold_months.rename('El Nino threshold')], axis=1)
        ONI_DF = pd.concat([ONI_DF, la_nina_threshold_months.rename('La Nina threshold')], axis=1)
        ONI_DF = pd.concat([ONI_DF, el_nino_threshold_months.diff().ne(0).cumsum().rename('El Nino event group ID')], axis=1)
        ONI_DF = pd.concat([ONI_DF, la_nina_threshold_months.diff().ne(0).cumsum().rename('La Nina event group ID')], axis=1)
        El_Nino_Series = ONI_DF.groupby('El Nino event group ID')['ONI'].filter(lambda x: len(x) >= 5,dropna=False).where(ONI_DF['El Nino threshold'] == True)
        ONI_DF = pd.concat([ONI_DF, El_Nino_Series.rename('El Nino')], axis=1)
        La_Nina_Series = ONI_DF.groupby('La Nina event group ID')['ONI'].filter(lambda x: len(x) >= 5,dropna=False).where(ONI_DF['La Nina threshold'] == True)
        ONI_DF = pd.concat([ONI_DF, La_Nina_Series.rename('La Nina')], axis=1)
    #
    # filter BRAN2020 data by ENSO
    # address monthly to daily
    #
        ONI_DF_BRANtime = ONI_DF['1993-01':'2023-06']
        ONI_DF_BRANtime['El Nino LOGICAL'] = ONI_DF_BRANtime['El Nino'].notnull()
        ONI_DF_BRANtime['La Nina LOGICAL'] = ONI_DF_BRANtime['La Nina'].notnull()
    # shift back from middle of month
        ONI_DF_BRANtime.index += pd.Timedelta(-14, 'd')
    # modify end value for upsample
        ONI_DF_BRANtime.loc[pd.to_datetime('2023-07-01 00:00:00')] = 'NaN'
    #upsample
        ONI_DF_BRANtime = ONI_DF_BRANtime.resample('D').ffill()
    #drop last dummy date
        ONI_DF_BRANtime = ONI_DF_BRANtime[:-1]
        El_Nino_mask = ONI_DF_BRANtime['El Nino LOGICAL']
        El_Nino_mask = El_Nino_mask.to_xarray()
        El_Nino_mask = El_Nino_mask.rename({'datetime':'Time'})
        sync_Time = var_chunked_time.Time
        El_Nino_mask['Time'] = sync_Time
        
        La_Nina_mask = ONI_DF_BRANtime['La Nina LOGICAL']
        La_Nina_mask = La_Nina_mask.to_xarray()
        La_Nina_mask = La_Nina_mask.rename({'datetime':'Time'})
        sync_Time = var_chunked_time.Time
        La_Nina_mask['Time'] = sync_Time

        ONI_DF_BRANtime['Neutral LOGICAL'] = (ONI_DF_BRANtime['El Nino LOGICAL'] == False) & (ONI_DF_BRANtime['La Nina LOGICAL'] == False)
    
    
    # mask events
        El_Nino_mask_0_1 = El_Nino_mask.to_dataframe().replace({True: 1, False: 0}).to_xarray()
        La_Nina_mask_0_1 = La_Nina_mask.to_dataframe().replace({True: 1, False: 0}).to_xarray()
        El_Nino_mask_TIMES = El_Nino_mask_0_1['Time'].where(El_Nino_mask_0_1,drop=True)
        La_Nina_mask_TIMES = La_Nina_mask_0_1['Time'].where(La_Nina_mask_0_1,drop=True)
        El_Nino_var_chunked_time = DS_rcTime.sel({'Time':El_Nino_mask_TIMES.Time})
        La_Nina_var_chunked_time = DS_rcTime.sel({'Time':La_Nina_mask_TIMES.Time})

    # Climatologies and anomalies
        var_monthly_climatology = get_monthly_climatology(var_chunked_time, time_coord_name = 'Time')
        if 'st_ocean' in var_monthly_climatology.coords:
            var_monthly_climatology_rc = var_monthly_climatology.chunk({'st_ocean':10,'xt_ocean':3600,'month':1})
        else:
            var_monthly_climatology_rc = var_monthly_climatology.chunk({'xt_ocean':3600,'month':-1})
    #El Nino
        El_Nino_var_monthly_climatology = get_monthly_climatology(El_Nino_var_chunked_time, time_coord_name = 'Time')
        if 'st_ocean' in var_monthly_climatology.coords:
            El_Nino_var_monthly_climatology_rc = El_Nino_var_monthly_climatology.chunk({'st_ocean':10,'xt_ocean':3600,'month':1})
        else:
            El_Nino_var_monthly_climatology_rc = El_Nino_var_monthly_climatology.chunk({'xt_ocean':3600,'month':12})
        
    #La Nina
        La_Nina_var_monthly_climatology = get_monthly_climatology(La_Nina_var_chunked_time, time_coord_name = 'Time')
        if 'st_ocean' in var_monthly_climatology.coords:
            La_Nina_var_monthly_climatology_rc = La_Nina_var_monthly_climatology.chunk({'st_ocean':10,'xt_ocean':3600,'month':1})
        else:
            La_Nina_var_monthly_climatology_rc = La_Nina_var_monthly_climatology.chunk({'xt_ocean':3600,'month':12})
        
    
    # make objects
        var_monthly_climatology_rc = var_monthly_climatology_rc.rename({var:'climatological_'+var})
        El_Nino_var_monthly_climatology_rc = El_Nino_var_monthly_climatology_rc.rename({var:'El_Nino_climatological_'+var})
        La_Nina_var_monthly_climatology_rc = La_Nina_var_monthly_climatology_rc.rename({var:'La_Nina_climatological_'+var})
        BRAN2020_var_climatology = xr.merge([var_monthly_climatology_rc,El_Nino_var_monthly_climatology_rc,La_Nina_var_monthly_climatology_rc])
        BRAN2020_var_climatology_rc = BRAN2020_var_climatology.chunk({'yt_ocean':-1})

    # write out results in NetCDF
        logger.info(var+" writing nc file")
        write_path = '/g/data/es60/users/thomas_moore/clim_demo_results/daily/test_14032024/'
        if 'st_ocean' in var_monthly_climatology.coords:
            settings = {'chunksizes':(1,10,1500,3600)}
        else:
            settings = {'chunksizes':(12,1500,3600)}    
        encoding = {var: settings for var in BRAN2020_var_climatology.data_vars}
        BRAN2020_var_climatology.to_netcdf(write_path+'BRAN2020_daily_'+var+'_climatology.nc', encoding = encoding)







        client.restart()

    # -------------
    logger.info("done with base zarr for all vars")
if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
