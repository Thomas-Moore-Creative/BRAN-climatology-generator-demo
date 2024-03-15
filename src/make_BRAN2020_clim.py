# ///////////////////////
# make_BRAN2020_stats.py
# 15 March 2024
#////////////////////////
# --------- packages --------------
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import xarray as xr
import pandas as pd
import numpy as np
import pandas as pd


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
    
    # -------------- setup -------------------
    logger.info("setting up")
    ARD_dir = '/scratch/es60/ard/reanalysis/BRAN2020/ARD/test_14032024/'
    var_request_list = ['eta_t','mld']
    # ------
    logger.info("starting stats workflow step")
    # -------- run over variables -----
    logger.info("for loop over requested vars")
    for var in var_request_list:
        logger.info("variable: "+var)
        ard_file_ID = 'BRAN2020-daily-'+var+'-v14032024.zarr'
        ard_rcTime_file_ID = 'BRAN2020-daily-'+var+'-chunk4time-v14032024.zarr'
        DS = xr.open_zarr(ARD_dir+ard_file_ID,consolidated=True)
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
    
    
    # mask events in both space and time chunked versions
        El_Nino_mask_0_1 = El_Nino_mask.to_dataframe().replace({True: 1, False: 0}).to_xarray()
        La_Nina_mask_0_1 = La_Nina_mask.to_dataframe().replace({True: 1, False: 0}).to_xarray()
        El_Nino_mask_TIMES = El_Nino_mask_0_1['Time'].where(El_Nino_mask_0_1,drop=True)
        La_Nina_mask_TIMES = La_Nina_mask_0_1['Time'].where(La_Nina_mask_0_1,drop=True)
        El_Nino_var_chunked_time = DS_rcTime.sel({'Time':El_Nino_mask_TIMES.Time})
        La_Nina_var_chunked_time = DS_rcTime.sel({'Time':La_Nina_mask_TIMES.Time})

        El_Nino_var_chunked_time = DS_rcTime.sel({'Time':El_Nino_mask_TIMES.Time})
        La_Nina_var_chunked_time = DS_rcTime.sel({'Time':La_Nina_mask_TIMES.Time})
    # Mean, Median, Max , Min, Std, 05 & 95 quantiles
        El_Nino_mean = El_Nino_var_chunked_time.mean('Time')
        El_Nino_median = El_Nino_var_chunked_time.median('Time')
        El_Nino_max = El_Nino_var_chunked_time.max('Time')
        El_Nino_min = El_Nino_var_chunked_time.min('Time')
        El_Nino_std = El_Nino_var_chunked_time.std('Time')
        El_Nino_quant = El_Nino_var_chunked_time.quantile([0.05,0.95],skipna=False,dim='Time')
        La_Nina_mean = La_Nina_var_chunked_time.mean('Time')
        La_Nina_median = La_Nina_var_chunked_time.median('Time')
        La_Nina_max = La_Nina_var_chunked_time.max('Time')
        La_Nina_min = La_Nina_var_chunked_time.min('Time')
        La_Nina_std = La_Nina_var_chunked_time.std('Time')
        La_Nina_quant = La_Nina_var_chunked_time.quantile([0.05,0.95],skipna=False,dim='Time')
        mean = var_chunked_time.mean('Time')
        median = var_chunked_time.median('Time')
        max = var_chunked_time.max('Time')
        min = var_chunked_time.min('Time')
        std = var_chunked_time.std('Time')
        quant = var_chunked_time.quantile([0.05,0.95],skipna=False,dim='Time')
    # make objects
        mean = mean.rename({var:'mean_'+var})
        median = median.rename({var:'median_'+var})
        max = max.rename({var:'max_'+var})
        min = min.rename({var:'min_'+var})
        std = std.rename({var:'std_'+var})
        quant = quant.rename({var:'quantile_'+var})
    #El_Nino_
        El_Nino_mean = El_Nino_mean.rename({var:'El_Nino_mean_'+var})
        El_Nino_median = El_Nino_median.rename({var:'El_Nino_median_'+var})
        El_Nino_max = El_Nino_max.rename({var:'El_Nino_max_'+var})
        El_Nino_min = El_Nino_min.rename({var:'El_Nino_min_'+var})
        El_Nino_std = El_Nino_std.rename({var:'El_Nino_std_'+var})
        El_Nino_quant = El_Nino_quant.rename({var:'El_Nino_quantile_'+var})
    #La_Nina_
        La_Nina_mean = La_Nina_mean.rename({var:'La_Nina_mean_'+var})
        La_Nina_median = La_Nina_median.rename({var:'La_Nina_median_'+var})
        La_Nina_max = La_Nina_max.rename({var:'La_Nina_max_'+var})
        La_Nina_min = La_Nina_min.rename({var:'La_Nina_min_'+var})
        La_Nina_std = La_Nina_std.rename({var:'La_Nina_std_'+var})
        La_Nina_quant = La_Nina_quant.rename({var:'La_Nina_quantile_'+var})
    #
        BRAN2020_stats = xr.merge([mean,median,max,min,std,
                                      El_Nino_mean,El_Nino_median,El_Nino_max,El_Nino_min,El_Nino_std,
                                      La_Nina_mean,La_Nina_median,La_Nina_max,La_Nina_min,La_Nina_std])
    #
        if 'st_ocean' in BRAN2020_stats.coords:
            BRAN2020_stats_rc = BRAN2020_stats.chunk({'st_ocean':-1,'yt_ocean':-1,'xt_ocean':360})
        else:
            BRAN2020_stats_rc = BRAN2020_stats.chunk({'yt_ocean':-1,'xt_ocean':-1})
    #
        BRAN2020_quant = xr.merge([quant,
                                      El_Nino_quant,
                                      La_Nina_quant])
        BRAN2020_quant_rc = BRAN2020_quant.chunk({'xt_ocean':3600})
    # write out results in NetCDF
        logger.info(var+" writing nc files")
        write_path = '/g/data/es60/users/thomas_moore/clim_demo_results/daily/test_14032024/'
    #
        #if 'st_ocean' in BRAN2020_stats.coords:
        #    settings = {'chunksizes':(51,1500,360)}
        #else:
        #    settings = {'chunksizes':(1500,3600)}    
        #encoding = {var: settings for var in BRAN2020_stats_rc.data_vars}
        BRAN2020_stats_rc.to_netcdf(write_path+'BRAN2020_daily_'+var+'_stats.nc')
        logger.info(var+" finished writing stats nc file")
        #if 'st_ocean' in BRAN2020_stats.coords:
        #    settings = {'chunksizes':(2,51,100,3600)}
        #else:
        #    settings = {'chunksizes':(2,100,3600)}    
        #encoding = {var: settings for var in BRAN2020_quant_rc.data_vars}
        BRAN2020_quant_rc.to_netcdf(write_path+'BRAN2020_daily_'+var+'_quant.nc')
        logger.info(var+" finished stats and writing nc files")
    # -------------
    logger.info("done with all vars")
if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
