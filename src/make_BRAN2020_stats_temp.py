# ///////////////////////
# make_BRAN2020_stats_temp.py
# 10 April 2024
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
import datetime

def main():
    """
    spin up cluster & do the work   
    """
    import warnings
    warnings.filterwarnings('ignore')
    logger = logging.getLogger(__name__)
    logger.info("Spinning up a dask cluster")
    # -----------  cluster -----------------------
    from dask.distributed import Client
    import dask
    import distributed

    with dask.config.set({"distributed.scheduler.worker-saturation": 1.0,
                      "distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0,
                    "logging.distributed'": "error"}):
        client = distributed.Client()
        print(client)
    ### masks for ENSO composites
    logger.info("ENSO masking")
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
    #
    El_Nino_Series = ONI_DF.groupby('El Nino event group ID')['ONI'].filter(lambda x: len(x) >= 5,dropna=False).where(ONI_DF['El Nino threshold'] == True)
    ONI_DF = pd.concat([ONI_DF, El_Nino_Series.rename('El Nino')], axis=1)
    La_Nina_Series = ONI_DF.groupby('La Nina event group ID')['ONI'].filter(lambda x: len(x) >= 5,dropna=False).where(ONI_DF['La Nina threshold'] == True)
    ONI_DF = pd.concat([ONI_DF, La_Nina_Series.rename('La Nina')], axis=1)

    ### run var on what variable
    var_name = 'temp'
    #var_name = 'mld'
    #var_name = 'eta_t'
    logger.info("starting variable: "+var_name)
    #
    zarr_path = '/scratch/es60/ard/reanalysis/BRAN2020/ARD/'
    path_dict = {'eta_t':'BRAN2020-daily-eta_t-chunk4time-v14032024.zarr',
                    'mld':'BRAN2020-daily-mld-chunk4time-v04042024.zarr',
                    'temp':'BRAN2020-daily-temp-chunk4time-v07022024.zarr'}
    depth_dict = {'eta_t':None,'mld':None,'temp':'st_ocean'}
    lon_dict = {'eta_t':'xt_ocean','mld':'xt_ocean','temp':'xt_ocean'}
    lat_dict = {'eta_t':'yt_ocean','mld':'yt_ocean','temp':'yt_ocean'}
    time_dim = 'Time'
    results_path = '/g/data/es60/users/thomas_moore/clim_demo_results/daily/draft_delivery/'
    results_file = 'BRAN2020_clim_demo_'+var_name+'.nc'
    collection_path = zarr_path + path_dict[var_name]
    #
    ds = xr.open_zarr(collection_path,consolidated=True)
    clim_ds = xr.merge([ds.groupby(time_dim+'.month').mean(dim=time_dim,engine='flox',method='cohorts').rename({var_name:'mean_'+var_name}),
                        ds.groupby(time_dim+'.month').min(dim=time_dim,engine='flox',method='cohorts').rename({var_name:'min_'+var_name}),
                        ds.groupby(time_dim+'.month').max(dim=time_dim,engine='flox',method='cohorts').rename({var_name:'max_'+var_name}),
                        ds.groupby(time_dim+'.month').std(dim=time_dim).rename({var_name:'std_'+var_name}),
                        ds.groupby(time_dim+'.month').median(dim=time_dim).rename({var_name:'median_'+var_name})
    ])
    quant = ds.groupby(time_dim+'.month').quantile([0.05,0.95],skipna=False,dim=time_dim)
    quant_ds = xr.merge([quant.isel(quantile=0).reset_coords(drop=True).rename({var_name:'quantile_05_'+var_name}),quant.isel(quantile=1).reset_coords(drop=True).rename({var_name:'quantile_95_'+var_name})])
    result_ds = xr.merge([clim_ds,quant_ds])
    ### ENSO composites
    # filter BRAN2020 data by ENSO
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
    #
    El_Nino_mask = ONI_DF_BRANtime['El Nino LOGICAL']
    El_Nino_mask = El_Nino_mask.to_xarray()
    El_Nino_mask = El_Nino_mask.rename({'datetime':'Time'})
    sync_Time = ds.Time
    El_Nino_mask['Time'] = sync_Time
    #
    La_Nina_mask = ONI_DF_BRANtime['La Nina LOGICAL']
    La_Nina_mask = La_Nina_mask.to_xarray()
    La_Nina_mask = La_Nina_mask.rename({'datetime':'Time'})
    sync_Time = ds.Time
    La_Nina_mask['Time'] = sync_Time
    #
    ONI_DF_BRANtime['Neutral LOGICAL'] = (ONI_DF_BRANtime['El Nino LOGICAL'] == False) & (ONI_DF_BRANtime['La Nina LOGICAL'] == False)
    ### mask out data
    El_Nino_ds = ds.where(El_Nino_mask)
    La_Nina_ds = ds.where(La_Nina_mask)
    ##### El Nino calc
    clim_El_Nino_ds = xr.merge([El_Nino_ds.groupby(time_dim+'.month').mean(dim=time_dim,engine='flox',method='cohorts').rename({var_name:'mean_'+'el_nino_'+var_name}),
                        El_Nino_ds.groupby(time_dim+'.month').min(dim=time_dim,engine='flox',method='cohorts').rename({var_name:'min_'+'el_nino_'+var_name}),
                        El_Nino_ds.groupby(time_dim+'.month').max(dim=time_dim,engine='flox',method='cohorts').rename({var_name:'max_'+'el_nino_'+var_name}),
                        El_Nino_ds.groupby(time_dim+'.month').std(dim=time_dim).rename({var_name:'std_'+'el_nino_'+var_name}),
                        El_Nino_ds.groupby(time_dim+'.month').median(dim=time_dim).rename({var_name:'median_'+'el_nino_'+var_name})
    ])
    quant_El_Nino = El_Nino_ds.groupby(time_dim+'.month').quantile([0.05,0.95],skipna=False,dim=time_dim)
    quant_El_Nino_ds = xr.merge([quant.isel(quantile=0).reset_coords(drop=True).rename({var_name:'quantile_05_'+'el_nino_'+var_name}),quant.isel(quantile=1).reset_coords(drop=True).rename({var_name:'quantile_95_'+'el_nino_'+var_name})])
    result_El_Nino_ds = xr.merge([clim_El_Nino_ds,quant_El_Nino_ds])
    #### La Nina calc
    clim_La_Nina_ds = xr.merge([La_Nina_ds.groupby(time_dim+'.month').mean(dim=time_dim,engine='flox',method='cohorts').rename({var_name:'mean_'+'la_nina_'+var_name}),
                        La_Nina_ds.groupby(time_dim+'.month').min(dim=time_dim,engine='flox',method='cohorts').rename({var_name:'min_'+'la_nina_'+var_name}),
                        La_Nina_ds.groupby(time_dim+'.month').max(dim=time_dim,engine='flox',method='cohorts').rename({var_name:'max_'+'la_nina_'+var_name}),
                        La_Nina_ds.groupby(time_dim+'.month').std(dim=time_dim).rename({var_name:'std_'+'la_nina_'+var_name}),
                        La_Nina_ds.groupby(time_dim+'.month').median(dim=time_dim).rename({var_name:'median_'+'la_nina_'+var_name})
    ])
    quant_La_Nina = La_Nina_ds.groupby(time_dim+'.month').quantile([0.05,0.95],skipna=False,dim=time_dim)
    quant_La_Nina_ds = xr.merge([quant.isel(quantile=0).reset_coords(drop=True).rename({var_name:'quantile_05_'+'la_nina_'+var_name}),quant.isel(quantile=1).reset_coords(drop=True).rename({var_name:'quantile_95_'+'la_nina_'+var_name})])
    result_La_Nina_ds = xr.merge([clim_La_Nina_ds,quant_La_Nina_ds])
    #
    result_ds = xr.merge([result_ds,result_El_Nino_ds,result_La_Nina_ds])
    result_ds.to_netcdf(results_path+results_file,engine='netcdf4')
    # Write log file
    log_path = '/scratch/es60/ard/reanalysis/BRAN2020/ARD/logs/'
    log_file = log_path + f'log_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}_{var_name}_stats.txt'
    with open(log_file, 'a') as f:
        f.write(f'{var_name} processing finished\n')
    f.close()
    logger.info("done with stats run for "+var_name)
if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
