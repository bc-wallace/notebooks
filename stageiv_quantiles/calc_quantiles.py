import numpy as np
import xarray as xr
import pandas as pd

import dask
import dask.array as da
from dask import delayed

from matplotlib.path import Path
import matplotlib.ticker as mticker

from glob import glob

from datetime import datetime, timedelta
import time

import zarr
import sys

if __name__ == '__main__':
    
    from dask.distributed import Client

    client = Client(n_workers=32,threads_per_worker=1)
    
    ds = xr.open_zarr('/eagle/climate_severe/bwallace_scratch/COARSENED/historical/continuous_chunks/AFWA_TOTPRECIP')
    
    #array([0.1  , 0.2  , 0.3  , 0.4  , 0.5  , 0.6  , 0.7  , 0.8  , 0.9  ,
    #       0.91 , 0.92 , 0.93 , 0.94 , 0.95 , 0.96 , 0.97 , 0.98 , 0.99 ,
    #       0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999])
    quantiles = np.concatenate((np.arange(0.1, 1, 0.1), np.arange(0.91, 1.0, 0.01), np.arange(0.991, 0.999, 0.001)))
    

    season_nums = [np.arange(1,13), [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    season_names = ['ALL', 'DJF', 'MAM', 'JJA', 'SON']

    for snum, sname in zip(season_nums, season_names):
        index_ds = ds['AFWA_TOTPRECIP'].sel(Time = np.isin(ds['Time.month'], snum))
        
        #use below line if you want to exclude values below a certain threshold from the calculation (i.e. wet day calculation)
        index_ds = index_ds.where(index_ds >= 0.254)
        
        
        all_result = index_ds.quantile(quantiles, dim='Time').compute()
        all_result.to_netcdf('/eagle/climate_severe/bwallace_scratch/quantiles/historical_'+sname+'_quantiles_gt0p254.nc')
        
        final = []
        for hr in np.arange(0,24):
            for mn in [0, 15, 30, 45]:
                result = index_ds.sel(Time = ((np.isin(index_ds['Time.hour'], hr)) & (np.isin(index_ds['Time.minute'], mn)))).quantile(quantiles, dim='Time').compute()

                result = result.assign_coords({'diurnal':str(hr).zfill(2)+str(mn).zfill(2)})
                final.append(result)

        ds_to_save = xr.concat(final, dim='diurnal')
        ds_to_save.to_netcdf('/eagle/climate_severe/bwallace_scratch/quantiles/historical_'+sname+'_quantiles_diurnal_gt0p254.nc')
