import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset
from datetime import datetime, timedelta, date
import os
import dask.array as da
from time import sleep

from glob import glob
from dask import delayed

import zarr
from rechunker import rechunk

if __name__ == "__main__":
    import dask
    from distributed import Client

    client = Client(n_workers=15,threads_per_worker=1)
    
    siv_folders = glob('/eagle/climate_severe/observations/stageiv/*')
    siv_folders.sort()

    
    zarrs = glob('/eagle/climate_severe/bwallace_scratch/zarr_stageiv_temp/*')
    zarrs.sort()

    zarrs_ds = xr.open_mfdataset(zarrs, engine='zarr')
    
    target_chunks={
        'p01m':{'Time':len(zarrs_ds.time),'y':20,'x':20},
        'time': None,
    }

    target_store='/eagle/climate_severe/bwallace_scratch/zarr_stageiv_temp/continuous_2002-2022_chunked_stageiv'
    temp_store='/eagle/climate_severe/bwallace_scratch/zarr_stageiv_temp/tmp_big_rechunk'

    max_mem='1.5GB'

    array_plan=rechunk(zarrs_ds,target_chunks,max_mem,target_store,temp_store=temp_store)

    array_plan.execute()