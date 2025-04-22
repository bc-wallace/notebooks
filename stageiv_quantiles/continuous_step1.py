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

def get_size(infile):
    return xr.open_mfdataset(infile).nbytes/1e9

if __name__ == "__main__":
    import dask
    from distributed import Client

    dask.config.set({'distributed.dashboard.link':'https://jupyter.alcf.anl.gov/theta/user/{USER}/proxy/{port}/status'})


    client = Client(n_workers=15,threads_per_worker=1)
    
    siv_folders = glob('/eagle/climate_severe/observations/stageiv/*')
    siv_folders.sort()

    all_files=[]
    for f in siv_folders:
        files_in_folder = glob(f+'/*.nc')
        files_in_folder.sort()
        for fif in files_in_folder:
            all_files.append(fif)
            
    result=[]
    for f in all_files:
        result.append(delayed(get_size)(f))

    all_filesizes = np.array(dask.compute(result)[0])
    total_filesize = all_filesizes.sum()
    
    file_step=round(len(all_files)/(total_filesize/100),0)

    interval=np.arange(0,len(all_files),int(file_step))
    for i,start in enumerate(interval):
        if i<len(interval)-1:
            print(start,interval[i+1])
        else:
            print(start,'Until end')
            
    for i,start in enumerate(interval):
        if i<len(interval)-1:
            print(start,interval[i+1])
            start_pos=start
            end_pos=interval[i+1]
        else:
            print(start,'Until end')
            start_pos=start
            end_pos=None

        mfds=xr.open_mfdataset(all_files[start_pos:end_pos],combine='nested',concat_dim='time')
        mfds = mfds.drop_vars(['lat','lon','p01m_status','time_bnds'])

        target_chunks={
            'p01m':{'time':len(mfds.time),'y':100,'x':100},
            'time': None,
            'lat': None,
            'lon': None,
        }

        target_store='/eagle/climate_severe/bwallace_scratch/zarr_stageiv_temp/target_'+str(i).zfill(2)
        temp_store='/eagle/climate_severe/bwallace_scratch/zarr_stageiv_temp/tmp_'+str(i).zfill(2)

        max_mem='1GB'

        array_plan=rechunk(mfds,target_chunks,max_mem,target_store,temp_store=temp_store)

        array_plan.execute()

        #clear some memory out between loops - not needed but want to see if
        #this can supress some of the garbage collection warnings
        sleep(2)
        client.restart()
        sleep(5)
    
    