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

def return_corrected_dataset(infile):    
    ds=xr.open_dataset(infile)['p01m']
    
    fileDate=datetime.strptime(infile.split('stageiv')[-1][:-3],'%Y%m%d')
    expectedDates=np.arange(fileDate,fileDate+timedelta(days=1),timedelta(hours=1))
    missingDates=np.where(~np.in1d(expectedDates,ds.time))[0]
    
    if missingDates.shape[0]>0:
        nanArr=np.zeros([missingDates.shape[0],ds.shape[1],ds.shape[2]])
        nanArr[:]=np.nan
        xarrayFill=xr.DataArray(nanArr,name='p01m',
                     dims=ds.dims,
                     coords={'time':expectedDates[missingDates].astype(datetime)})
        return xr.concat((ds,xarrayFill),dim='time').sortby('time').to_dataset()
    else:
        return ds.copy().to_dataset()

def process_multifile(multifiles,multioffsets,zarr_path,full_shape):  
    for file,offset_in in zip(multifiles,multioffsets):
        dataset=return_corrected_dataset(file)
        
        variable_list=['p01m','time']
        group=zarr.group(zarr_path)
        
        for v in variable_list:
            slices=da.core.slices_from_chunks(da.empty_like(dataset[v]).chunks)
        
            for slice_ in slices:
                time_slice,*rest=slice_
                time_slice=slice(time_slice.start+offset_in,time_slice.stop+offset_in)
                target_slice=(time_slice,)+tuple(rest)
        
                if v!='time':
                    group[v][target_slice]=dataset[v][slice_].values
                else:
                    tdelta=(dataset[v][slice_].values-np.datetime64('1980-01-01T00:00:00')).astype('timedelta64[m]')
                    group[v][target_slice]=tdelta

                    
if __name__ == '__main__':
    jid=sys.argv[1]
    print(jid)
    print(type(jid))
    jid_int=int(jid)

    geog=xr.open_mfdataset('/eagle/climate_severe/geog/geo_em.d01.nc')
    dummy_shape=geog.HGT_M[0].shape

    from dask.distributed import Client
    
    client = Client(n_workers=32,threads_per_worker=1)

    siv_folders = glob('/eagle/climate_severe/observations/stageiv/*')
    siv_folders.sort()

    all_files=[]
    for f in siv_folders:
        files_in_folder = glob(f+'/*.nc')
        files_in_folder.sort()
        for fif in files_in_folder:
            all_files.append(fif)

    geog = xr.open_mfdataset(all_files[0])
    dummy_shape = geog.p01m[0].shape #getting the geographic bounds
    
    #set zarr storage location
    zarr_store='/eagle/climate_severe/bwallace_scratch/data_request/STAGEIV/STAGEIV'

    npList=np.array(all_files)
    
    time_dims=np.array([1]*(npList.shape[0]*24))
    full_shape=(time_dims.sum(),)+(dummy_shape)
    offsets=np.cumsum(time_dims)
    offsets-=offsets[0]
    
    listChunks=np.array_split(npList,15)
    offsetChunks=np.array_split(offsets,15)
    
    #batch computations into groups that match chunksize
    #IMPORTANT! Dask workers would overwrite timestamps
    #in large chunks. Way around this is to have the
    #number of files processed by each worker equal
    #to the number of files in a chunk. Basically,
    #each worker takes care of all the files in a
    #chunk.
    batch=[]
    for i,zs in enumerate(zip(listChunks[jid_int],offsetChunks[jid_int])):
        f,o=zs[0],zs[1]                
        batch.append(delayed(process_multifile)([f],[o],zarr_store,full_shape))
        files_to_write.append(f)
        
        if len(batch)==50:
            dask.compute(batch)
            batch=[]            
    
            
    if batch:
        dask.compute(batch)