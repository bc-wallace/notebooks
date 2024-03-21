import numpy as np
import xarray as xr
import pandas as pd

import dask
import dask.array as da
from dask import delayed

from glob import glob

from datetime import datetime, timedelta
import time

import zarr
import sys

def return_corrected_dataset(infile):
    '''
    Given a path leading to an xarray dataset, ensure that the number of expected times within the file are there. 
    If not, fill in the appropriate missing times with np.nan arrays.
    
    Parameters
    ----------------

    infile: filepath leading to an xarray compatible dataset

    Returns:

     xarray.Dataset object 
    
    '''
    ds=xr.open_dataset(infile)['AFWA_TOTPRECIP']
    
    fileDate=datetime.strptime(infile.split('AFWA_TOTPRECIP_')[-1][:-3],'%Y-%m-%d')
    expectedDates=np.arange(fileDate,fileDate+timedelta(days=1),timedelta(minutes=15))
    missingDates=np.where(~np.in1d(expectedDates,ds.Time))[0]
    
    if missingDates.shape[0]>0:
        nanArr=np.zeros([missingDates.shape[0],ds.shape[1],ds.shape[2]])
        nanArr[:]=np.nan
        xarrayFill=xr.DataArray(nanArr,name='AFWA_TOTPRECIP',
                     dims=ds.dims,
                     coords={'Time':expectedDates[missingDates].astype(datetime)})
        return xr.concat((ds,xarrayFill),dim='Time').sortby('Time').to_dataset()
    else:
        return ds.copy().to_dataset()

def process_multifile(multifiles,multioffsets,zarr_path,full_shape):  
    '''
    Given a list of files and their offsets relative to the entire dataset, write these files to a zarr store

    Parameters
    ----------------

    multifiles: filepaths leading to xarray datasets
    multioffsets: offset of file relative to full dataset. This points each dask worker to the right spot within the zarr array to insert the file
    -> e.g. 10 files with shape (96, 50, 50) appended along dim0 would yield total array size of (960, 50, 50). File 0 would start at 0, File 1 would start at 96, File 2 would start at 192, ... File 10 would start at 864.
    zarr_path: path to the zarr that is being written to
    full_shape: the total shape of the zarr array

    Returns:

     None (just writes files to a zarr)
    
    '''

    #for each file, offset beingn given
    for file,offset_in in zip(multifiles,multioffsets):
        #correct the dataset (ensure that dim has shape 96. if it doesn't, fill missing times with np.nans)
        dataset=return_corrected_dataset(file)
        
        #variable list (should match zarr initialization file)
        variable_list=['AFWA_TOTPRECIP','Time']
        #create a zarr group using provided path to zarr
        group=zarr.group(zarr_path)
        
        #loop through each var
        for v in variable_list:
            slices=da.core.slices_from_chunks(da.empty_like(dataset[v]).chunks)
        
            for slice_ in slices:
                time_slice,*rest=slice_
                time_slice=slice(time_slice.start+offset_in,time_slice.stop+offset_in)
                target_slice=(time_slice,)+tuple(rest)
        
                #if var is not time, insert dataset into appropriate slot
                if v!='Time':
                    group[v][target_slice]=dataset[v][slice_].values
                #need to do time formatting for the Time var (minutes since 1980/01/01)
                else:
                    tdelta=(dataset[v][slice_].values-np.datetime64('1980-01-01T00:00:00')).astype('timedelta64[m]')
                    group[v][target_slice]=tdelta



if __name__ == '__main__':
    jid=sys.argv[1]
    print(jid)
    print(type(jid))
    jid_int=int(jid)

    sim='end_of_century_8p5'

    geog=xr.open_mfdataset('/eagle/climate_severe/geog/geo_em.d01.nc')
    dummy_shape=geog.HGT_M[0].shape

    from dask.distributed import Client
    
    client = Client(n_workers=32,threads_per_worker=1) #restriction for polaris - multithreaded throws errors

    #set zarr storage location
    zarr_store='/eagle/climate_severe/bwallace_scratch/COARSENED/'+sim+'/AFWA_TOTPRECIP'

    #establish path to pull data from
    base_path='/eagle/climate_severe/Derived_Subsets/AFWA_TOTPRECIP/'+sim+'/'
    #gather up the water year folders
    basePathFiles=glob(base_path+'*')
    basePathFiles.sort()

    npList=np.array(basePathFiles)
    
    #this is important - we're assuming that each file has exactly 96 times within it.
    #the above function return_corrected_dataset ensures that this will always happen.
    #if file uniformity is not guaranteed, you'll need to manually calculate this by
    #looping through each file and recording the shape (probably).
    #can also change 96 for whatever other time value assuming files are uniformly sized
    time_dims=np.array([1]*(npList.shape[0]*96))
    #full shape of the zarr array is sum of time_dims plus tuple of the spatial dimensions
    full_shape=(time_dims.sum(),)+(dummy_shape)
    #offset between each file is (ideally) 96 hours! this will change depending on file dims!
    offsets=np.cumsum(np.array([96]*(npList.shape[0])))
    #first offset should be == 0 so we'll subtract entire array by first instance
    offsets-=offsets[0]
    
    #these next two lines are PBS job array stuff. we split the entire computation into 15 "chunks" that are
    #offloaded onto 15 different nodes
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
        
        if len(batch)==50:
            dask.compute(batch)
            batch=[]          
            print(f)
            
    
            
    if batch:
        dask.compute(batch)


