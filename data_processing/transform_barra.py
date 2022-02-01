#Demo
import os
sys.path.append('../')
import util.PrepareData as pdata
import util.data_processing_tool as dpt
from datetime import timedelta, date, datetime
import numpy as np
import time
import xarray as xr

import netCDF4 as nc
from netCDF4 import Dataset, num2date, date2num
import platform 


def get_filename_with_no_time_order(rootdir):
    '''get filename first and generate label '''
    _files = []
    list = os.listdir(rootdir) 
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i],)
        if os.path.isdir(path):
            _files.extend(get_filename_with_no_time_order(path))
        if os.path.isfile(path):
            if path[-3:]==".nc":
                _files.append(path)
    return _files


def get_filename_with_time_order(rootdir,ensemble,dates,var_name):
    '''get filename first and generate label ,one different w'''
    _files = []
    for en in ensemble:
        for date in dates:
            access_path=rootdir+en+"/"+"da_"+var_name+"_"+date.strftime("%Y%m%d")+"_"+en+".nc"
            if os.path.exists(access_path):
                path=[access_path]
                path.append(en)
                path.append(date)
                _files.append(path)

    return _files



def main():
    sys = platform.system()
    init_date=date(1970, 1, 1)
    start_date=date(1990, 1, 1)
    end_date=date(2012,12,25) 

    file_ACCESS_dir_pr="/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
    file_ACCESS_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/"
    file_BARRA_dir="/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/"


    dates=[start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]
    for i in dates:
        data_high=dpt.read_barra_data_fc(file_BARRA_dir,i,nine2nine=True)
        label=dpt.map_aust(data_high,domain=[112.9, 154.25, -43.7425, -9.0])#,domain=domain)

        f_w = nc.Dataset('../data/barra_aus/'+i.strftime("%Y%m%d")+'.nc','w',format = 'NETCDF4')
        f_w.createDimension('lat',len(label["lat"].data))
        f_w.createDimension('lon',len(label["lon"].data))

        f_w.createVariable('lat',np.float32,('lat'))
        f_w.createVariable('lon',np.float32,('lon'))


        f_w.variables['lat'][:] = label["lat"].data
        f_w.variables['lon'][:] = label["lon"].data

        f_w.createVariable( 'barra', np.float32, ('lat','lon'))
        f_w.variables['barra'][:] = label.data

        f_w.close()

if __name__=='__main__':
    main()
