import sys
sys.path.append('../util')
import PrepareData as pada
import data_processing_tool as dpt

from datetime import date
from netCDF4 import Dataset
import numpy as np
import netCDF4 as nc




station_5={
     'WAGGA WAGGA AMO': [-35.1583, 147.4575],
     'MENINGIE': [-35.6902, 139.3375],
     'SURAT': [-27.1591, 149.0702],
     'MILDURA AIRPORT': [-34.2358, 142.0867],
     'ORANGE AGRICULTURAL INSTITUTE': [-33.3211, 149.0828],
}

station_5_idx={
#     坐标
    'WAGGA WAGGA AMO': (98, 176),
     'MENINGIE': (97, 167),
     'SURAT': (113, 178),
     'MILDURA AIRPORT': (100, 170),
     'ORANGE AGRICULTURAL INSTITUTE': (102, 178)}

# root_dir='../../data/'
root_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/"
var_name='pr'


for station in station_5_idx.keys():
    container=[]
    for en in ['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']:        
        ensamble=[]
        for i in range(1,13):
            date_time=date(2012,i,1)
            filename=root_dir+var_name+"/daily/"+en+"/"+"da_"+var_name+"_"+date_time.strftime("%Y%m%d")+"_"+en+".nc"
            data = Dataset(filename, 'r')
            
            var = data[var_name][:,station_5_idx[station][0],station_5_idx[station][1]]
#             var=np.zeros((224,324,432))[:,station_5_idx[station][0],station_5_idx[station][1]]
    
            ensamble.append(var)
            data.close()
        container.append(ensamble)
    container=np.array(container)
    shape=container.shape
    f_w = nc.Dataset('../save/5_station_nc_data/'+station+'.nc','w',format = 'NETCDF4')
    f_w.createDimension('leading_time',shape[2])
    f_w.createDimension('month',shape[1])
    f_w.createDimension('ensembles',shape[0])

    f_w.createVariable('leading_time',np.int,('leading_time'))
    f_w.createVariable('month',np.int,('month'))
    f_w.createVariable('ensembles',np.int,('ensembles'))



    f_w.variables['leading_time'][:] = np.arange(1,shape[2]+1)
    f_w.variables['month'][:] =  np.arange(1,shape[1]+1)
    f_w.variables['ensembles'][:] =  np.arange(1,shape[0]+1)


    f_w.createVariable( var_name, np.float32, ('ensembles','month','leading_time'))
    f_w.variables[var_name][:] = container

    f_w.close()
        

#     print(dpt.read_barra_data_fc("../data/barra_aus/",demo_date)[78,315])

#  'WAGGA WAGGA AMO': [78, 315],