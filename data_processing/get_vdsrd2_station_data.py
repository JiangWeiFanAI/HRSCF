import sys
sys.path.append('../')
import util.PrepareData as pdata
import util.data_processing_tool as dpt

from datetime import date
from netCDF4 import Dataset
import numpy as np
import netCDF4 as nc
from datetime import timedelta, date, datetime
import platform 
import torch
from torchvision import datasets, models, transforms
import os
from PIL import Image
import util.constant_param as consparam

from model import vdsrd2


def get_filename_with_no_time_order(rootdir):
    '''get filename first and generate label '''
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
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
            access_path=rootdir+var_name+'/daily/'+en+"/"+"da_"+var_name+"_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#             print(access_path)
            if os.path.exists(access_path):
#                 print(access_path)
                path=[access_path]
                path.append(en)
                path.append(date)
                _files.append(path)

#最后去掉第一行，然后shuffle
#     if nine2nine and date_minus_one==1:
#         del _files[0]
    return _files

def to01(lr):
    return (lr-np.min(lr))/(np.max(lr)-np.min(lr))



#50 station
def get_access_SR_data():# 顺便保存 BI数据
    
    
    lr_transforms = transforms.Compose([
    transforms.Resize((316, 376)),
#     transforms.RandomResizedCrop(IMG_SIZE),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(30),
    transforms.ToTensor()
#     transforms.Normalize(IMG_MEAN, IMG_STD)
    ])
    
    
    import netCDF4 as nc
#     ensemble=['e01','e02']
    sys = platform.system()
    init_date=date(1970, 1, 1)
    start_date=date(2012, 1, 1)
    end_date=date(2012,12,25) #if 929 is true we should substract 1 day  
    if sys == "Windows":
        init_date=date(1970, 1, 1)
        start_date=date(2012, 1, 1)
        end_date=date(2012,12,25) #if 929 is true we should substract 1 day   
        file_ACCESS_dir="H:/climate/access-s1/" 
        file_BARRA_dir="D:/dataset/accum_prcp/"
    #         args.file_ACCESS_dir="E:/climate/access-s1/"
    #         args.file_BARRA_dir="C:/Users/JIA059/barra/"
        file_DEM_dir="../DEM/"
    else:
        file_ACCESS_dir_pr="/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
        file_ACCESS_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/"
        # training_name="temp01"
        file_BARRA_dir="/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/"

        
    ensemble=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']

    var_name="pr"
    dates=[start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]
    file_list=get_filename_with_time_order(file_ACCESS_dir,ensemble,dates,var_name)
    file_zg_list=get_filename_with_time_order(file_ACCESS_dir,ensemble,dates,'zg')

#             lr_zg=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"zg")[-3,82:144,134:188]

    time_leading=217

    lat_name="lat"
    lon_name="lon"

    for i,k in zip(file_list,file_zg_list):
#         print(i)
        data = Dataset(i[0], 'r')
        data_zg = Dataset(k[0], 'r')
        var = data[var_name][:,82:144,134:188]*86400
        zg_var = data_zg['zg'][:,-3,82:144,134:188]

        lr_zg=to01(zg_var)
        lr_zg=lr_zg*np.max(var)

        lat = data[lat_name][:][82:144]
        lon = data[lon_name][:][134:188]
#         print(var.shape)
        data.close()
    #         lr=dpt.read_access_data(i,idx=time_leading).data[82:144,134:188]*86400
        result= torch.zeros((217,1,316,376),dtype=torch.float32)

        for idx,j in enumerate(var):
            if idx>= 217:
                break
            lr=dpt.interp_tensor_2d(j,(79,94))
            result[idx]=lr_transforms(Image.fromarray(lr))
            
        result_zg= torch.zeros((217,1,316,376),dtype=torch.float32)
        for idx,j in enumerate(lr_zg):
            if idx>= 217:
                break
            result_zg[idx]=lr_transforms(Image.fromarray(j))            


        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net_state=torch.load("../save/model/zg_val2012/best_test_20.pth",map_location=device)['model'].module.state_dict()
        #net_state=torch.load("../save/model/zg_val10/best_test_50.pth",map_location=device)['model'].module.state_dict()
        net = vdsrd2().to(device)
        net.load_state_dict(net_state)

#         net=torch.load('../save/model/vdsr_pr/best_test.pth',map_location=device)
        def prepare( l, device=False):
            def _prepare(tensor):
                tensor = tensor.float()
                return tensor.to(device)
            return [_prepare(_l) for _l in l]
        pr,zg=prepare([result,result_zg],device)


        with torch.set_grad_enabled(False):
            sr = net(pr,zg)
        sr_np=np.squeeze(sr.cpu().numpy())
        
        data_50=[]
        station_name=[]
        for station in consparam.station_50_index_for_size_of_hr_sr_station_code.keys():
            idx_i=consparam.station_50_index_for_size_of_hr_sr_station_code[station][0]
            idx_j=consparam.station_50_index_for_size_of_hr_sr_station_code[station][1]
            data_50.append(sr_np[:,idx_i,idx_j])
            station_name.append(station)
        data_50=np.array(data_50)
        print(data_50.shape)
        station_name=np.array(station_name)
        
        if not os.path.exists('../Data/'+var_name+'/daily_vdsrd2_50station/'+i[1]):
            os.makedirs('../Data/'+var_name+'/daily_vdsrd2_50station/'+i[1])
        
        f_w = nc.Dataset('../Data/'+var_name+'/daily_vdsrd2_50station/'+i[1]+"/da_"+var_name+"_"+i[2].strftime("%Y%m%d")+"_"+i[1]+'.nc','w',format = 'NETCDF4')

        f_w.createDimension('time',time_leading)
        f_w.createDimension('station',50)

        f_w.createVariable('station',np.int32,('station'))
        f_w.createVariable('time',np.int,('time'))



        f_w.variables['station'][:] = station_name
        f_w.variables['time'][:] =  np.array(range(1,218))


        f_w.createVariable( 'pr', np.float32, ('station','time'))
        f_w.variables['pr'][:] = data_50

        f_w.close()
            

            
if __name__=='__main__':
    get_access_SR_data()