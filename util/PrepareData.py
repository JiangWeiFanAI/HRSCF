import os
import util.data_processing_tool as dpt
from datetime import timedelta, date, datetime
# import args_parameter as args
import torch,torchvision
import numpy as np
import random

from torch.utils.data import Dataset,random_split
from torchvision import datasets, models, transforms

import time
import xarray as xr
from PIL import Image
# from sklearn.model_selection import StratifiedShuffleSplit

# file_ACCESS_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
# file_BARRA_dir="/g/data/ma05/BARRA_R/analysis/acum_proc"

# ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
# ensemble=[]
# for i in range(args.ensemble):
#     ensemble.append(ensemble_access[i])
    
# ensemble=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']

# leading_time=217
# leading_time_we_use=31


# init_date=date(1970, 1, 1)
# start_date=date(1990, 1, 1)
# end_date=date(1990,12,31) #if 929 is true we should substract 1 day
# dates=[start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

# domain = [111.975, 156.275, -44.525, -9.975]

# domain = [111.975, 156.275, -44.525, -9.975]



class ACCESS_BARRA_Probabilistic(Dataset):
    '''

    read 11 probalistic forecast
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",lr_transform=None,hr_transform=None,shuffle=False,args=None):
#         print("=> BARRA_R & ACCESS_S1 loading")
#         print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.start_date = start_date
        self.end_date = end_date
        
        self.regin = regin
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
    
        if shuffle:
            random.shuffle(self.filename_list)
            
            
#         data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
#         self.lat=data_high[1]
#         self.lon=data_high[1]

#         self.data_dem=dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif")
#         self.data_dem=self.lr_transform(Image.fromarray(self.data_dem))
#         print(type(self.data_dem))
        
#             data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
#             self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for date in self.dates:
            for i in range(self.leading_time_we_use+1):
            

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
#                 access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                ensemble_path=[]
                for en in np.random.choice(self.ensemble_access,size=self.args.ensemble, replace=False):
                    access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                    if os.path.exists(access_path):
                        
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        ensemble_path.append(path)
                if os.path.exists(access_path):
                    _files.append(ensemble_path)

        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
#         t=time.time()
        
        #read_data filemame[idx]
        
        lr_ensemble=np.zeros((self.args.hr_size[0],self.args.hr_size[1],self.args.ensemble))
        for ix,i in enumerate(self.filename_list[idx]):
            en,access_date,barra_date,time_leading=i
            lr_ensemble[:,:,ix]=dpt.interp_tensor_2d(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr"),self.args.hr_size)
#             print(lr_ensemble[ix].shape)
    #         lr=np.expand_dims(lr,axis=2)
    #         lr=np.expand_dims(self.mapping(lr),axis=2)
    
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)
        return self.lr_transform(lr_ensemble),self.hr_transform(Image.fromarray(label)),torch.tensor(int(en[1:])),torch.tensor(int(access_date.strftime("%Y%m%d"))),torch.tensor(time_leading)


    
class ACCESS_BARRA_Probabilistic(Dataset):
    '''

    read 11 probalistic forecast
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",lr_transform=None,hr_transform=None,shuffle=False,args=None):
#         print("=> BARRA_R & ACCESS_S1 loading")
#         print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.start_date = start_date
        self.end_date = end_date
        
        self.regin = regin
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
    
        if shuffle:
            random.shuffle(self.filename_list)
            
            
#         data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
#         self.lat=data_high[1]
#         self.lon=data_high[1]

#         self.data_dem=dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif")
#         self.data_dem=self.lr_transform(Image.fromarray(self.data_dem))
#         print(type(self.data_dem))
        
#             data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
#             self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #åˆ—å‡ºæ–‡ä»¶å¤¹ä¸‹æ‰€æœ‰çš„ç›®å½•ä¸Žæ–‡ä»¶
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for date in self.dates:
            for i in range(self.leading_time_we_use+1):
            

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
#                 access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                ensemble_path=[]
                for en in self.ensemble:
                    access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                    if os.path.exists(access_path):
                        
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        ensemble_path.append(path)
                if os.path.exists(access_path):
                    _files.append(ensemble_path)

        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #å°†æ•°æ®æ˜ å°„åˆ°[-1,1]åŒºé—´ å³a=-1ï¼Œb=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
#         t=time.time()
        
        #read_data filemame[idx]
        
        lr_ensemble=np.zeros((self.args.hr_size[0],self.args.hr_size[1],self.args.ensemble))
        for ix,i in enumerate(self.filename_list[idx]):
            en,access_date,barra_date,time_leading=i
            lr_ensemble[:,:,ix]=dpt.interp_tensor_2d(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr"),self.args.hr_size)*86400
#             print(lr_ensemble[ix].shape)
    #         lr=np.expand_dims(lr,axis=2)
    #         lr=np.expand_dims(self.mapping(lr),axis=2)
    
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)
        return self.lr_transform(lr_ensemble),self.hr_transform(Image.fromarray(label)),torch.tensor(int(en[1:])),torch.tensor(int(access_date.strftime("%Y%m%d"))),torch.tensor(time_leading)



class ACCESS_AWAP_GAN(Dataset):

    def __init__(self, start_date, end_date, regin="AUS", lr_transform=None,
                 hr_transform=None, shuffle=True):
        print("=> ACCESS_S1 & AWAP loading")
        print("=> from " + start_date.strftime("%Y/%m/%d") + " to " + end_date.strftime("%Y/%m/%d") + "")
        self.file_ACCESS_dir = "G:/Dataset/access_60_masked/"
        self.file_AWAP_dir = "G:/Dataset/split_awap_masked/"

        # self.regin = regin
        self.start_date = start_date
        self.end_date = end_date

        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.leading_time_we_use = 7

        self.ensemble = ['e01', 'e02', 'e03', 'e04', 'e05', 'e06', 'e07', 'e08', 'e09', 'e10', 'e11']

        self.dates = self.date_range(start_date, end_date)

        self.filename_list = self.get_filename_with_time_order(self.file_ACCESS_dir)
        if not os.path.exists(self.file_ACCESS_dir):
            print(self.file_ACCESS_dir + "pr/daily/")
            print("no file or no permission")

        _, _, date_for_AWAP, time_leading = self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)

        # data_high = read_awap_data_fc_get_lat_lon(self.file_AWAP_dir, date_for_AWAP)
        # print("data_high")
        # print(data_high)
        # self.lat = data_high[1]
        # print(self.lat)
        # self.lon = data_high[1]
        # print(self.lon)
        # sshape = (79, 94)

    def __len__(self):
        return len(self.filename_list)

    def get_filename_with_no_time_order(self, rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(rootdir, list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:] == ".nc":
                    _files.append(path)
        return _files

    def get_filename_with_time_order(self, rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:
                access_path = rootdir + en + "/da_pr_" + date.strftime("%Y%m%d") + "_" + en + ".nc"
                #                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date == self.end_date and i == 1:
                            break
                        path = [en]
                        AWAP_date = date + timedelta(i)
                        path.append(date)
                        path.append(AWAP_date)
                        path.append(i)
                        _files.append(path)

        # 最后去掉第一行，然后shuffle
        return _files
    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    def mapping(self, X, min_val=0., max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        # 将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b - a) / (Xmax - Xmin) * (X - Xmin)
        return Y

    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t = time.time()

        # read_data filemame[idx]
        en, access_date, awap_date, time_leading = self.filename_list[idx]

        lr = dpt.read_access_data_wr(self.file_ACCESS_dir, en, access_date, time_leading, "pr")

        hr = dpt.read_awap_data(self.file_AWAP_dir, awap_date)


        return lr, hr, awap_date.strftime("%Y%m%d"), time_leading




class ACCESS_BARRA_cali(Dataset):
    '''


   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",lr_transform=None,hr_transform=None,shuffle=True,args=None):
#         print("=> BARRA_R & ACCESS_S1 loading")
#         print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.start_date = start_date
        self.end_date = end_date
        
        self.regin = regin
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"cali/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"cali/daily/"):
            print(args.file_ACCESS_dir+"clai/daily/")
            print("no file or no permission")
        
        
        _,_,date_for_BARRA,time_leading=self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)
        
        
#         if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
# #             print(self.file_BARRA_dir)
#             print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
        self.lat=data_high[1]
        self.lon=data_high[1]
#         self.shape=(316, 376)

        self.data_dem=dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif")
#         print(type(self.data_dem))
        
#             data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
#             self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for date in self.dates:
            for i in range(self.leading_time_we_use,self.leading_time_we_use+1):
            

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
#                 access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                for en in self.ensemble:
                    access_path=rootdir+en+"/"+"daq5_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                    if os.path.exists(access_path):
                        
                    
                        if date==self.end_date and i==1:
                            break
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)

#     def get_filename_with_time_order(self,rootdir):
#         '''get filename first and generate label ,one different w'''
#         _files = []
#         for date in self.dates:
#             for en in self.ensemble:

# #                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
#                 access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
# #                 print(access_path)
#                 if os.path.exists(access_path):
#                     for i in range(self.leading_time_we_use):
#                         if date==self.end_date and i==1:
#                             break
#                         path=[]
#                         path.append(en)
#                         barra_date=date+timedelta(i)
#                         path.append(date)
#                         path.append(barra_date)
#                         path.append(i)
#                         _files.append(path)                        

    #最后去掉第一行，然后shuffle
        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        en,access_date,barra_date,time_leading=self.filename_list[idx]
        

        lr=dpt.read_access_data_calibrataion(self.file_ACCESS_dir,en,access_date,time_leading,"pr")
#         lr=np.expand_dims(lr,axis=2)

#         lr=np.expand_dims(self.mapping(lr),axis=2)
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)

        if self.args.zg:
            lr_zg=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"zg"),axis=2)

        if self.args.psl:
            lr_psl=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"psl")

        if self.args.tasmax:
            lr_tasmax=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmax"),axis=2)


        if self.args.tasmin:
            lr_tasmin=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmin")
            
#         if self.args.channels==1:
#             lr=np.repeat(lr,3,axis=2)
        return np.array(lr),1,self.hr_transform(Image.fromarray(label)),torch.tensor(int(en[1:])),torch.tensor(int(access_date.strftime("%Y%m%d"))),torch.tensor(time_leading)
#         return self.lr_transform(Image.fromarray(lr)),self.lr_transform(Image.fromarray(self.data_dem)),self.hr_transform(Image.fromarray(label)),torch.tensor(int(en[1:])),torch.tensor(int(access_date.strftime("%Y%m%d"))),torch.tensor(time_leading)


class ACCESS_BARRA_crps(Dataset):
    '''

   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",lr_transform=None,hr_transform=None,shuffle=True,args=None):
#         print("=> BARRA_R & ACCESS_S1 loading")
#         print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.start_date = start_date
        self.end_date = end_date
        
        self.regin = regin
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
#         if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
#             print(args.file_ACCESS_dir+"pr/daily/")
#             print("no file or no permission")
        
        
        _,_,date_for_BARRA,time_leading=self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)
        
        
#         if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
#             print(self.file_BARRA_dir)
#             print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
        self.lat=data_high[1]
        self.lon=data_high[1]
#         self.shape=(316, 376)

        self.data_dem=dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif")
        self.data_dem=self.lr_transform(Image.fromarray(self.data_dem))
#         print(type(self.data_dem))
        
#             data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
#             self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for date in self.dates:
            for i in range(self.leading_time_we_use,self.leading_time_we_use+1):
            

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
#                 access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                for en in self.ensemble:
                    access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                    if os.path.exists(access_path):
                        
                    
                        if date==self.end_date and i==1:
                            break
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)

#     def get_filename_with_time_order(self,rootdir):
#         '''get filename first and generate label ,one different w'''
#         _files = []
#         for date in self.dates:
#             for en in self.ensemble:

# #                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
#                 access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
# #                 print(access_path)
#                 if os.path.exists(access_path):
#                     for i in range(self.leading_time_we_use):
#                         if date==self.end_date and i==1:
#                             break
#                         path=[]
#                         path.append(en)
#                         barra_date=date+timedelta(i)
#                         path.append(date)
#                         path.append(barra_date)
#                         path.append(i)
#                         _files.append(path)                        

    #最后去掉第一行，然后shuffle
        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        en,access_date,barra_date,time_leading=self.filename_list[idx]
        

        lr=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr")
#         lr=np.expand_dims(lr,axis=2)

#         lr=np.expand_dims(self.mapping(lr),axis=2)
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)

        if self.args.zg:
            lr_zg=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"zg"),axis=2)

        if self.args.psl:
            lr_psl=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"psl")

        if self.args.tasmax:
            lr_tasmax=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmax"),axis=2)


        if self.args.tasmin:
            lr_tasmin=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmin")
            
#         if self.args.channels==1:
#             lr=np.repeat(lr,3,axis=2)
#         return self.lr_transform(Image.fromarray(lr)),self.lr_transform(Image.fromarray(self.data_dem)),self.hr_transform(Image.fromarray(label))

        return self.lr_transform(Image.fromarray(lr)),self.data_dem,self.hr_transform(Image.fromarray(label)),torch.tensor(int(en[1:])),torch.tensor(int(access_date.strftime("%Y%m%d"))),torch.tensor(time_leading)



class ACCESS_BARRA_vdsr_pr_dem(Dataset):
    '''

   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",lr_transform=None,hr_transform=None,shuffle=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.start_date = start_date
        self.end_date = end_date
        
        self.regin = regin
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,date_for_BARRA,time_leading=self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)
        
        
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
        self.lat=data_high[1]
        self.lon=data_high[1]
#         self.shape=(316, 376)

        self.data_dem=dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif")
        self.data_dem=self.lr_transform(Image.fromarray(self.data_dem))
#         print(type(self.data_dem))
        
#             data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
#             self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for date in self.dates:
            for i in range(self.leading_time_we_use):
            

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
#                 access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                for en in self.ensemble:
                    access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                    if os.path.exists(access_path):
                        
                    
                        if date==self.end_date and i==1:
                            break
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)

#     def get_filename_with_time_order(self,rootdir):
#         '''get filename first and generate label ,one different w'''
#         _files = []
#         for date in self.dates:
#             for en in self.ensemble:

# #                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
#                 access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
# #                 print(access_path)
#                 if os.path.exists(access_path):
#                     for i in range(self.leading_time_we_use):
#                         if date==self.end_date and i==1:
#                             break
#                         path=[]
#                         path.append(en)
#                         barra_date=date+timedelta(i)
#                         path.append(date)
#                         path.append(barra_date)
#                         path.append(i)
#                         _files.append(path)                        

    #最后去掉第一行，然后shuffle
        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        en,access_date,barra_date,time_leading=self.filename_list[idx]
        

        lr=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr")
#         lr=np.expand_dims(lr,axis=2)

#         lr=np.expand_dims(self.mapping(lr),axis=2)
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)

        if self.args.zg:
            lr_zg=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"zg"),axis=2)

        if self.args.psl:
            lr_psl=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"psl")

        if self.args.tasmax:
            lr_tasmax=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmax"),axis=2)


        if self.args.tasmin:
            lr_tasmin=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmin")
            
#         if self.args.channels==1:
#             lr=np.repeat(lr,3,axis=2)
#         return self.lr_transform(Image.fromarray(lr)),self.lr_transform(Image.fromarray(self.data_dem)),self.hr_transform(Image.fromarray(label))

        return self.lr_transform(Image.fromarray(lr)),self.data_dem,self.hr_transform(Image.fromarray(label)),torch.tensor(int(en[1:])),torch.tensor(int(access_date.strftime("%Y%m%d"))),torch.tensor(time_leading)



class ACCESS_BARRA_vdsr(Dataset):
    '''
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",lr_transform=None,hr_transform=None,shuffle=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.start_date = start_date
        self.end_date = end_date
        
        self.regin = regin
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,date_for_BARRA,time_leading=self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)
        
        
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
        self.lat=data_high[1]
        self.lon=data_high[1]
        self.shape=(79,94)
#         if self.args.dem:
#             data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
#             self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        en,access_date,barra_date,time_leading=self.filename_list[idx]
        

        lr=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr")

#         lr=np.expand_dims(lr,axis=2)

#         lr=np.expand_dims(self.mapping(lr),axis=2)
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)
        lr_zg=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"zg"),axis=2)

        return self.lr_transform(Image.fromarray(lr)),self.lr_transform(Image.fromarray(lr_zg)),self.hr_transform(Image.fromarray(label)),torch.tensor(int(barra_date.strftime("%Y%m%d"))),torch.tensor(time_leading)



class ACCESS_BARRA_v2_pr_dem(Dataset):
    '''
channel we use is pr+dem 
tranning my_net
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,shuffle=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        

        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,date_for_BARRA,time_leading=self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)
        
        
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
        self.lat=data_high[1]
        self.lon=data_high[1]
        self.shape=(79,94)
        if self.args.dem:
            data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
            self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        en,access_date,barra_date,time_leading=self.filename_list[idx]
        

        lr=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr")
        lr=np.expand_dims(lr,axis=2)

#         lr=np.expand_dims(self.mapping(lr),axis=2)
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)

        if self.args.zg:
            lr_zg=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"zg"),axis=2)

        if self.args.psl:
            lr_psl=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"psl")

        if self.args.tasmax:
            lr_tasmax=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmax"),axis=2)


        if self.args.tasmin:
            lr_tasmin=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmin")
            
#         if self.args.channels==1:
#             lr=np.repeat(lr,3,axis=2)
         
        if self.transform:#channel 数量需要整理！！

            return self.transform(lr),self.transform(self.dem_data),self.transform(label),torch.tensor(int(barra_date.strftime("%Y%m%d"))),torch.tensor(time_leading)
        else:
            return lr*86400,label,torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)



class ACCESS_BARRA_v2_1(Dataset):
    '''

2.using my net to train one channel to one channel.
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,shuffle=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,date_for_BARRA,time_leading=self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)
        
        
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
        self.lat=data_high[1]
        self.lon=data_high[1]
        self.shape=(79,94)
        if self.args.dem:
            data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
            self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        en,access_date,barra_date,time_leading=self.filename_list[idx]
        

        lr=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr")
        lr=np.expand_dims(lr,axis=2)

#         lr=np.expand_dims(self.mapping(lr),axis=2)
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)

        if self.args.zg:
            lr_zg=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"zg"),axis=2)

        if self.args.psl:
            lr_psl=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"psl")

        if self.args.tasmax:
            lr_tasmax=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmax"),axis=2)


        if self.args.tasmin:
            lr_tasmin=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmin")
            
#         if self.args.channels==1:
#             lr=np.repeat(lr,3,axis=2)
         
        if self.transform:#channel 数量需要整理！！
            if self.args.channels==1:
                return self.transform(lr),self.transform(label),torch.tensor(int(barra_date.strftime("%Y%m%d"))),torch.tensor(time_leading)
        else:
            return lr*86400,label,torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)

class ACCESS_BARRA_v2_0(Dataset):
    '''
    1. using transfer learning dupalicate lr_pr and hr
    scale is size(hr)=size(lr)*scale
    version_3_documention: compare with ver1, I modify:
    1. access file is created on getitem,the file list is access_date,barra,barra_date,time_leading
      in order to read more data like zg etc. more easier, we change access_filepath to access_date

    2. in ver., norm the every inputs 
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,shuffle=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,date_for_BARRA,time_leading=self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)
        
        
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
        self.lat=data_high[1]
        self.lon=data_high[1]
        self.shape=(79,94)
        if self.args.dem:
            data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
            self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        en,access_date,barra_date,time_leading=self.filename_list[idx]
        

        lr=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr")
        lr=np.expand_dims(lr,axis=2)

#         lr=np.expand_dims(self.mapping(lr),axis=2)
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)

        if self.args.zg:
            lr_zg=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"zg"),axis=2)
            lr=np.concatenate((lr,self.mapping(lr_zg)),axis=2)

        if self.args.psl:
            lr_psl=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"psl")

        if self.args.tasmax:
            lr_tasmax=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmax"),axis=2)
            lr=np.concatenate((lr,self.mapping(lr_tasmax)),axis=2)

        if self.args.tasmin:
            lr_tasmin=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmin")
            
        if self.args.channels==1:
            lr=np.repeat(lr,3,axis=2)
            label=np.repeat(np.expand_dims(label,axis=2),3,axis=2)
         
        if self.transform:#channel 数量需要整理！！
            return self.transform(lr),self.transform(label),torch.tensor(int(barra_date.strftime("%Y%m%d"))),torch.tensor(time_leading)
        else:
            return lr*86400,label,torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)


class ACCESS_BARRA_v4(Dataset):
    '''
    scale is size(hr)=size(lr)*scale
    version_3_documention: compare with ver1, I modify:
    1. access file is created on getitem,the file list is access_date,barra,barra_date,time_leading
      in order to read more data like zg etc. more easier, we change access_filepath to access_date

    2. in ver., norm the every inputs 
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",transform=None,train=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.transform = transform
        self.start_date = start_date
        self.end_date = end_date
        
        self.scale = args.scale[0]
        self.regin = regin
        self.leading_time=217
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,_,date_for_BARRA,time_leading=self.filename_list[0]
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        data_exp=dpt.map_aust(data_high,domain=args.domain,xrarray=True)#,domain=domain)
        self.lat=data_exp["lat"]
        self.lon=data_exp["lon"]
        self.shape=(79,94)
        if self.args.dem:
            data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
            self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:
                
                    
                
#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[access_path]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        if self.args.nine2nine and self.args.date_minus_one==1:
            del _files[0]
        return _files

    

        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        access_filename_pr,en,access_date,date_for_BARRA,time_leading=self.filename_list[idx]
#         print(type(date_for_BARRA))
#         low_filename,high_filename,time_leading=self.filename_list[idx]

        lr=dpt.read_access_data(access_filename_pr,idx=time_leading).data[82:144,134:188]*86400
#         lr=dpt.map_aust(lr,domain=self.args.domain,xrarray=False)
        lr=np.expand_dims(dpt.interp_tensor_2d(lr,self.shape),axis=2)
        lr.dtype="float32"

        data_high=dpt.read_barra_data_fc(self.file_BARRA_dir,date_for_BARRA,nine2nine=True)
        label=dpt.map_aust(data_high,domain=self.args.domain,xrarray=False)#,domain=domain)

        if self.args.zg:
            access_filename_zg=self.args.file_ACCESS_dir+"zg/daily/"+en+"/"+"da_zg_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            lr_zg=dpt.read_access_zg(access_filename_zg,idx=time_leading).data[:][83:145,135:188]
            lr_zg=dpt.interp_tensor_3d(lr_zg,self.shape)
        
        if self.args.psl:
            access_filename_psl=self.args.file_ACCESS_dir+"psl/daily/"+en+"/"+"da_psl_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            lr_psl=dpt.read_access_data(access_filename_psl,var_name="psl",idx=time_leading).data[82:144,134:188]
            lr_psl=dpt.interp_tensor_2d(lr_psl,self.shape)

        if self.args.tasmax:
            access_filename_tasmax=self.args.file_ACCESS_dir+"tasmax/daily/"+en+"/"+"da_tasmax_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            lr_tasmax=dpt.read_access_data(access_filename_tasmax,var_name="tasmax",idx=time_leading).data[82:144,134:188]
            lr_tasmax=dpt.interp_tensor_2d(lr_tasmax,self.shape)
            
        if self.args.tasmin:
            access_filename_tasmin=self.args.file_ACCESS_dir+"tasmin/daily/"+en+"/"+"da_tasmin_"+access_date.strftime("%Y%m%d")+"_"+en+".nc"
            lr_tasmin=dpt.read_access_data(access_filename_tasmin,var_name="tasmin",idx=time_leading).data[82:144,134:188]
            lr_tasmin=dpt.interp_tensor_2d(lr_tasmin,self.shape)

            
#         if self.args.dem:
# #             print("add dem data")
#             lr=np.concatenate((lr,np.expand_dims(self.dem_data,axis=2)),axis=2)

            
#         print("end loading one data,time cost %f"%(time.time()-t))


        if self.transform:#channel 数量需要整理！！
            if self.args.channels==27:
                return self.transform(lr),self.transform(self.dem_data),self.transform(lr_psl),self.transform(lr_zg),self.transform(lr_tasmax),self.transform(lr_tasmin),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
            if self.args.channels==2:
                return self.transform(lr*86400),self.transform(self.dem_data),self.transform(label),torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)

        else:
            return lr*86400,label,torch.tensor(int(date_for_BARRA.strftime("%Y%m%d"))),torch.tensor(time_leading)
#         return np.reshape(train_data,(78,100,1))*86400,np.reshape(label,(312,400,1))


    


