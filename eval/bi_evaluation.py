import os
import sys
sys.path.append('../')
import util.data_processing_tool as dpt
from model import vdsr

from datetime import timedelta, date, datetime
# import args_parameter as args
import torch,torchvision
import numpy as np
import random
import properscoring as ps
from torch.utils.data import Dataset,random_split
from torchvision import datasets, models, transforms

import time
import xarray as xr
from PIL import Image


import matplotlib as plt
import argparse
import sys
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, models, transforms
import platform
from datetime import timedelta, date, datetime

import torch.nn as nn

import torch.optim as optim

from math import log10
import time
# from PrepareData import ACCESS_BARRA_crps
import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
# ===========================================================
# Training settings
# ===========================================================


def rmse(ens,hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.sqrt((ens-hr).sum(axis=(0)))

def mae(ens,hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((ens-hr)).sum(axis=0)


class ACCESS_BARRA_crps(Dataset):
    '''

2.using my net to train one channel to one channel.
   
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
        list = os.listdir(rootdir) #Ã¥Ë†â€”Ã¥â€¡ÂºÃ¦â€“â€¡Ã¤Â»Â¶Ã¥Â¤Â¹Ã¤Â¸â€¹Ã¦â€°â‚¬Ã¦Å“â€°Ã§Å¡â€žÃ§â€ºÂ®Ã¥Â½â€¢Ã¤Â¸Å½Ã¦â€“â€¡Ã¤Â»Â¶
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

    #Ã¦Å“â‚¬Ã¥ÂÅ½Ã¥Å½Â»Ã¦Å½â€°Ã§Â¬Â¬Ã¤Â¸â‚¬Ã¨Â¡Å’Ã¯Â¼Å’Ã§â€žÂ¶Ã¥ÂÅ½shuffle
        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #Ã¥Â°â€ Ã¦â€¢Â°Ã¦ÂÂ®Ã¦ËœÂ Ã¥Â°â€žÃ¥Ë†Â°[-1,1]Ã¥Å’ÂºÃ©â€”Â´ Ã¥ÂÂ³a=-1Ã¯Â¼Å’b=1
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
        

        lr=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr")[82:144,134:188]*86400
        lr=dpt.interp_tensor_2d(lr,(79,94))
        
#         lr=np.expand_dims(lr,axis=2)

#         lr=np.expand_dims(self.mapping(lr),axis=2)
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)

            
#         if self.args.channels==1:
#             lr=np.repeat(lr,3,axis=2)
#         return self.lr_transform(Image.fromarray(lr)),self.lr_transform(Image.fromarray(self.data_dem)),self.hr_transform(Image.fromarray(label))

        return self.lr_transform(Image.fromarray(lr)),torch.tensor(int(en[1:])),self.hr_transform(Image.fromarray(label)),torch.tensor(int(en[1:])),torch.tensor(int(access_date.strftime("%Y%m%d"))),torch.tensor(time_leading)

    

def crps(ensin,obs):
    '''
    @param ensin A vector of prediction
    @param obs  A vector of observations
    
'''

#     assert not np.isnan(ensin).any() and not np.isnan(obs).any(), "data contains nan"
         
    Fn = ECDF(ensin)
    xn=np.sort(np.unique(ensin))
    m=len(xn)
    dn=np.diff(xn)
    eq1=0
    eq2=0
    if(obs>xn[0] and obs<xn[m-1]): #obsÃ¥Å“Â¨Ã¨Å’Æ’Ã¥â€ºÂ´Ã¥â€ â€¦
        k=np.max(np.where(xn<=obs))#Ã¥Â°ÂÃ¤ÂºÅ½obsÃ§Å¡â€žÃ¦Å“â‚¬Ã¥Â¤Â§Ã¥â‚¬Â¼Ã¤Â¸â€¹Ã¦Â â€¡
        x0 = xn[k] #Ã¥Â°ÂÃ¤ÂºÅ½obsÃ§Å¡â€žÃ¦Å“â‚¬Ã¥Â¤Â§Ã¥â‚¬Â¼
        if k>0:
            eq1=np.sum(Fn(xn[0:k+1])**2*np.append(dn[0:k], obs - xn[k]))#Ã¥Â°ÂÃ¤ÂºÅ½obsÃ§Å¡â€žÃ¦â€°â‚¬Ã¦Å“â€°Ã¥â‚¬Â¼ Ã§Å¡â€ž Ã§â„¢Â¾Ã¥Ë†â€ Ã¦Â¯â€Ã¦â€¢Â° Ã§Å¡â€žÃ¥Â¹Â³Ã¦â€“Â¹
        else:
            eq1 =np.sum(Fn(xn[0])**2*(obs - xn[0]))
        if k<m-2:

            eq2=np.sum((1-Fn(xn[k:m-1]))**2*np.append(xn[k+1] - obs, dn[(k+1):(m-1)]))
        else:
            eq2 =np.sum((1-Fn(xn[m-2]))**2*(xn[m-1] - obs))

    if obs <= xn[0]: # Ã¨Â§â€šÃ¦Âµâ€¹Ã¥â‚¬Â¼Ã¥Å“Â¨Ã¤Â¹â€¹Ã¥Â¤â€“
        eq2 =np.sum(np.append(1, 1-Fn(xn[0:(m-1)]))**2*np.append(xn[0]-obs, dn))
    if obs >= xn[m-1]:
        eq1= np.sum(Fn(xn)**2*np.append(dn, obs - xn[m-1]))
            
    return eq1+eq2 




def vectcrps_v(fct_ens,obs):
    '''
    #' @param fct_ens A 2D prediction
    #' @param obs  A vector of observations
    #' @return a crps vector'''
    score =0

    
    fct_ens=fct_ens
    assert not np.isnan(fct_ens).any() and not np.isnan(obs).any(),"data contains nan"
    for i in range(obs.shape[0]):
#         print(fct_ens[:,i],obs[i])
        score+=crps(fct_ens[:,i],obs[i])
  
    return score


def vectcrps_m(fct_ens,obs):
    '''
    #' @param fct_ens A 2D prediction 11*1*1
    #' @param obs  A vector of observations
    #' @return a crps vector'''
    score =0
#     assert np.isnan(fct_ens).any() and np.isnan(obs).any(),"data contains nan"
    score_map=np.zeros((obs.shape[0],obs.shape[1]))
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            score_map[i,j]=crps(fct_ens[:,i,j],obs[i,j])
#             score+=crps(fct_ens[:,i,j],obs[i,j])
    return score_map
    return score/(obs.shape[0]*obs.shape[1])   


def vectcrps_cali(fct_ens,obs):

    score =0
    mapp=np.load("mmap.npy")
#     assert np.isnan(fct_ens).any() and not np.isnan(obs).any(),"data contains nan"
    score_map=np.zeros((obs.shape[0],obs.shape[1]))
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
#             if (np.array(fct_ens[:,i,j],dtype=np.float32)>2000).any() or (mapp[i,j]==-1).any():
#             if (np.array(fct_ens[:,i,j],dtype=np.float32)>2000).any(): 
            if (np.array(fct_ens[:,mapp[i,j,0],mapp[i,j,1]],dtype=np.int)>=-0.00001).all():
                score_map[i,j]=crps(fct_ens[:,mapp[i,j,0],mapp[i,j,1]],obs[i,j])
            else:
                score_map[i,j]=np.inf
#             score+=crps(fct_ens[:,i,j],obs[i,j])
    return score_map


def write_log(log,args):
    print(log)
    if not os.path.exists("./save/"+args.train_name+"/"):
        os.mkdir("./save/"+args.train_name+"/")
    my_log_file=open("./save/"+args.train_name + '/train.txt', 'a')
#     log="Train for batch %d,data loading time cost %f s"%(batch,start-time.time())
    my_log_file.write(log + '\n')
    my_log_file.close()
    return
 
def main(year):
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=0,
                        help='number of threads for data loading')

    parser.add_argument('--cpu', action='store_true',help='cpu only?') 

    # hyper-parameters
    parser.add_argument('--train_name', type=str, default="cali_crps", help='training name')

    parser.add_argument('--batch_size', type=int, default=44, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

    # model configuration
    parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
    parser.add_argument('--model', '-m', type=str, default='vdsr', help='choose which model is going to use')

    #data
    parser.add_argument('--pr', type=bool, default=True,help='add-on pr?')

    parser.add_argument('--train_start_time', type=type(datetime(1990,1,25)), default=datetime(1990,1,2),help='r?')
    parser.add_argument('--train_end_time', type=type(datetime(1990,1,25)), default=datetime(1990,2,9),help='?')
    parser.add_argument('--test_start_time', type=type(datetime(2012,1,1)), default=datetime(2010,1,1),help='a?')
    parser.add_argument('--test_end_time', type=type(datetime(2012,12,31)), default=datetime(2010,12,31),help='')

    parser.add_argument('--dem', action='store_true',help='add-on dem?') 
    parser.add_argument('--psl', action='store_true',help='add-on psl?') 
    parser.add_argument('--zg', action='store_true',help='add-on zg?') 
    parser.add_argument('--tasmax', action='store_true',help='add-on tasmax?') 
    parser.add_argument('--tasmin', action='store_true',help='add-on tasmin?')
    parser.add_argument('--leading_time_we_use', type=int,default=1
                        ,help='add-on tasmin?')
    parser.add_argument('--ensemble', type=int, default=11,help='total ensambles is 11') 
    parser.add_argument('--channels', type=float, default=0,help='channel of data_input must') 
    #[111.85, 155.875, -44.35, -9.975]
    parser.add_argument('--domain', type=list, default=[112.9, 154.25, -43.7425, -9.0],help='dataset directory')

    parser.add_argument('--file_ACCESS_dir', type=str, default="/g/data/ub7/access-s1/hc/raw_model/atmos/",help='dataset directory')
    parser.add_argument('--file_BARRA_dir', type=str, default="../Data/barra_aus/",help='dataset directory')
    parser.add_argument('--file_DEM_dir', type=str, default="../DEM/",help='dataset directory')
    parser.add_argument('--precision', type=str, default='single',choices=('single', 'half','double'),help='FP precision for test (single | half)')

    args = parser.parse_args()
    
    # def main():

#     init_date=date(1970, 1, 1)
#     start_date=date(1990, 1, 2)
#     end_date=date(2011,12,25)
    sys = platform.system()
    args.dem=False
    args.train_name="pr_dem"
    args.channels=0
    if args.pr:
        args.channels+=1
    if args.zg:
        args.channels+=1
    if args.psl:
        args.channels+=1
    if args.tasmax:
        args.channels+=1
    if args.tasmin:
        args.channels+=1
    if args.dem:
        args.channels+=1
    print("training statistics:")
    print("  ------------------------------")
    print("  trainning name  |  %s"%args.train_name)
    print("  ------------------------------")
    print("  num of channels | %5d"%args.channels)
    print("  ------------------------------")
    print("  num of threads  | %5d"%args.n_threads)
    print("  ------------------------------")
    print("  batch_size     | %5d"%args.batch_size)
    print("  ------------------------------")
    print("  using cpu only | %5d"%args.cpu)

    lr_transforms = transforms.Compose([
        transforms.Resize((316, 376)),
    #     transforms.RandomResizedCrop(IMG_SIZE),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(30),
        transforms.ToTensor()
    #     transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    hr_transforms = transforms.Compose([
    #         transforms.Resize((316, 376)),
    #     transforms.RandomResizedCrop(IMG_SIZE),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(30),
        transforms.ToTensor()
    #     transforms.Normalize(IMG_MEAN, IMG_STD)
    ])
    args.test_start_time=datetime(year,1,1)
    args.test_end_time=datetime(year,12,31)
    data_set=ACCESS_BARRA_crps(args.test_start_time,args.test_end_time,lr_transform=lr_transforms,hr_transform=hr_transforms,shuffle=False,args=args)



    #     #######################################################################

    test_data=DataLoader(data_set,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                num_workers=args.n_threads,drop_last=True)

    #     #######################################################################


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net=torch.load("./save/vdsr_pr/best_test.pth")
    # net=torch.load("../data/model/vdsr_pr/best_test.pth")




    #     ##############################################
    write_log("start",args)
    #     max_error=np.inf
    #     val_max_error=np.inf

    #     print(data_set.filename_list)

    # for e in range(args.nEpochs):
    #         loss=0
    for lead in range(217):
        args.leading_time_we_use=lead

        data_set=ACCESS_BARRA_crps(args.test_start_time,args.test_end_time,lr_transform=lr_transforms,hr_transform=hr_transforms,shuffle=False,args=args)


        test_data=DataLoader(data_set,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                    num_workers=args.n_threads,drop_last=False)


        crps_score_vsdr=[]
        mae_score_vsdr=[]
        rmse_score_vsdr=[]
        start=time.time()
        fmt = '%Y%m%d'


    #         test_data=tqdm.tqdm(test_data)
        for batch, (pr,dem,hr,en,data_time,idx) in enumerate(test_data):

            with torch.set_grad_enabled(False):

    #                 sr = net(pr)
                sr_np=pr.cpu().numpy()
                hr_np=hr.cpu().numpy()
#                 print(pr.shape)
#                 print(hr.shape)
#                 print(en)
#                 print(data_time)
                for i in range(args.batch_size//args.ensemble):
                    a=np.squeeze( sr_np[i*args.ensemble:(i+1)*args.ensemble])
                    b=np.squeeze(hr_np[i*args.ensemble])
                    #skil=vectcrps_m(a,b)
#                    skil=ps.crps_ensemble(b,np.transpose(a,(1,2,0)))
                    rmes_score=rmse(a,b)
                    mae_score=mae(a,b)




                    #crps_score_vsdr.append(skil)
                    mae_score_vsdr.append(mae_score)
                    rmse_score_vsdr.append(rmes_score)
        if not os.path.exists("../save/rmse/bi_217/"+str(year)):
            dpt.mkdir("../save/rmse/bi_217/"+str(year))
            
        if not os.path.exists("../save/mae/bi_217/"+str(year)):
            dpt.mkdir("../save/mae/bi_217/"+str(year))
            
        if not os.path.exists("../save/crps/bi_217/"+str(year)):
            dpt.mkdir("../save/crps/bi_217/"+str(year))
#         np.save("./save/crps/bi_217/2010/lead_time_"+str(lead),crps_score_vsdr)
        np.save("../save/rmse/bi_217/"+str(year)+"/lead_time_"+str(lead),rmse_score_vsdr)
        np.save("../save/mae/bi_217/"+str(year)+"/lead_time_"+str(lead),mae_score_vsdr)
        
        print(str(lead)+" : "+str(np.array(rmse_score_vsdr).mean()),",")
        print(str(lead)+" : "+str(np.array(mae_score_vsdr).mean()),",")
            


if __name__=='__main__':
    main(year=1997)    
    main(year=2010)    
    main(year=2012)    
  
