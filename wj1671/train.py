from __future__ import print_function
import torch
import argparse
import sys
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, models, transforms
import platform
from datetime import timedelta, date, datetime
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from math import log10
import time

sys.path.append("../")
from util.PrepareData import ACCESS_BARRA_Probabilistic
from model.vdsr import pcfsr,vdsr
from loss.CRPSLoss import crps_loss,crps_loss_batch_add_first,crps_loss_batch_mean_first
# from config import param_args
import torch
from datetime import timedelta, date, datetime

class param_args():
    '''
    Config class
    '''
    def __init__(self):
        self.train_name   ='VDSRd_pr_L1'
        self.resume     =''#module path
        self.test       =False
        self.test_model_name="VDSRd_pr_L1"
#         self.loss='L1'
        
        self.n_threads    =0
        self.file_ACCESS_dir = "../data/"
        self.file_BARRA_dir='../data/barra_aus/'
        self.lr_size=(79,94)
        self.hr_size=(316, 376)
        
        self.precision='single'
        self.device       = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        
        
        self.lr           = 0.001             # learning rate
        self.batch_size   = 16                  # batch size
        self.testBatchSize= 16
        
        
        self.nEpochs      = 100                # epochs
        self.checkpoints  = './checkpoints'     # checkpoints dir
        self.seed         = 123

        
        self.train_start_time =date(1990,1,2)
        self.train_end_time   =date(2011,12,31)
        self.test_start_time  =date(2012,1,1)
        self.test_end_time    =date(2012,12,31)
        
        self.leading_time_we_use=7
        self.ensemble=11
        self.domain  =[112.9, 154.25, -43.7425, -9.0]




def write_log(log,args):
    print(log)
    if not os.path.exists("./save/"+args.train_name+"/"):
        os.makedirs("./save/"+args.train_name+"/")
    my_log_file=open("./save/"+args.train_name + '/train.txt', 'a')
    my_log_file.write(log + '\n')
    my_log_file.close()
    return

class trainer():
    def __init__(self):
        self.args=param_args()
        
#     def get_loss_func():
#         if self.args.loss=='L1':
#             return nn.L1Loss(),nn.L1Loss()
#         if self.args.loss=='crps'
#             return crps_loss_batch_mean_first(),crps_loss_batch_mean_first()
#         if self.args.loss=='L1crps'
#             return crps_loss_batch_mean_first(),nn.L1Loss()
        

    def main(self):
        

        for i in self.args.__dict__:
            write_log((i.ljust(20)+':'+str(self.args.__dict__[i])),self.args)

        lr_transforms = transforms.Compose([
            transforms.Resize((self.args.hr_size),interpolation=Image.BICUBIC),
            transforms.ToTensor()

            
        ])

        hr_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        data_set=ACCESS_BARRA_vdsr(self.args.start_date,self.args.end_date,lr_transform=lr_transforms,hr_transform=hr_transforms,args=self.args)
        train_data,val_data=random_split(data_set,[int(len(data_set)*0.8),len(data_set)-int(len(data_set)*0.8)])
        train_dataloders =DataLoader(train_data,
                                                batch_size=self.args.batch_size,
                                                shuffle=True,
                                    num_workers=self.args.n_threads)
        val_dataloders =DataLoader(val_data,
                                                batch_size=self.args.batch_size,
                                                shuffle=True,
                                  num_workers=self.args.n_threads)
        model=vdsr()
        model.to(self.args.device)

        
        
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        if torch.cuda.device_count() > 1:
            write_log("!!!!!!!!!!!!!Let's use"+str(torch.cuda.device_count())+"GPUs!",self.args)
            model = nn.DataParallel(model,range(torch.cuda.device_count()))
        else:
            write_log("Let's use"+str(torch.cuda.device_count())+"GPU!",self.args)
            
            
            
        start_epoch=0
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                write_log("=> loading checkpoint '{}'".format(self.args.resume),self.args)
                checkpoint = torch.load(self.args.resume)
                start_epoch = checkpoint["epoch"] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                model.load_state_dict(checkpoint["model"].state_dict())
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99,last_epoch=start_epoch)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.resume))  


        criterion_1=nn.L1Loss()
        criterion_2=crps_loss_batch_mean_first()
        write_log("start",self.args)
        max_error=np.inf
        best_test=np.inf
        train_loss_list=[]
        test_loss_list=[]
        for epoch in range(start_epoch, self.args.nEpochs + 1): 

            start=time.time()
            write_log("epoch = "+str(epoch)+", lr = "+str(optimizer.param_groups[0]["lr"]),self.args)
            model.train()    
            optimizer.zero_grad()
            avg_loss=0
            for iteration, (pr,hr,_,_,_) in enumerate(train_dataloders):

                pr,hr= self.prepare([pr,hr],self.args.device)

                sr = model(pr)

#                 loss = 0.5*criterion_2(sr,hr) + 0.5*criterion_1(sr,hr)
                loss = criterion_1(sr, hr)

                loss.backward()

                optimizer.step()

                avg_loss+=loss.item()
            write_log("epoche: %d,lr: %f,time cost %f s, train_loss: %f "%(
                              epoch,
                              optimizer.state_dict()['param_groups'][0]['lr'],
                              time.time()-start,
                              avg_loss / len(train_dataloders),
                         ),self.args)
            scheduler.step()
            

            start=time.time()
            model.eval()
            with torch.no_grad():
                avg_crps=0
                for iteration, (pr,hr,_,_,_) in enumerate(val_dataloders):

                    pr,hr= self.prepare([pr,hr],self.args.device)
                    out_ensemble,sr = model(pr)

                    crp_score = criterion_2(out_ensemble,hr)
                    avg_crps+= crp_score.item()


                test_crps= avg_crps / len(val_dataloders)
                write_log("epoche: %d,lr: %f,time cost %f s, test: %f "%(
                          epoch,
                          optimizer.state_dict()['param_groups'][0]['lr'],
                          time.time()-start,
                          test_crps,
                     ),self.args)

            
            if best_test >=test_crps:
                best_test=test_crps
                self.save_checkpoint(model,epoch,optimizer)
                write_log('best crps epoches: %d: best_test : %f '%(epoch,best_test),self.args)
                

#             self.save_checkpoint(model, epoch,optimizer)
        
        
    def train(self,train_dataloders, optimizer,scheduler, model, criterion_1,criterion_2, epoch):
            

        start=time.time()
        write_log("epoch = "+str(epoch)+", lr = "+str(optimizer.param_groups[0]["lr"]),self.args)
        model.train()    
        optimizer.zero_grad()
        avg_loss=0
        for iteration, (pr,hr,_,_,_) in enumerate(train_dataloders):

            pr,hr= self.prepare([pr,hr],self.args.device)

            out_ensemble,sr = model(pr)

            loss = 0.5*criterion_1(sr, hr)+0.5*criterion_2(out_ensemble,hr)
            loss.backward()

            optimizer.step()
            
            avg_loss+=loss.item()*self.args.batch_size
        write_log("epoche: %d,lr: %f,time cost %f s, train_loss: %f "%(
                          epoch,
                          optimizer.state_dict()['param_groups'][0]['lr'],
                          time.time()-start,
                          avg_loss / len(train_dataloders),
                     ),self.args)
        scheduler.step()         
#                 print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_dataloders), loss.data[0]))
        return avg_loss / len(train_dataloders)
                
    def test(self,val_dataloders, optimizer,model, criterion, best_test,epoch):
            

        start=time.time()
        model.eval()
        with torch.no_grad():
            avg_crps=0
            for iteration, (pr,hr,_,_,_) in enumerate(val_dataloders):

                pr,hr= self.prepare([pr,hr],self.args.device)
                out_ensemble,sr = model(pr)

                crp_score = criterion(out_ensemble,hr)
                avg_crps+= crp_score.item() * self.args.batch_size


            test_crps= avg_crps / len(val_dataloders)
            write_log("epoche: %d,lr: %f,time cost %f s, test: %f "%(
                      epoch,
                      optimizer.state_dict()['param_groups'][0]['lr'],
                      time.time()-start,
                      test_crps,
                 ),self.args)

#                     print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(val_dataloders), loss.data[0]))
        

        return test_crps

                
    def prepare(self,l,device=False):
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            if self.args.precision == 'single': tensor = tensor.float()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]     
    
    def save_checkpoint(self,model, epoch,optimizer):
        model_folder = "./save/"+self.args.train_name
        model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
        state = {"epoch": epoch ,
                 "model": model ,
                 'optimizer': optimizer.state_dict(),
                 'argparse': self.args
                }
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        torch.save(state, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))
        
if __name__=='__main__':
    trainner=trainer()
    trainner.main()    