from __future__ import print_function
import torch
import argparse
import sys
sys.path.append("../")
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, models, transforms
import platform
from datetime import timedelta, date, datetime
from model import vdsr
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from math import log10
import time
from util.PrepareData import ACCESS_AWAP_GAN
from model.vdsr import vdsr


def write_log(log,args):
    print(log)
    if not os.path.exists("./save/"+args.train_name+"/"):
        os.mkdir("./save/"+args.train_name+"/")
    my_log_file=open("./save/"+args.train_name + '/train.txt', 'a')
#     log="Train for batch %d,data loading time cost %f s"%(batch,start-time.time())
    my_log_file.write(log + '\n')
    my_log_file.close()
    return




def evaluation(net,val_dataloders,loss,criterion,args):
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loss=0
    avg_psnr = 0
    start=time.time()
    with torch.no_grad():
        for batch, (pr,hr,_,_) in enumerate(val_dataloders):
            pr,hr = prepare([pr, hr],device,args)
            sr = net(pr)
            val_loss=criterion(sr, hr)
            test_loss+=val_loss.item()
            psnr = 10 * log10(1000 / val_loss.item())
            avg_psnr += psnr
        write_log("evalutaion: time cost %f s, test_loss: %f, psnr: avg_psnr %f"%(
                      time.time()-start,
                      test_loss/(batch + 1),
                      avg_psnr / len(val_dataloders)
                 ),args)
    return test_loss

def prepare(l,device=False,args=0):
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        if args.precision == 'single': tensor = tensor.float()
        return tensor.to(device)

    return [_prepare(_l) for _l in l]

def main():
    
    
    # ===========================================================
    # Training settings
    # ===========================================================
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=36,
                        help='number of threads for data loading')

    parser.add_argument('--cpu', action='store_true',help='cpu only?') 
    parser.add_argument('--test', action='store_true',help='cpu only?') 

    # hyper-parameters
    parser.add_argument('--train_name', type=str, default="vdsr_pr_awap", help='training name')

    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

    # model configuration
    parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
    parser.add_argument('--model', '-m', type=str, default='vdsr', help='choose which model is going to use')

    #data
    parser.add_argument('--pr', type=bool, default=True,help='add-on pr?')

    parser.add_argument('--train_start_time', type=type(datetime(1990,1,2)), default=datetime(1990,1,2),help='r?')
    parser.add_argument('--train_end_time', type=bool, default=datetime(2011,12,31),help='?')
    parser.add_argument('--test_start_time', type=bool, default=datetime(2012,1,1),help='a?')
    parser.add_argument('--test_end_time', type=bool, default=datetime(2012,12,24),help='')

    parser.add_argument('--start_date', type=bool, default=datetime(1990,1,2),help='a?')
    parser.add_argument('--end_date', type=bool, default=datetime(2012,12,24),help='')
    parser.add_argument('--val_start_date', type=bool, default=datetime(1996,12,25),help='a?')
    parser.add_argument('--val_end_date', type=bool, default=datetime(1998,8,1),help='')


    parser.add_argument('--dem', action='store_true',help='add-on dem?') 
    parser.add_argument('--psl', action='store_true',help='add-on psl?') 
    parser.add_argument('--zg', action='store_true',help='add-on zg?') 
    parser.add_argument('--tasmax', action='store_true',help='add-on tasmax?') 
    parser.add_argument('--tasmin', action='store_true',help='add-on tasmin?')
    parser.add_argument('--leading_time_we_use', type=int,default=7,help='add-on tasmin?')
    parser.add_argument('--ensemble', type=int, default=11,help='total ensambles is 11') 
    parser.add_argument('--channels', type=float, default=0,help='channel of data_input must') 
    #[111.85, 155.875, -44.35, -9.975]
    parser.add_argument('--domain', type=list, default=[112.9, 154.25, -43.7425, -9.0],help='dataset directory')

    parser.add_argument('--file_ACCESS_dir', type=str, default="../data/",help='dataset directory')
    parser.add_argument('--file_BARRA_dir', type=str, default="../data/barra_aus/",help='dataset directory')
    parser.add_argument('--file_DEM_dir', type=str, default="../DEM/",help='dataset directory')
    parser.add_argument('--precision', type=str, default='single',choices=('single', 'half','double'),help='FP precision for test (single | half)')

    args = parser.parse_args()

    
    
#     init_date=date(1970, 1, 1)
#     start_date=date(1990, 1, 2)
#     end_date=date(2011,12,25)
    sys = platform.system()
    if sys == "Windows":
        init_date=date(1970, 1, 1)
        args.start_date=date(1990, 1, 2)
        args.end_date=date(1990,2,9) #if 929 is true we should substract 1 day   
#         args.file_ACCESS_dir="E:/climate/access-s1/"
#         args.file_BARRA_dir="C:/Users/JIA059/barra/"
        args.file_DEM_dir="../DEM/"
    
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
    data_set=ACCESS_AWAP_GAN(args.train_start_time,args.train_end_time)
    if args.test:
        print(1)
    else:
    
    
    
        train_data,val_data=random_split(data_set,[int(len(data_set)*0.8),len(data_set)-int(len(data_set)*0.8)])
        #######################################################################
        train_dataloders =DataLoader(train_data,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                    num_workers=args.n_threads)
        val_dataloders =DataLoader(val_data,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                  num_workers=args.n_threads)
        #######################################################################


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net=vdsr()

    #     criterion = nn.MSELoss(size_average=False)
        criterion=nn.L1Loss()

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    #     optimizer_my = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        if torch.cuda.device_count() > 1:
            write_log("!!!!!!!!!!!!!Let's use"+str(torch.cuda.device_count())+"GPUs!",args)
            net = nn.DataParallel(net,range(torch.cuda.device_count()))
        else:
            write_log("Let's use"+str(torch.cuda.device_count())+"GPUs!",args)

        net.to(device)
        ##############################################
        write_log("start",args)
        max_error=np.inf
        val_max_error=np.inf


        for e in range(args.nEpochs):
            loss=0
            start=time.time()
            for batch, (pr,hr,_,_) in enumerate(train_dataloders):
                pr,hr= prepare([pr,hr],device,args)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    sr = net(pr)
                    running_loss =criterion(sr, hr)
                    running_loss.backward()
                    optimizer.step()
                loss+=running_loss.item()
                if batch%10==0:
                    state = {'model': net, 'optimizer': optimizer.state_dict(), 'epoch': e}
                    torch.save(state, "./save/"+args.train_name+"/last.pth")

            write_log("epoche: %d,lr: %f,time cost %f s, train_loss: %f "%(
                          e,
                          optimizer.state_dict()['param_groups'][0]['lr'],
                          time.time()-start,
                          loss/(batch + 1),
                     ),args)
            scheduler.step() 
            if e%10==0:
                test_loss=evaluation(net,val_dataloders,loss,criterion,args)
                if test_loss<val_max_error:
                    write_log("saveing model for best test",args)
                    state = {'model': net, 'optimizer': optimizer.state_dict(), 'epoch': e}
                    torch.save(state, "./save/"+args.train_name+"/best_test_"+str(e)+".pth")

            if max_error>loss:
                write_log("saveing model for best train",args)
                state = {'model': net, 'optimizer': optimizer.state_dict(), 'epoch': e}
                torch.save(state, "./save/"+args.train_name+"/best_train_"+str(e)+".pth")
            
if __name__=='__main__':
    main()            