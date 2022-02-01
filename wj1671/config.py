import torch
from datetime import timedelta, date, datetime

class param_args():
    '''
    Config class
    '''
    def __init__(self):
        self.train_name   ='pcfsr_pr'
        self.resume     =''#module path
        self.test       =False
        self.test_model_name="pcfsr"
        
        
        self.n_threads    =0
        self.file_ACCESS_dir = "../../Data/"
        self.file_BARRA_dir='../../Data/barra_aus/'
        self.lr_size=(79,94)
        self.hr_size=(316, 376)
        
        self.precision='single'
        self.device       = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        
        
        self.lr           = 0.00001             # learning rate
        self.batch_size   = 4                  # batch size
        self.testBatchSize= 4
        
        self.nEpochs      = 100                # epochs
        self.checkpoints  = './checkpoints'     # checkpoints dir
        self.seed         = 123
#         self.upscale_factor= 4
        
        self.train_start_time =date(1990,1,1)
        self.train_end_time   =date(1990,12,31)
        self.test_start_time  =date(2012,1,1)
        self.test_end_time    =date(2012,12,31)
        
        self.leading_time_we_use=7
        self.ensemble=11
        self.domain  =[112.9, 154.25, -43.7425, -9.0]



        
        
#         self.n_resgroups  =10
#         self.n_resblocks  =20
#         self.n_feats      =64
#         self.reduction    =16
#         self.rgb_range    =255
#         self.n_colors     =3
#         self.res_scale    =1
#         self.patch_size   =48
#         self.__mkdir(self.checkpoints)
# param_args()