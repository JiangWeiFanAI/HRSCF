from datetime import timedelta, date, datetime
import os
import sys
import numpy as np

class aaa(object):
    def __init__(self,lead):
        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.lead_time=lead
        self.files=self.get_filename_with_time_order()
    def get_filename_with_time_order(self):
        _files = []
        for mm in range(1,13):
            for dd in [1,9,17,25]:
#                 for i in range(self.lead_time,self.lead_time+1):
#                 for en in ['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']:
                path=[]
                date_time=date(2012, mm, dd)
                barra_date=date_time+timedelta(self.lead_time)
                path.append(date_time)
                path.append(barra_date)
                path.append(self.lead_time)
                _files.append(path)
        return _files
    def __getitem__(self,idx):
        return self.files[idx]

def load_climatology_data(l):
    data=aaa(l)
    climtology_lead_time=[]
    climatology_data=np.load('./save/crps/climatology_all_lead_time.npy')
    dates_needs=date_range(date(2012, 1, 1),date(2013, 7, 29))
    date_map=np.array(dates_needs)
    for _,target_date,_ in data.files:
        idx=np.where(date_map==target_date)[0]
        climtology_lead_time.append(climatology_data[idx][0])
    return np.array(climtology_lead_time)

def date_range(start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
land=np.load("./save/crps/whole_calibration/lead_time"+str(0)+"_whole.npy").mean(0)
x=list(range(1,218))

plt.figure(dpi=100,figsize=(15,5))

data=[]
mean_my=[]
# for i in range(30):
#     a=np.load("./save_vdsr_pr_best_test/lead_time"+str(i)+"_50station_my.npy")
#     mean_my.append(1-a.mean()/station50_int[i])
# for q in [95,75,50,25,5]:
#     data[q]=[]
for q in [90,75,50,25,10]:
    t=[]
    for lead_time in range(217):
        
#        my=np.load("./save/crps/my_217/lead_time"+str(lead_time)+".npy").mean(0)[~np.isinf(land)]
        climat=load_climatology_data(lead_time).mean(0)[~np.isinf(land)]
        cali=np.load("./save/crps/whole_calibration/lead_time"+str(lead_time)+"_whole.npy").mean(0)[~np.isinf(land)]

       # BI=np.load("./save/crps/bi_217/lead_time"+str(lead_time)+".npy").mean(0)[~np.isinf(land)]
#         print(my)
#         t.append(1-(np.percentile(my,q)/climat))
        t.append(np.percentile(    1-cali/climat,q)   )

    data.append(t)

data_np=np.array(data)
# plt.plot(x,mean_my,label="",color="b")#mean
plt.plot(x,data_np[2,:],label="",color="r")#median
    
plt.plot(x,[0]*217,color="#000000")
plt.fill_between(x,data_np[0,:],data_np[1,:],color="#cacaca")
plt.fill_between(x,data_np[1,:],data_np[2,:],color="#989898")
plt.fill_between(x,data_np[2,:],data_np[3,:],color="#989898")
plt.fill_between(x,data_np[3,:],data_np[4,:],color="#cacaca")
# plt.xlim(0,217)
# plt.grid()
plt.xlabel(" Leadtime (day)")
plt.ylabel(" SS_CRPS")
plt.ylim(-1.2,1.2)

pdf = PdfPages('ours_BI.pdf')
pdf.savefig()
pdf.close() 
