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
    climatology_data=np.load('./save/crps/climatology_all_lead_time_windows_0.npy')
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



import csv
import numpy as np


def calculate_csv_file(file_dir="./save/mean_table_mean_crps_mean_ss.csv",option='50'):
    '''
    @option:'whole', '50', '214'
            'whole' presents the whole australia.
            '50': means 50 observation station.
            '214' presents 214 new observation application station.
    '''
    print(0)
    if option == "whole":
        print(1)
        with open(file_dir, "w", newline='') as file:
            csv_file = csv.writer(file)
            head = ["leading","climatology_11",'bilinear interpolation',"our model","calibration",'cspr_ss_mean_bilinearinterpolation_against_climatology',"cspr_ss_mean_ourmodel_against_climatology","cspr_ss_mean_calibration_against_climatology"]
            csv_file.writerow(head)
            t=np.load("./save/crps/whole_calibration/lead_time"+str(0)+"_whole.npy").mean(0)
            for lead_time in range(217):
                line=[lead_time,]
                cali=np.load("./save/crps/whole_calibration/lead_time"+str(lead_time)+"_whole.npy").mean(0)[~np.isinf(t)]

                climat=load_climatology_data(lead_time).mean(0)[~np.isinf(t)]
                my=np.load("./save/crps/my_217/lead_time"+str(lead_time)+".npy").mean(0)[~np.isinf(t)]

                BI=np.load("./save/crps/bi_217/lead_time"+str(lead_time)+".npy").mean(0)[~np.isinf(t)]


                my_mean=my.mean()

        #         print(my_mean.shape)
                cali_mean=cali.mean()
                qm_mean=climat.mean()
                BI_mean=BI.mean()

        #         my_cali=(1- my/cali)
                my_qm=(1- my/climat)
                cali_qm=(1- cali/climat)
                bi_qm=(1- BI/climat)
        #         np.save("./save/crps/result/my_cali"+str(lead_time),my_cali)
        #         np.save("./save/crps/result/my_qm"+str(lead_time),my_qm)
        #         np.save("./save/crps/result/cali_qm"+str(lead_time),cali_qm)
        #         np.save("./save/crps/result/bi_qm"+str(lead_time),bi_qm)

        #         cspr_ss_mean_my_cali=my_cali.mean()
                cspr_ss_mean_my_qm=my_qm.mean()
                cspr_ss_mean_cali_qm=cali_qm.mean()
                cspr_ss_mean_bi_qm=bi_qm.mean()

                line.append(qm_mean)
                line.append(BI_mean)
                line.append(my_mean)
                line.append(cali_mean)
                line.append(cspr_ss_mean_bi_qm)
                line.append(cspr_ss_mean_my_qm)
                line.append(cspr_ss_mean_cali_qm)
        #         line.append(cspr_ss_mean_my_cali)
                csv_file.writerow(line)
    else:
        print(2)
        if option=='50':
            from constant_param import station_50_index_for_size_of_hr_sr as station_dict
        if option=='214':
            from constant_param import station_214_index_for_size_of_hr_sr as station_dict
        print(3)
        t=np.load("./save/crps/whole_calibration/lead_time"+str(0)+"_whole.npy").mean(0)
        station_index=np.zeros((t.shape))
        for i in station_dict.keys():
            station_index[station_dict[i]]=1
        with open(file_dir, "w", newline='') as file:
            csv_file = csv.writer(file)
            head = ["leading","climatology_11",'bilinear interpolation',"our model","calibration",'cspr_ss_mean_bilinearinterpolation_against_climatology',"cspr_ss_mean_ourmodel_against_climatology","cspr_ss_mean_calibration_against_climatology"]
            csv_file.writerow(head)
            t=np.load("./save/crps/whole_calibration/lead_time"+str(0)+"_whole.npy").mean(0)
            for lead_time in range(217):
                line=[lead_time,]

                cali=np.load("./save/crps/whole_calibration/lead_time"+str(lead_time)+"_whole.npy").mean(0)[station_index==1]
                climat=load_climatology_data(lead_time).mean(0)[station_index==1]
                my=np.load("./save/crps/my_217/lead_time"+str(lead_time)+".npy").mean(0)[station_index==1]

                BI=np.load("./save/crps/bi_217/lead_time"+str(lead_time)+".npy").mean(0)[station_index==1]


                my_mean=my.mean()
                cali_mean=cali.mean()
                qm_mean=climat.mean()
                BI_mean=BI.mean()

        #         my_cali=(1- my/cali)
                my_qm=(1- my/climat)
                cali_qm=(1- cali/climat)
                bi_qm=(1- BI/climat)
        #         np.save("./save/crps/result/my_cali"+str(lead_time),my_cali)
        #         np.save("./save/crps/result/my_qm"+str(lead_time),my_qm)
        #         np.save("./save/crps/result/cali_qm"+str(lead_time),cali_qm)
        #         np.save("./save/crps/result/bi_qm"+str(lead_time),bi_qm)
        #         cspr_ss_mean_my_cali=my_cali.mean()
                cspr_ss_mean_my_qm=my_qm.mean()
                cspr_ss_mean_cali_qm=cali_qm.mean()
                cspr_ss_mean_bi_qm=bi_qm.mean()

                line.append(qm_mean)
                line.append(BI_mean)
                line.append(my_mean)
                line.append(cali_mean)
                line.append(cspr_ss_mean_bi_qm)
                line.append(cspr_ss_mean_my_qm)
                line.append(cspr_ss_mean_cali_qm)
        #         line.append(cspr_ss_mean_my_cali)
                csv_file.writerow(line)
calculate_csv_file()