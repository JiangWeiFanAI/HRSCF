# HRSCF_v1


[Warren Jin](https://people.csiro.au/J/W/Warren-Jin), [Weifan Jiang](https://www.linkedin.com/in/jeffery-jiang-3b966615a/), Minzhe Chen, Ming Li, K.shuvo Bakar,
[Applying Deep Learning for Operational Long Lead Time Probabilistic Daily Rainfall Forecasts](https://github.com/JiangWeiFanAI/HRSCF/blob/main/Applying%20Deep%20Learning%20for%20Operational%20Long%20Lead%20Time%20Probabilistic%20Daily%20Rainfall%20Forecasts.pdf)


The code is built and tested on Windows 10  environment (Python3.7, PyTorch_1.3.0, netCDF4_1.5.3, basemap_1.2.1, matplotlib_3.1.3,libtiff_0.4.2, xarray, CUDA10.2, cuDNN7.2) with Rtx2070 GPU. And push to [NCI](https://nci.org.au/) for traning and evaluation.

The tool will be include [Panoply](https://www.giss.nasa.gov/tools/panoply/), [Qgis](https://qgis.org/en/site/)





## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Results](#results)
4. [Citation](#citation)


## Introduction
Downscaling has become a hot research content in super-resolution and analysis and computer vision because of its high application value in many real-life scenarios. When it applies to rainfall forecasts, This will have immeasurable value to people's lives and climate-sensitive industries, such as agriculture, mining and constructions. However, the existing algorithms aim to downscale probabilistic forecasts in terms of statistics, which are time-consuming and do not have a skilful improvement. Concerning the deep learning method on rainfall forecasts, recent work focus on making a precision mapping from low resolution to high resolution where the ensemble is not considered as a fact. However, for the physical model of probabilistic prediction, it is also impossible to carry out information statistics. To this end, we apply the deep learning method to implement a downscaling model for the model with ensemble forecasts. After many repeated experiments, I proposed the VDSRd model to downscaling rainfall forecasts with 11 ensemble number and leading 217 days and achieved skill on 8.1\% on first seven days forecasts on average and -4.9\% on 217 days leading by CRPSS(Continuous Ranked Probability skill score) .

![VDSRD](/data/img/net.png)


## Train
### Training data 

In 2017, the Australian Meteorology Bureau announced the next generation access series GCM, which were later installed on supercomputers in the office in 2018. A worldwide combined model, the seasonal climate and earth system Simulator (ACCESS-S), is based on the UK's global combined seasonal prediction system glosea5-gc2. The ACCESS-S contains 11 different ensemble members for seasonal forecasting and leading 217 days due to disruptions and improved ensemble technology,  including ten disturbed members and oneunperturbed centre member. 



Bureau of Meteorology Atmospheric high-resolution Regional Reanalysis for Australia(BARRA) is Australia's regional climate prediction and numerical climate forecasts models based on an Australian area, using ACCESS-R, Australia's first atmospheric reanalysis model. ACCESS-R employs the UKMO system other than ACCESS-S. In addition, any uncertainty is not considered in this system. i.e. no ensemble member is present.

In the trainingset, we used 60km Raw atmosphere grid ACCESS-S precipitation data as input and 25km BARRA-R data as label.
All the data was stored on the project named [iu60](http://poama.bom.gov.au/) and [ma05](http://www.bom.gov.au/clim_data/rrp/BARRA_sample/)
, you need to require permission from [NCI](https://nci.org.au/) on path (g/data/iu60/) and (g/data/ma05/) respectively.


### pre-processing data
To reduce IO time, we round the data to the same geographic location, longtitude and latitude is 112.9, 154.25, -43.7425, -9.0. Go to data_processing folder and type following command, and the data will be marked and relocated:

 ```bash
    python3 transform_access_pr.py
    python3 transform_Barra_data.py

 ```

### Begin to train



Cd to 'wj1671' , run the following scriptto train model.


```bash
    python train.py 
```

### Begin to evaluation
Cd to '..' back to main folder , Grab the best model to /save path, and use jupyter notebooks to generate climatology, Bicubic, bcsd, and VDSRd data respectively.
After that use save_all_results ipy to generate all visulized and statistical results.


## Results
### CRPS comparison
| ![space-1.jpg](/data/img/crps2012_whole_mean.png) | 
|:--:| 
| *Average CRPS Skill Scores across Australia for forecasts made in 2012* |


| ![space-1.jpg](/data/img/mae2012_whole_mean.png) | 
|:--:| 
| *Average MAE skill scores across Australia for daily precipitation fore-casts made on 48 different initialisation dates in 2012* |


| ![space-1.jpg](/data/img/crps2010_whole_mean.png) | 
|:--:| 
| *Average CRPS Skill Scores across Australia for forecasts made in 2010* |


| ![space-1.jpg](/data/img/mae2010_whole_mean.png) | 
|:--:| 
| *Average MAE skill scores across Australia for daily precipitation fore-casts made on 48 different initialisation dates in 2010* |


### Visual Results

| ![Watch the video](/data/img/2012_by_time_serise_fps5.gif) | 
|:--:| 
| *A* |


## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@inproceedings{kim2016accurate,
  title={Accurate image super-resolution using very deep convolutional networks},
  author={Kim, Jiwon and Lee, Jung Kwon and Lee, Kyoung Mu},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1646--1654},
  year={2016}
}
```
