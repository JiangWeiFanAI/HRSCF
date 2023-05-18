# VDSD


[Warren Jin](https://people.csiro.au/J/W/Warren-Jin), [Weifan Jiang](https://www.linkedin.com/in/jeffery-jiang-3b966615a/), Minzhe Chen, Ming Li, K.Shuvo Bakar, Quanxi Shao, [Downscaling Long Lead Time Daily RainfallEnsemble Forecasts through Deep Learning]([Full pdf paper](https://link.springer.com/content/pdf/10.1007/s00477-023-02444-x.pdf?pdf=button))

The Python code is built and tested on Windows 10  environment (Python3.7, PyTorch_1.3.0, netCDF4_1.5.3, basemap_1.2.1, matplotlib_3.1.3,libtiff_0.4.2, xarray, CUDA10.2, cuDNN7.2) with Rtx2070 GPU. And push to [NCI](https://nci.org.au/) for training and evaluation.

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Results](#results)
4. [Citation](#citation)


## Introduction
Skilful and localised daily weather forecasts for upcoming seasons are desired by climate-sensitive sectors like agriculture, construction, and mining. Various General circulation models routinely provide such long lead time ensemble forecasts, also known as seasonal climate forecasts (SCF), but require downscaling techniques to enhance their skills from historical observations. Traditional downscaling techniques, like quantile mapping (QM), learn empirical relationships from pre-engineered predictors. Deep-learning-based downscaling techniques automatically generate and select predictors but almost all of them focus on simplified situations where low-resolution images match well with high-resolution ones, which is not the case in ensemble forecasts. To downscale ensemble rainfall forecasts, we take a two-step procedure. As illustrated in the flow chart below, 

![Flow chart](/data/img/fig2flowChart4VDSD.jpg)

we first choose a suitable deep learning model, very deep super-resolution (VDSR), from several outstanding candidates, based on an ensemble forecast skill metric, continuous ranked probability score (CRPS). Secondly, via incorporating other climate variables as extra input, we develop and finalise a very deep statistical downscaling (VDSD) model based on CRPS. The VDSD network structure looks like:  

![VDSD](/data/img/fig3VDSDstructure.jpg)

Both VDSR and VDSD are tested on downscaling 60 km rainfall forecasts from the Australian Community Climate and Earth-System Simulator Seasonal model version 1 (ACCESS-S1) to 12 km with lead times up to 217 days. They are separately tested on two years' hindcast data with a bit results in [Results](#results).  

## Train
### Training data 

In 2017, the Australian Meteorology Bureau announced the next generation access series GCM, which was later installed on supercomputers in the office in 2018. A worldwide combined model, the seasonal climate and earth system Simulator (ACCESS-S), is based on the UK's global combined seasonal prediction system glosea5-gc2. The ACCESS-S contains 11 different ensemble members for seasonal forecasting and leading 217 days due to disruptions and improved ensemble technology, including ten disturbed members and one unperturbed centre member. 

Bureau of Meteorology Atmospheric high-resolution Regional Reanalysis for Australia(BARRA) is Australia's regional climate prediction and numerical climate forecasts models based on an Australian area, using ACCESS-R, Australia's first atmospheric reanalysis model. ACCESS-R employs the UKMO system other than ACCESS-S. In addition, any uncertainty is not considered in this system. i.e. no ensemble member is present.

In the trainingset, we used 60km Raw atmosphere grid ACCESS-S precipitation data as input and 12km BARRA-R data as the target.
All the data were stored on the project named [iu60](http://poama.bom.gov.au/) and [ma05](http://www.bom.gov.au/clim_data/rrp/BARRA_sample/), you need to require permission from [NCI](https://nci.org.au/) on the paths (g/data/iu60/) and (g/data/ma05/) respectively.

### pre-processing data
To reduce IO time, we cropped the data to the same geographic region, with longitude and latitude ranges are 112.9 to 154.25 and -43.7425 to -9.0, respectively. Go to the data_processing folder and type the following commands, and the data will be cropped and relocated:

 ```bash
    python3 transform_access_pr.py
    python3 transform_Barra_data.py

 ```

### Begin to train


cd to 'wj1671', run the following scriptto train a model.

```bash
    python train.py 
```

### Begin to evaluation
cd to '..' back to the main folder, Grab the best model to /save path, and use jupyter notebooks to generate climatology, Bicubic, VDSR and VDSD data respectively.
After that use save_all_results ipy to generate all visualisation and statistical results.

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

| ![Watch the video](/data/img/2012_by_time_serise.gif) | 
|:--:| 
| *Example of ensemble_1 comparison* |


## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{Jin.etal.VDSD,
title = {Downscaling Long Lead Time Daily Rainfall Ensemble Forecasts through Deep Learning},
author ={Huidong Jin and Weifan Jiang and Minzhe Chen and Ming Li and K. Shuvo Bakar and Quanxi Shao},
journal={Stochastic Environmental Research and Risk Assessment},
pages={1--19},
year={2023},
publisher={Springer},
doi={https://doi.org/10.1007/s00477-023-02444-x},
note={Accepted in April 2023}
}
```
