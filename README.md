# FAO_crop_boundary
Scalable workflow for crop boundary delineation using pre-trained deep learning model
## Description  

This repository contains a scalable workflow for crop boundary delineation using CSIRO's [DECODE model](https://www.mdpi.com/2072-4292/13/11/2197) and [pre-trained weight](https://arxiv.org/abs/2201.04771). It is desgined to run in the DE Africa analysis sandbox. Ensure your sandbox environment have the dependency packages listed in the 'requirements.txt' installed before running. The repository contains:  

* 0_Data_preparation: contains notebooks for querying, downloading monthly Planet basemap data, preparing input and validation data to chunks at requested sizes and format as required to feed in the FracTAL-ResUNet model. Planet API key is required.  

* 1_Identify_months_thresholds_model_evaluation: contains notebooks and dependent modules to identify most suitable months and fine-tune field boundary and extent probabilities thresholds for crop field instance segmentation, using the validation dataset and pre-trained model weight. Evaluation of the model predictions and instance segmentation is also included.  

* 2_Predict_all_postprocessing: contains notebooks to apply predictions and instance segmentation on all Planet image chunks within the AOI and post-processing including masking the crop field instances using DE Africa crop mask layer. 
