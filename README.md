# FAO_crop_boundary
Scalable workflow for crop boundary delineation using a pre-trained deep learning model

## Description  

This repository contains a scalable workflow for crop boundary delineation using CSIRO's [DECODE model](https://www.mdpi.com/2072-4292/13/11/2197) and [pre-trained weight](https://arxiv.org/abs/2201.04771). The workflow is implemented as Python notebooks that are desgined to run in the DE Africa analysis sandbox. Ensure your sandbox environment has the dependency packages listed in the 'requirements.txt' installed before running.

The notebooks are organized in three folders and numbered according to the order they should be run:  

* 0_Data_preparation: contains notebooks for querying and downloading monthly Planet basemap data, and transforming input and validation data into files of size and format required by the FracTAL-ResUNet model. Planet API key is required.

* 1_Identify_months_thresholds_model_evaluation: contains notebooks and Python scripts to apply the model with pre-trained weights over locations with validation data, evaluate the model predictions, and use the validation results to identify the most suitable months of mosaic data and the optimal thresholds for crop field instance segmentation.

* 2_Predict_all_postprocessing: contains notebooks to apply field extent and boundary identification and instance segmentation on all transfromed Planet input images, mask with DE Africa Cropland Extent Map and merge the results into final maps.
