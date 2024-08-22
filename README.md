# Time-lapse monitoring of insects in floral enviroments #
This project contains Python code for processing time-lapse images from insect camera traps (detection, classification and floral cover estimation)
The repository contains Python code for steps in figure below. (The training datasets are not included only the trained models and weights)

The work and results are described in the paper: "A deep learning pipeline for time-lapse camera monitoring of floral environments and insect populations"
https://www.biorxiv.org/content/10.1101/2024.04.12.589205v2.full


![Alt text](ProcessPipeline.png)

## Python environment files ##
envreq.txt - environment requirements

condaInstall.sh - edit file to install conda environment on Linux

## Python source code files, configuration, models and scripts ##

### YOLOv5 weights for insect detection and localisation

The modified YOLOv5 code is an older version of the repository from: https://github.com/ultralytics/yolov5

### Getting started ###

1. Install the environment requirements condaInstall.sh (Linux with Anaconda)

2. Activate python environment

   - Anaconda: $ conda activate PyTorch_NI2

3. Run the python code to plot the abundance of arthropods and flower cover estimates
   (In the file it is possible to select year 2020 or 2021, detections with and without MIE, arthropods to plot)

   - python FlowerAndInsectsSortedShowPlots.py


## Training and testing insect detector model ##

### YOLOv5 object detector files ###
data - YOLO configuration yaml files

models - YOLO yaml models and code

utils - YOLO source code

### Training YOLOv5 insect detector ###
trainF1.py

trainInsectsMoths.sh

### Validate YOLOv5 insect detector ###
val.py

testInsectsMoths.sh

## Detecting, classifying and tracing insects ##

### Combined YOLOv5 detection, ResNet50 order and species classifier ###
detectClassifyInsects.py - Detector and order classifier

detectClassifySpecies.py - Detector, order and species classifier

insectMoths-bestF1-1280m6.pt - YOLOv5m6 medium model trained to detect insects

insectMoths-bestF1-1280s6.pt - YOLOv5s6 small model trained to detect insects

CSV - contains CSV files with detections and npy files with features

Content of *.csv files which contain lines for each detection (YYYYMMDD.csv):

	year,trap,date,time,detectConf,detectId,x1,y1,x2,y2,fileName

 
## Plotting, making movies and printing results ##
createMoveiCSV.py - Create movies based on the detection and classification CSV files without tracking

plotResults.py - Plotting results for tracking and order classifications

plotSampleTimeResults.py - Plotting results for comparing tracking and different time-lapse sampling intervals

plotStatistics.py - Calculating and printing statistics for tracking and order classifications

