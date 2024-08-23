# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 21:52:04 2019

Python script to create movies, plot observations 
and create empty background label txt files

@author: Kim Bjerge (Made from scratch)
"""

import os
import cv2
from matchFilter.matchFilter import MatchFilter
from classifier.cnn_classifier import CnnClassifier
from loader.loader import load_predictions, getMonthDay, getMonth, getDay
import numpy as np

imgWidth = 6080
imgHeight = 3420

# Label names for plot
labelNames = ["A1-Coccinellidae", "B2-Coleoptera", "C3-Background", "D4-Bombus", "E5-Syrphidae", 
              "F6-Lepidoptera", "G7-Araneae", "H8-Formidicidae", "I9-Diptera", "J10-Hemiptera", 
              "K11-Isopoda", "L12-Unspecified", "N13-Hymenoptera", "O14-Orthoptera", "P15-Rhagnoycha_fulva", 
              "Q16-Satyrinae", "R17-Aglais_urticea", "S18-Odonata", "T19-Apis_mellifera"]

def save_predictions(predictions, resultFile):
    
    file = open(resultFile, 'a')
    for insect in predictions:
        
        line = insect['system'] + ','
        line += str(insect['camera']) + ','
        line += insect['dateStr'] + ','
        line += insect['timeStr'] + ','
        line += str(insect['prob']) + ','
        line += str(insect['class']) + ','
        line += str(insect['x1']) + ','
        line += str(insect['y1']) + ','
        line += str(insect['x2']) + ','
        line += str(insect['y2']) + ','
        line += str(insect['image']) + '\n'
        
        #print(line)
        file.write(line)
 
    file.close()

def filter_predict_crops(classifier, img_path, system, crops_dic_insect, crops_dic_back, predictions, width=imgWidth, height=imgHeight, create=True):
    
    matchFilter = MatchFilter(3)
    
    #threshold = 500 # Gray + diff + variance
    #threshold = 3.0 # Gray + SQDIFF_NORMED
    threshold = 16.0 # Gray + Hist + SQDIFF_NORMED
    
    #border = 30
    border = 10
    countNoMove = 0
    countMatch = 0
    countInsect = 0
    
    for idx in range(len(predictions)):
        
        insect = predictions[idx]
        if insect['class'] < 0: # Filtered objects due to no movement
            countNoMove += 1
            continue

        #className = labelNames[insect['class']-1]
        className = str(insect['class'])
        systemSplit = system.split('-')
        filePath = systemSplit[1] + '-' + systemSplit[2] + '-' +systemSplit[3] + '-'
        imgName = systemSplit[0] + '/' +  filePath + insect['name'].split('.J')[0]  + '-' + str(insect['x1']) + '-' + str(insect['y1']) + '-' + str(insect['x2']) + '-' +  str(insect['y2']) + '.jpg'
        #print(imgName)
        #print(img_path + insect['image'])
        tempVariance = matchFilter.calcFeatures(img_path + insect['image'], insect['x1'], insect['y1'], insect['x2'], insect['y2'])
        print(img_path+insect['image'])
        img = cv2.imread(img_path+insect['image'])
        w = (insect['w'] + border*2)
        h = (insect['h'] + border*2)
        if h > w: 
            wh2 = int(round(h/2))
        else:
            wh2 = int(round(w/2))
        
        #x1 = insect['x1']-border
        x1 = insect['xc']-wh2
        if x1 < 0:
           x1 = 0
        #x2 = insect['x2']+border
        x2 = insect['xc']+wh2
        if x2 >= width:
            x2 = width-1
        #y1 = insect['y1']-border
        y1 = insect['yc']-wh2
        if y1 < 0: 
            y1 = 0
        #y2 = insect['y2']+border
        y2 = insect['yc']+wh2
        if y2 >= height:
            y2 = height-1
        imgCrop = img[y1:y2, x1:x2,  :]
        #print(imgName)
        imgNameOnly = imgName.split('/')[1]
        if tempVariance < threshold: # Background image if low variance (between 300 - 600)
            if create:
                cv2.imwrite(crops_dic_back + imgNameOnly, imgCrop)
            predictions[idx]['class'] = -1
            countMatch += 1
        else:
            index, species, probability = classifier.makePrediction(imgCrop)
            
            # ResNet50v2
            #cv2.imwrite(crops_dic_insect + "Insect/tmp.jpg", imgCrop)
            #index, species = classifier.makePredictionFile(crops_dic_insect + 'Insect/tmp.jpg')
            
            predictions[idx]['class'] = index
            predictions[idx]['prob'] = int(np.round(probability*100))
            #print(imgName)
            if create:
                if os.path.exists(crops_dic_insect + species) == False:
                    print("Create directory:", crops_dic_insect + species)
                    os.mkdir(crops_dic_insect + species)
                cv2.imwrite(crops_dic_insect + species + '/' + imgNameOnly, imgCrop)
            countInsect += 1
                 
    return countInsect, countMatch, countNoMove, predictions
        
                
if __name__=='__main__': 
    
    
    #path = 'O:/Tech_TTH-KBE/NI_2/'
    path = '/mnt/Dfs/Tech_TTH-KBE/NI_2/'
    dir_csv = path + 'Data_CSV_NI21/2021-csv-MIE/'
    pathInsectCrop = './Data_2021_19cls_MIE/' # insect images
    pathBackCrop = './Data_2021_19cls_MIE/Background/' # background images
    
    threshold = [0, 0, 0, 0, 0, 0, 0, 0, 0] 
   
    modelFile = "./models_save/EfficientNetB4-softmax-19cls-75"
    print("Loading model file", modelFile)
    classifier = CnnClassifier(modelFile, labelNames) # 80 iterations, transfer learning, F1 0.8114  (TestData-20-6E)
    
    count_insects = 0
    count_noMove = 0
    count_Match = 0
    delayStart = False
    for file_dir in sorted(os.listdir(dir_csv)):
        if file_dir.endswith('csv'):
            if "Data_2021-12_01" in file_dir:
                delayStart = False
            if delayStart:
                print("Already analyzed", file_dir)
                continue

            result_file = dir_csv + file_dir
            data_year = file_dir.split('-')[0]
            predictions = load_predictions(result_file, selection="All", filterTime=15, threshold=threshold) 
            count, countMatch, countNoMove, predictions = filter_predict_crops(classifier, path + data_year + '/', file_dir.split('.')[0], pathInsectCrop, pathBackCrop, predictions, create=True)
            count_insects += count
            count_noMove += countNoMove
            count_Match += countMatch
            save_predictions(predictions, pathInsectCrop + 'result-2021-MIE.csv')
    
    print("No movement, Match background, Insects", count_noMove, count_Match, count_insects)
 