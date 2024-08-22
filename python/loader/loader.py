# -*- coding: utf-8 -*-
"""
Created on Wen 23 21:52:04 2022

Python script to load labels and predicions

@author: Kim Bjerge
"""

import os
import math

# Functions to get seconds, minutes and hours from time in predictions
def getSeconds(recTime):
    return int(recTime%100)

def getMinutes(recTime):
    minSec = recTime%10000
    return int(minSec/100)

def getHours(recTime):
    return int(recTime/10000)

# Functions to get day, month and year from date in predictions
def getDay(recDate):
    return int(recDate%100)

def getMonthDay(recDate):
    return int(recDate%10000)

def getMonth(recDate):
    return int(getMonthDay(recDate)/100)

def getYear(recDate):
    return int(recDate/10000)


# Substract filterTime in minutes from recTime, do not handle 00:00:00
def substractMinutes(recTime, filterTime):
    
    minute = getMinutes(recTime)
    
    newRecTime = recTime - int(filterTime)*100
    if minute < filterTime: # No space to substract filterTime
        newRecTime = newRecTime - 4000 # Adjust minutes
    
    return newRecTime

# Filter predictions - if the positions are very close and of same class
# Checked within filterTime in minutes (must be a natural number 0,1,2..60)
# It is is assumed that the new prediction belong to the same object
def filter_prediction(lastPredictions, newPrediction, filterTime):
    
    newObject = True
    
    # Filter disabled
    if filterTime == 0:
        return lastPredictions, newObject
    
    # Update last predictions within filterTime window
    timeWindow = substractMinutes(newPrediction['time'], filterTime)
    newLastPredictions = []
    for lastPredict in lastPredictions:
        # Check if newPredition is the same date as last predictions and within time window
        if (lastPredict['date'] == newPrediction['date']) and (lastPredict['time'] > timeWindow):
            newLastPredictions.append(lastPredict)
    
    # Check if new predition is found in last Preditions - nearly same position and class
    for lastPredict in newLastPredictions:
        # Check if new prediction is of same class
        if lastPredict['class'] == newPrediction['class']:
            xlen = lastPredict['xc'] - newPrediction['xc']
            ylen = lastPredict['yc'] - newPrediction['yc']
            # Compute distance between predictions
            dist = math.sqrt(xlen*xlen + ylen*ylen)
            #print(dist)
            if dist < 50: # NB adjusted for no object movement
                # Distance between center of boxes are very close
                # Then we assume it is not a new object
                #print("Time filter", newPrediction['name'], newPrediction['class'], newPrediction['time'])
                newObject = False
    
    # Append new prediction to last preditions
    newLastPredictions.append(newPrediction)
    
    return newLastPredictions, newObject
    
filteredPredictions = 0
totalPredictions = 0

# Load prediction CSV file
# filterTime specifies in minutes how long time window used
# to decide if predictions belongs to the same object
# probability threshold for each class, default above 50%
def load_predictions(filename, selection = 'All', filterTime=0, threshold=[0,0,0,0,0,0,0,0]):
    
    file = open(filename, 'r')
    content = file.read()
    file.close()
    splitted = content.split('\n')
    lines = len(splitted)
    foundObjects = []
    lastObjects = []
    #filteredPredictions = 0
    for line in range(lines):
        subsplit = splitted[line].split(',')
        if len(subsplit) == 11: # required 11 data values
            imgname = subsplit[10]
            imgpath = imgname.split('/')
            prob = int(subsplit[4])
            objClass = int(subsplit[5])
            # Check selection 
            if (selection == 'All') or (selection == "Insects" and objClass >= 0 and objClass < 10): # and prob >= threshold[objClass-1]:        
            #if (selection == imgpath[0] or selection == 'All') and prob < threshold[objClass-1]: # All others
                x1 = int(subsplit[6])
                y1 = int(subsplit[7])
                x2 = int(subsplit[8])
                y2 = int(subsplit[9])
                # Convert points of box to YOLO format: center point and w/h
                width = x2-x1
                height = y2-y1
                xc = x2 - round(width/2)
                if xc < 0: xc = 0
                yc = y2 - round(height/2)
                if yc < 0: yc = 0
                
                record = {'system': subsplit[0], # 1-5
                'camera': subsplit[1], # 0 or 1
                'date' : int(subsplit[2]),
                'dateStr' : subsplit[2],
                'time' : int(subsplit[3]),
                'timeStr' : subsplit[3],
                'prob' : prob, # Class probability 0-100%
                'class' : objClass, # Classes 1-6
                # Box position and size
                'x1' : x1,
                'y1' : y1,
                'x2' : x2,
                'y2' : y2,
                'xc' : xc,
                'yc' : yc,
                'w' : width,
                'h' : height,
                'image' : imgname,
                'name' : imgpath[3],
                'label' : 0} # Class label (Unknown = 0)
                
                lastObjects, newObject =  filter_prediction(lastObjects, record, filterTime)
                
                global totalPredictions, filteredPredictions
                totalPredictions += 1
                if newObject:
                    foundObjects.append(record)
                else:
                    record['class'] = -2
                    foundObjects.append(record)
                    filteredPredictions += 1
           
    print("Predictions total, filtered", totalPredictions, filteredPredictions)
    return foundObjects


# Load labled images txt files with number of classes    
def load_labels(dirname, classes=6, width=1920, height=1080):
    
    labelCounts = []
    for i in range(classes):
        labelCounts.append(0)
    
    labledObjects = []
    for filename in os.listdir(dirname):
        if (filename.endswith('.txt')):
            file = open(dirname+filename, 'r')
            content = file.read()
            file.close()
            splitted = content.split('\n')
            lines = len(splitted)
            for line in range(lines):
                subsplit = splitted[line].split(' ')
                if len(subsplit) == 5: # required: index x y w h
                    index = int(subsplit[0])
                    if index < classes:
                        imagename = filename.split('.')
                        xc = float(subsplit[1])*width;
                        yc = float(subsplit[2])*height;
                        w = float(subsplit[3])*width;
                        h = float(subsplit[4])*height;
                        x1 = xc - w/2;
                        y1 = yc - h/2;
                        x2 = xc + w/2;
                        y2 = yc + h/2;
                        record = {'class' : int(subsplit[0])+1,
                            'xc' : int(round(xc)),
                            'yc' : int(round(yc)),
                            'w' : int(round(w)),
                            'h' : int(round(h)),
                            'x1' : int(round(x1)),
                            'y1' : int(round(y1)),
                            'x2' : int(round(x2)),
                            'y2' : int(round(y2)),
                            'image' : imagename[0]+'.jpg',
                            'name' : imagename[0],
                            'found' : False}
                        labledObjects.append(record)
                        labelCounts[index] +=1
                    
    return labledObjects, labelCounts
    
    
 