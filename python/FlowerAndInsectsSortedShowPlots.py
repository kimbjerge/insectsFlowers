# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:26:46 2023

@author: Kim Bjerge
"""

import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Label names for plot
#labelNames = ["Insect"]

validCameras2020 = [
    "NAIM01",
    "NAIM02",
    "NAIM03",
    "NAIM04",
    "NAIM05",
    "NAIM06",
    "NAIM07",
    "NAIM08",
    "NAIM09",
    "NAIM10",
    "NAIM11",
    "NAIM12",
    "NAIM13",
    "NAIM14",
    "NAIM15",
    "NAIM16",
    "NAIM17",
    "NAIM18",
    "NAIM19",
    "NAIM20",

    "NAIM33",
    "NAIM34",
    "NAIM35",
    "NAIM36",
    "NAIM37",
    "NAIM38",
    "NAIM39",
    "NAIM40",
    "NAIM41",
    "NAIM42",
    "NAIM43",
    "NAIM44",
    "NAIM45",
    "NAIM46",
    "NAIM47",
    "NAIM48",
    "NAIM49",
    "NAIM50",
    "NAIM51",
    "NAIM52",
    "NAIM53",
    "NAIM54",
    "NAIM55",
    "NAIM56"
    ]

validCameras2021 = {
    "BOR01"  : "",
    "BOR02"  : "",
    "BOR04"  : "",
    "BOR05"  : "",
    
    "ECOS01" : "Agriculture-1",
    "ECOS04" : "AgriNAculture-1",
    "ECOS05" : "AgriNAculture-1",
    "ECOS06" : "Agriculture-1",
    "ECOS07" : "Agriculture-2",
    "ECOS08" : "Agriculture-2",
    "ECOS09" : "Agriculture-2",
    "ECOS10" : "Agriculture-2",

    "NAIM01" : "Urban-1",
    "NAIM02" : "Urban-1",
    "NAIM03" : "Urban-1",
    "NAIM04" : "UrNAban-1",
    "NAIM05" : "Urban-3",
    "NAIM06" : "Urban-3",
    "NAIM07" : "Urban-2",
    "NAIM08" : "Urban-2",
    "NAIM09" : "Urban-3",
    "NAIM10" : "Urban-3",
    "NAIM11" : "Urban-2",
    "NAIM12" : "Urban-2",
    "NAIM13" : "Urban-4",
    "NAIM14" : "Urban-4",
    "NAIM15" : "Urban-4",
    "NAIM16" : "Urban-4",
    "NAIM17" : "Agriculture-3",
    "NAIM18" : "Agriculture-3",
    "NAIM19" : "Agriculture-3",
    "NAIM20" : "Agriculture-3",
    "NAIM21" : "Agriculture-4",
    "NAIM22" : "Agriculture-4",
    "NAIM23" : "Agriculture-4",
    "NAIM24" : "Agriculture-4",
    "NAIM25" : "Grassland-1",
    "NAIM26" : "Grassland-1",
    "NAIM27" : "Grassland-1",
    "NAIM28" : "Grassland-1",
    "NAIM29" : "Grassland-2",
    "NAIM30" : "Grassland-2",
    "NAIM31" : "Grassland-2",
    "NAIM32" : "Grassland-2",
    "NAIM33" : "Grassland-3",
    "NAIM34" : "Grassland-3",
    "NAIM35" : "Grassland-3",
    "NAIM36" : "Grassland-3",
    "NAIM37" : "Grassland-4",
    "NAIM38" : "Grassland-4",
    "NAIM39" : "Grassland-4",
    "NAIM40" : "Grassland-4",
    
    "NAIM49" : "",
    "NAIM52" : "",
    "NAIM57" : "", # Ekstra mindegade
    "NAIM58" : "",
    "NAIM59" : "",
    "NAIM60" : ""
}
    

# Class Id 0-18, Class Id = -2 same object in same position, Class Id = -1 match in last 3 images equals to no movement
labelNames = ["A1-Coccinellidae", "B2-Coleoptera", "C3-Background", "D4-Bombus", "E5-Syrphidae", 
              "F6-Lepidoptera", "G7-Araneae", "H8-Formidicidae", "I9-Diptera", "J10-Hemiptera", 
              "K11-Isopoda", "L12-Unspecified", "N13-Hymenoptera", "O14-Orthoptera", "P15-Rhagnoycha_fulva", 
              "Q16-Satyrinae", "R17-Aglais_urticea", "S18-Odonata", "T19-Apis_mellifera"]

def loadImagesMetaData(path):
    
    print('Loading image meta data')
    images = {}
    for file_dir in sorted(os.listdir(path)):
        if file_dir.endswith('csv'):
            print(file_dir)
            df = pd.read_csv(path+file_dir)
            #print(df)
            #sourceFiles = df.SourceFile
            #dateTime = df.DateTimeOriginal
            for index, row in df.iterrows():
                sourceFile = row['SourceFile']
                cameraDir = sourceFile.split('/')[5] # Camera from path, not always correct
                camera = row['MakerNoteUnknownText'].split(':')[3].rstrip() # Camera from file meta data
                imgIdx = sourceFile.find('Data_20')
                imgName = sourceFile[imgIdx+10:] # Position in path /Data_2021/*
                dateTime = row['DateTimeOriginal']
                if cameraDir != camera:
                    print("Replaced camera", cameraDir, camera, dateTime.replace(':', ''), imgName)
                images[imgName] = camera + ' ' + dateTime.replace(':', '')
            
    return images
   
def findCameraInMetaData(imagesMetaData, cameraDir, imgName):

    date = "0"
    time = "0"
    camera = cameraDir
    if imgName in imagesMetaData.keys():
        cameraDateTime = imagesMetaData[imgName] # Use camera name in meta data instead of directory name
        cameraDataTimeSplit = cameraDateTime.split(' ')
        camera = cameraDataTimeSplit[0]
        date = cameraDataTimeSplit[1]
        time = cameraDataTimeSplit[2]
        #print("Meta data found", imgName, cameraDir, camera, date, time)
    else:
        print("Meta data for image missing", imgName)
    
    return camera, date, time
  
        
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

def getDateStr(recDate):    
    return str(getDay(recDate)) + '/' + str(getMonth(recDate))

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
            if dist < 25: # NB adjusted for no object movement
                # Distance between center of boxes are very close
                # Then we assume it is not a new object
                newObject = False
    
    # Append new prediction to last preditions
    newLastPredictions.append(newPrediction)
    
    return newLastPredictions, newObject
    
# Load prediction CSV file
# filterTime specifies in minutes how long time window used
# to decide if predictions belongs to the same object
# probability threshold for each class, default above 50%
def load_predictions(imagesMetaData, filename, selection = 'All', filterTime=0):
    
    file = open(filename, 'r')
    content = file.read()
    file.close()
    splitted = content.split('\n')
    lines = len(splitted)
    foundObjects = []
    lastObjects = []
    for line in range(lines):
        subsplit = splitted[line].split(',')
        if len(subsplit) == 11: # required 11 data values
            imgname = subsplit[10]
            imgpath = imgname.split('/')
            prob = int(subsplit[4])
            objClass = int(subsplit[5])
            # Check selection 
            if (selection == imgpath[0] or selection == 'All'):
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
                
                system = subsplit[0]
                cameraDir = subsplit[1]
                camera, date, time = findCameraInMetaData(imagesMetaData, cameraDir, imgname) # Use camera name in meta data instead of directory name
                #if camera != cameraDir:
                #     print("Camera data used", imgname, camera, cameraDir)
                record = {
                        'system': system, 
                        'camera': camera,
                        'date' : int(date), #int(subsplit[2]), # Use date and time in image meta data
                        'time' : int(time), #int(subsplit[3]),
                        'prob' : prob, # Class probability 0-100%
                        'class' : objClass, # Classes -2, -1 and 0-18 
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
                        'label' : 0} # Class label (Unknown = 0)
                
                              # Camera,     date,      time,  pro,  class
                              #      0,        1,         2,    3,      4
                recordArray = [camera, int(date), int(time), prob, objClass]
                print(camera, date, time, prob, objClass, imgname)
                
                lastObjects, newObject =  filter_prediction(lastObjects, record, filterTime)
                if newObject:
                    foundObjects.append(recordArray)
            
    return foundObjects

def findInsects(camera, insectsList, datesFlowers, insectsToPlot):
    
    insectsCamera = np.zeros(len(datesFlowers))
    datesCamera = datesFlowers
    for insect in insectsList: # Used array index from function load_predictions
        if insect[0] == camera:
            classId = int(insect[4])
            if classId < 0:
                continue # Plant in background (-2, -1)
            className = labelNames[classId]
            if className in insectsToPlot:
                date = int(insect[1])
                time = int(insect[2])
                probability = int(insect[3])
                print(camera, date, '{:06d}'.format(time), '{:3d}'.format(probability), className)
                if date in datesFlowers:
                    idx = datesFlowers.index(date)
                    insectsCamera[idx] += 1
                else:
                    print("Insect found for camera:", camera, date)
            
    return datesCamera, insectsCamera


def findInsectsInCameras(cameras, insectsList, datesFlowers, insectsToPlot):
    
    insectsCamera = np.zeros(len(datesFlowers))
    datesCamera = datesFlowers
    for insect in insectsList: # Used array index from function load_predictions
        camera = insect[0]
        if camera in cameras:
            classId = int(insect[4])
            if classId < 0:
                continue # Plant in background (-2, -1)
            className = labelNames[classId]
            if className in insectsToPlot:
                date = int(insect[1])
                time = int(insect[2])
                probability = int(insect[3])
                print(camera, date, '{:06d}'.format(time), '{:3d}'.format(probability), className)
                if date in datesFlowers:
                    idx = datesFlowers.index(date)
                    insectsCamera[idx] += 1
                else:
                    print("Insect found for camera:", camera, date)
            
    return datesCamera, insectsCamera


def findBackground(camera, insectsList, datesFlowers):
    
    insectsCamera = np.zeros(len(datesFlowers))
    datesCamera = datesFlowers
    for insect in insectsList:
        if insect[0] == camera:
            classId = int(insect[4])
            #if classId != -1: # To plot match filtered
            #if classId != -2: # To plot no movement filtered
            if classId >= 0: # To plot all backgrounds
                continue # Plant in background (-2, -1)
            date = int(insect[1])
            if date in datesFlowers:
                idx = datesFlowers.index(date)
                insectsCamera[idx] += 1
            else:
                print("Insect found for camera:", camera, date)
            
    return datesCamera, insectsCamera    
        
def plotInsectFlowers(plotDir, insectLabel, imageFlowers, insectsList, insectsToPlot=[], plotBackgrounds=True, cameraToPlot = "All"):
     
    flowersPlot = []
    camera = 'Unknown'
    for flowers in imageFlowers:
        
        if camera == 'Unknown':
            camera = flowers[0]

        if camera != flowers[0]:
            print(camera)

            if camera == cameraToPlot or cameraToPlot == 'All':
                
                flowersPlot = sorted(flowersPlot) # sort plots according to date and time
                datesFlowers = [r[1] for r in flowersPlot]
                yellow = [r[2] for r in flowersPlot]
                white =  [r[3] for r in flowersPlot]
                red = [r[4] for r in flowersPlot]
                flowering = [r[5] for r in flowersPlot]
                
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 18),
                                         sharex=False, sharey=False)
                ax = axes.ravel()
                
                plt.rcParams.update({'font.size': 24})
              
                if plotBackgrounds:
                    datesInsects, insects = findBackground(camera, insectsList, datesFlowers)
                else:
                    datesInsects, insects = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                
                ax[0].plot(insects, 'k-o', label=insectLabel)
                ax[0].set_title("Camera " + camera)
                ax[0].set_xlabel("Day")
                ax[0].set_ylabel("Observations")
                ax[0].legend(loc="upper right")
                
                ax[1].plot(flowering, 'k-o', label="Flowers")
                ax[1].plot(white, 'c-.', label="White")
                ax[1].plot(yellow, 'y', label="Yellow")
                ax[1].plot(red, 'r--', label="Red")
                ax[1].legend(loc="upper right")
                #ax[0].plot(expMA(flowering), 'b-o')
                #ax[1].set_title(camera + " Flowers")
                ax[1].set_xlabel("Day")
                ax[1].set_ylabel("Percentage of flowers")
                if os.path.exists(plotDir) == False:
                    print("Create directory", plotDir)
                    os.mkdir(plotDir)
                plt.savefig(plotDir + camera + "_" + insectLabel + "_Flowers" + ".jpg")
                plt.show()
                #plt.plot(white, 'bo')
                #plt.show()
                #plt.plot(yellow, 'yo')
                #plt.show()

            camera = flowers[0]        
            flowersPlot = []
            
        dateStr = flowers[2].split(":") # Format YYYY:MM:DD
        date = int(dateStr[0])*10000 + int(dateStr[1])*100 + int(dateStr[2])
        yellowPct = float(flowers[4])*100  
        whitePct = float(flowers[5])*100
        redPct = float(flowers[6])*100
        flowerPct = float(flowers[7])*100
                                              #    1      2         3         4         5
        record = [flowers[2] + ' ' + flowers[3], date, yellowPct, whitePct, redPct, flowerPct] # data+time and parameters to plot
        flowersPlot.append(record)

    
def plotSpecialInsectFlowers(plotDir, insectLabel, imageFlowers, insectsList, plotPollinators=False):
     
    flowersPlot = []
    camera = 'Unknown'
    for flowers in imageFlowers:
        
        if camera == 'Unknown':
            camera = flowers[0]

        if camera != flowers[0]:
            print(camera)
            
            flowersPlot = sorted(flowersPlot)
            datesFlowers = [r[1] for r in flowersPlot]
            #checkAllDates(datesFlowers)
            yellow = [r[2] for r in flowersPlot]
            white =  [r[3] for r in flowersPlot]
            red = [r[4] for r in flowersPlot]
            flowering = [r[5] for r in flowersPlot]
            
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 18),
                                     sharex=False, sharey=False)
            ax = axes.ravel()
            
            plt.rcParams.update({'font.size': 24})
            
            if plotPollinators:
                insectsToPlot =  ["D4-Bombus", "E5-Syrphidae", "F6-Lepidoptera", "Q16-Satyrinae", "R17-Aglais_urticea", "T19-Apis_mellifera"]   
                datesInsects, pollinators = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                insectsToPlot =  ["D4-Bombus"]   
                datesInsects, bombus = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                insectsToPlot =  ["E5-Syrphidae"]   
                datesInsects, syrphidae = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                insectsToPlot = ["F6-Lepidoptera", "Q16-Satyrina", "R17-Aglais_urticea"]
                datesInsects, lepidoptera = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                #insectsToPlot = ["Q16-Satyrina"]
                #datesInsects, satyrina = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                #insectsToPlot = ["R17-Aglais_urticea"]
                datesInsects, aglais_urticea = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                insectsToPlot = ["T19-Apis_mellifera"]
                datesInsects, apis_mellifera = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                
                #ax[0].plot(insects, 'k-o', label=insectLabel)
                ax[0].plot(pollinators, 'k-o', label="Pollinators")
                ax[0].plot(bombus, 'c-.', label="Bumblebees")
                ax[0].plot(syrphidae, 'y', label="Hoverflies")
                ax[0].plot(lepidoptera, 'r--', label="Butterflies")
                #ax[0].plot(satyrina, 'm--', label="Satyrina")
                #ax[0].plot(aglais_urticea, 'm-.', label="A. urticea")
                ax[0].plot(apis_mellifera, 'g--', label="Honeybees")
            else:
                #insectsToPlot =  ["D4-Bombus", "E5-Syrphidae", "I9-Diptera", "N13-Hymenoptera", "R17-Aglais_urticea",  "T19-Apis_mellifera"]   
                #datesInsects, insects = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                insectsToPlot =  ["D4-Bombus", "E5-Syrphidae", "F6-Lepidoptera", "N13-Hymenoptera",  "R17-Aglais_urticea", "T19-Apis_mellifera"]   
                datesInsects, pollinators = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                insectsToPlot =  ["I9-Diptera"]   
                datesInsects, dipetra = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                insectsToPlot =  ["G7-Araneae"]   
                datesInsects, araneae = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                #insectsToPlot = ["S18-Odonata"]
                insectsToPlot = ["O14-Orthoptera"]
                datesInsects, odonata = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                #insectsToPlot = ["A1-Coccinellidae"]
                insectsToPlot = ["K11-Isopoda"]
                datesInsects, conccinellidae = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                insectsToPlot = ["H8-Formidicidae"]
                datesInsects, formidicidae = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                
                #ax[0].plot(insects, 'k-o', label=insectLabel)
                ax[0].plot(pollinators, 'k-o', label="Pollinators")
                ax[0].plot(araneae, 'c-.', label="Spiders")
                #ax[0].plot(odonata, 'y', label="Dragonflies")
                ax[0].plot(odonata, 'y', label="Grashoppers")
                ax[0].plot(dipetra, 'r--', label="Flies")
                #ax[0].plot(conccinellidae, 'm-.', label="Ladybirds")
                ax[0].plot(conccinellidae, 'm-.', label="Isopods")
                ax[0].plot(formidicidae, 'g--', label="Ants")
            
            ax[0].set_title(camera + '  ' + getDateStr(datesInsects[0]) + '-' + getDateStr(datesInsects[-1]) )
            ax[0].set_xlabel("Day")
            ax[0].set_ylabel("Observations")
            ax[0].legend(loc="upper right")
            
            ax[1].plot(flowering, 'k-o', label="Flowers")
            ax[1].plot(white, 'c-.', label="White")
            ax[1].plot(yellow, 'y', label="Yellow")
            ax[1].plot(red, 'r--', label="Red")
            ax[1].legend(loc="upper right")
            #ax[0].plot(expMA(flowering), 'b-o')
            #ax[1].set_title(camera + " Flowers")
            ax[1].set_xlabel("Day")
            ax[1].set_ylabel("Percentage of flowers")
            if os.path.exists(plotDir) == False:
                print("Create directory", plotDir)
                os.mkdir(plotDir)
            plt.savefig(plotDir + camera + "_" + insectLabel + "_Flowers" + ".jpg")
            plt.show()
            #plt.plot(white, 'bo')
            #plt.show()
            #plt.plot(yellow, 'yo')
            #plt.show()
 
            camera = flowers[0] 
            flowersPlot = []
            
        dateStr = flowers[2].split(":") # Format YYYY:MM:DD
        date = int(dateStr[0])*10000 + int(dateStr[1])*100 + int(dateStr[2])
        yellowPct = float(flowers[4])*100  
        whitePct = float(flowers[5])*100
        redPct = float(flowers[6])*100
        flowerPct = float(flowers[7])*100
                                              #    1      2         3         4         5
        record = [flowers[2] + ' ' + flowers[3], date, yellowPct, whitePct, redPct, flowerPct] # data+time and parameters to plot
        flowersPlot.append(record)



def plotSpecialInsectFlowersInCameras(plotDir, insectLabel, imageFlowers, insectsList, habitat="Agriculture", plotPollinators=False):
     
    flowersPlot = []
    camera = 'Unknown'
    for flowers in imageFlowers:
        
        if camera == 'Unknown':
            camera = flowers[0]

        if camera != flowers[0]:
            #print(camera)
            print(camera, validCameras2021[camera])
            
            if habitat in validCameras2021[camera]: 
                
                flowersPlot = sorted(flowersPlot)
                datesFlowers = [r[1] for r in flowersPlot]
                #checkAllDates(datesFlowers)
                yellow = [r[2] for r in flowersPlot]
                white =  [r[3] for r in flowersPlot]
                red = [r[4] for r in flowersPlot]
                flowering = [r[5] for r in flowersPlot]
                
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 18),
                                         sharex=False, sharey=False)
                ax = axes.ravel()
                
                plt.rcParams.update({'font.size': 24})
                
                if plotPollinators:
                    insectsToPlot =  ["D4-Bombus", "E5-Syrphidae", "F6-Lepidoptera", "Q16-Satyrinae", "R17-Aglais_urticea", "T19-Apis_mellifera"]   
                    datesInsects, pollinators = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    insectsToPlot =  ["D4-Bombus"]   
                    datesInsects, bombus = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    insectsToPlot =  ["E5-Syrphidae"]   
                    datesInsects, syrphidae = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    insectsToPlot = ["F6-Lepidoptera", "Q16-Satyrina", "R17-Aglais_urticea"]
                    datesInsects, lepidoptera = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    #insectsToPlot = ["Q16-Satyrina"]
                    #datesInsects, satyrina = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    #insectsToPlot = ["R17-Aglais_urticea"]
                    datesInsects, aglais_urticea = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    insectsToPlot = ["T19-Apis_mellifera"]
                    datesInsects, apis_mellifera = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    
                    #ax[0].plot(insects, 'k-o', label=insectLabel)
                    ax[0].plot(pollinators, 'k-o', label="Pollinators")
                    ax[0].plot(bombus, 'c-.', label="Bumblebees")
                    ax[0].plot(syrphidae, 'y', label="Hoverflies")
                    ax[0].plot(lepidoptera, 'r--', label="Butterflies")
                    #ax[0].plot(satyrina, 'm--', label="Satyrina")
                    #ax[0].plot(aglais_urticea, 'm-.', label="A. urticea")
                    ax[0].plot(apis_mellifera, 'g--', label="Honeybees")
                else:
                    #insectsToPlot =  ["D4-Bombus", "E5-Syrphidae", "I9-Diptera", "N13-Hymenoptera", "R17-Aglais_urticea",  "T19-Apis_mellifera"]   
                    #datesInsects, insects = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    insectsToPlot =  ["D4-Bombus", "E5-Syrphidae", "F6-Lepidoptera", "N13-Hymenoptera",  "R17-Aglais_urticea", "T19-Apis_mellifera"]   
                    datesInsects, pollinators = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    insectsToPlot =  ["I9-Diptera"]   
                    datesInsects, dipetra = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    insectsToPlot =  ["G7-Araneae"]   
                    datesInsects, araneae = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    #insectsToPlot = ["S18-Odonata"]
                    insectsToPlot = ["O14-Orthoptera"]
                    datesInsects, odonata = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    #insectsToPlot = ["A1-Coccinellidae"]
                    insectsToPlot = ["K11-Isopoda"]
                    datesInsects, conccinellidae = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    insectsToPlot = ["H8-Formidicidae"]
                    datesInsects, formidicidae = findInsects(camera, insectsList, datesFlowers, insectsToPlot)
                    
                    #ax[0].plot(insects, 'k-o', label=insectLabel)
                    ax[0].plot(pollinators, 'k-o', label="Pollinators")
                    ax[0].plot(araneae, 'c-.', label="Spiders")
                    #ax[0].plot(odonata, 'y', label="Dragonflies")
                    ax[0].plot(odonata, 'y', label="Grashoppers")
                    ax[0].plot(dipetra, 'r--', label="Flies")
                    #ax[0].plot(conccinellidae, 'm-.', label="Ladybirds")
                    ax[0].plot(conccinellidae, 'm-.', label="Isopods")
                    ax[0].plot(formidicidae, 'g--', label="Ants")
                
                ax[0].set_title(validCameras2021[camera] + '-' + camera + '  ' + getDateStr(datesInsects[0]) + '-' + getDateStr(datesInsects[-1]) )
                ax[0].set_xlabel("Day")
                ax[0].set_ylabel("Observations")
                ax[0].legend(loc="upper right")
                
                ax[1].plot(flowering, 'k-o', label="Flowers")
                ax[1].plot(white, 'c-.', label="White")
                ax[1].plot(yellow, 'y', label="Yellow")
                ax[1].plot(red, 'r--', label="Red")
                ax[1].legend(loc="upper right")
                #ax[0].plot(expMA(flowering), 'b-o')
                #ax[1].set_title(camera + " Flowers")
                ax[1].set_xlabel("Day")
                ax[1].set_ylabel("Percentage of flowers")
                if os.path.exists(plotDir) == False:
                    print("Create directory", plotDir)
                    os.mkdir(plotDir)
                plt.savefig(plotDir + camera + "_" + insectLabel + "_Flowers" + ".jpg")
                plt.show()
                #plt.plot(white, 'bo')
                #plt.show()
                #plt.plot(yellow, 'yo')
                #plt.show()
     
            camera = flowers[0]
            flowersPlot = []
            
        dateStr = flowers[2].split(":") # Format YYYY:MM:DD
        date = int(dateStr[0])*10000 + int(dateStr[1])*100 + int(dateStr[2])
        yellowPct = float(flowers[4])*100  
        whitePct = float(flowers[5])*100
        redPct = float(flowers[6])*100
        flowerPct = float(flowers[7])*100
                                              #    1      2         3         4         5
        record = [flowers[2] + ' ' + flowers[3], date, yellowPct, whitePct, redPct, flowerPct] # data+time and parameters to plot
        flowersPlot.append(record)
     
        
#%% MAIN
if __name__=='__main__':
    
    useMIEdata = True
    use2021data = True

    if use2021data:
        imagesPath = "O:/Tech_TTH-KBE/NI_2/Data_2021/" # Only used to create sorted npy files the first time
    else:
        imagesPath = "O:/Tech_TTH-KBE/NI_2/Data_2020/" # Only used to create sorted npy files the first time

    insectCSVPath = "/Data_CSV_Arthropods/"
    #savedSortedInsectsPath = "C:/IHAK/SemanticSegmentation/Sorted_insects-19cls-meta-data.npy"
    if useMIEdata:
        if use2021data:
            savedSortedInsectsPath = "Sorted_insects-19cls-MIE-meta-data-csv-2021.npy"
            csvFileName = "2021-MIE.csv"
        else:
            savedSortedInsectsPath = "Sorted_insects-19cls-MIE-meta-data-csv-2020.npy"        
            csvFileName = "2020-MIE.csv"
    else:
        if use2021data:
            savedSortedInsectsPath = "Sorted_insects-19cls-meta-data-csv-2021.npy"
            csvFileName = "2021_19cls.csv"
        else:
            savedSortedInsectsPath = "Sorted_insects-19cls-meta-data-csv-2020.npy"
            csvFileName = "2020_19cls.csv"
            
    if os.path.exists(savedSortedInsectsPath):
        print("Uses saved sorted insects list from", savedSortedInsectsPath)
        insectsList = np.load(savedSortedInsectsPath)
    else:   
        imagesMetaData = loadImagesMetaData(imagesPath)
        insectsList = []
        for insectCSVfile in os.listdir(insectCSVPath):
            if csvFileName in insectCSVfile:
                print(insectCSVfile)
                # No filter 351505 insects, 5 min 111057, 10 min 100440, 15 min 96393
                foundObjects = load_predictions(imagesMetaData, insectCSVPath+insectCSVfile, filterTime=15) 
                insectsList += foundObjects
            
        insectsList = sorted(insectsList)
        np.save(savedSortedInsectsPath, insectsList)              
    
    #flowerPath = "FlowersInImages_Combined_1200_30_0_3v2b.npy" # Same dataset as above but retrained
    if use2021data:
        flowerPath = "FlowersInImages_Combined_2021_1200_30_0_3v2b_Final.npy" # Same dataset as above but retrained, with corrected cameras
        yearData = "2021"
    else:
        flowerPath = "FlowersInImages_Combined_2020_1200_30_0_3v2b_Final.npy" # Same dataset as above but retrained, with corrected cameras
        yearData = "2020"

    print(flowerPath)
    imageFlowers = np.load(flowerPath)
    
    insectLabel = "Insects"
    #insectLabel = "Pollinators"

    #insectsToPlot = ["A1-Coccinellidae", "B2-Coleoptera", "C3-Background", "D4-Bombus", "E5-Syrphidae", 
    #                 "F6-Lepidoptera", "G7-Araneae", "H8-Formidicidae", "I9-Diptera", "J10-Hemiptera", 
    #                 "K11-Isopoda", "L12-Unspecified", "N13-Hymenoptera", "O14-Orthoptera", "P15-Rhagnoycha_fulva", 
    #                 "Q16-Satyrinae", "R17-Aglais_urticea", "S18-Odonata", "T19-Apis_mellifera"]   
    insectsToPlot = ["A1-Coccinellidae", "B2-Coleoptera", "D4-Bombus", "E5-Syrphidae", 
                    "F6-Lepidoptera", "G7-Araneae", "H8-Formidicidae", "I9-Diptera", "J10-Hemiptera", 
                     "K11-Isopoda", "L12-Unspecified", "N13-Hymenoptera", "O14-Orthoptera", "P15-Rhagnoycha_fulva", 
                     "Q16-Satyrinae", "R17-Aglais_urticea", "S18-Odonata", "T19-Apis_mellifera"]   
    
    
    #plotDir = "../Data_2021_PlotsFlowersBackgrounds/" # Match and no movement filtered    
    #plotDir = "../Data_2021_PlotsFlowersBackgrounds-1/" # Match filtered    
    #plotDir = "../Data_2021_PlotsFlowersBackgrounds-2/" # No movement filtered   
    #plotDir = "../Data_2021_PlotsFlowersInsectsFiltered2_MIE/"

    #plotInsectFlowers(plotDir, insectLabel, imageFlowers, insectsList, insectsToPlot, plotBackgrounds=False, cameraToPlot='All') # 'All'

    if useMIEdata:
        #plotDir = "../Data_" + yearData + "_PlotsFlowersPollinators_MIE/" # plotPollinators=True
        #plotDir = "../Data_" + yearData + "_PlotsFlowersMixed_MIE/" # plotPollinators=False
        plotDir = "../Data_" + yearData + "_PlotsFlowersMixed_Habitat/" # plotPollinators=False
    else:
        #plotDir = "../Data_" + yearData + "_PlotsFlowersPollinators/" # plotPollinators=True
        plotDir = "../Data_" + yearData + "_PlotsFlowersMixed/" # plotPollinators=False
           
    #plotSpecialInsectFlowers(plotDir, insectLabel, imageFlowers, insectsList, plotPollinators=False)
    # Grassland, Urband Agriculture
    plotSpecialInsectFlowersInCameras(plotDir, insectLabel, imageFlowers, insectsList, habitat="Urban", plotPollinators=False)
    
    