# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 21:52:04 2019

Python script to create movies, plot observations 
and create empty background label txt files

@author: Kim Bjerge (Made from scratch)
"""

import os
import cv2
#import math
from matchFilter.matchFilter import MatchFilter
from classifier.cnn_classifier import CnnClassifier
from loader.loader import load_predictions, getMonthDay, getMonth, getDay
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

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
        

# def countPredictClasses(predictions):
#     countClasses = []
#     for i in range(len(labelNames)):
#         countClasses.append(0);
        
#     for object in predictions:
#         countClasses[object['class']-1] += 1
    
#     return countClasses    
    
# def findObjectsInImage(filename, predictions):
    
#     found = False
#     objectsFound = []
#     for object in predictions:
#         filepath = object['name']
#         if filepath == filename:
#             objectsFound.append(object)
#             found = True
    
#     return found, objectsFound

# def create_label_crops(img_path, crops_dic, labelObjects, width=imgWidth, height=imgHeight):
    
#     border = 0
#     for insect in labelObjects:
#         img = cv2.imread(img_path+insect['image'])
#         x1 = insect['x1']-border
#         if x1 < 0:
#            x1 = 0
#         x2 = insect['x2']+border
#         if x2 >= width:
#             x2 = width-1
#         y1 = insect['y1']-border
#         if y1 < 0: 
#             y1 = 0
#         y2 = insect['y2']+border
#         if y2 >= height:
#             y2 = height-1
#         imgCrop = img[y1:y2, x1:x2,  :]
#         className = labelNames[insect['class']-1]
#         imgName = insect['name'] + '-' + str(insect['xc']) + '-' + str(insect['yc']) + '.jpg'
#         print(className + '/' + imgName)
#         cv2.imwrite(crops_dic + className + '/' + imgName, imgCrop)
    
# def create_filtered_crops(img_path, system, crops_dic_insect, crops_dic_back, predictions, width=imgWidth, height=imgHeight, create=True):
    
#     matchFilter = MatchFilter(3)
#     #threshold = 500 # Gray + diff + variance
#     #threshold = 3.0 # Gray + SQDIFF_NORMED
#     threshold = 16.0 # Gray + Hist + SQDIFF_NORMED
    
#     #border = 30
#     border = 10
#     countBack = 0
#     countInsect = 0
#     for insect in predictions:
#         if insect['class'] < 0: # Filtered objects due to no movement
#             continue

#         #className = labelNames[insect['class']-1]
#         className = str(insect['class'])
#         systemSplit = system.split('-')
#         filePath = systemSplit[1] + '-' + systemSplit[2] + '-' +systemSplit[3] + '-'
#         imgName = systemSplit[0] + '/' +  filePath + insect['name'].split('.')[0]  + '-' + className + '-' + str(insect['xc']) + '-' + str(insect['yc']) + '-' + str(insect['w']) + '-' +  str(insect['h']) + '.jpg'
#         #print(imgName)
#         #print(img_path + insect['image'])
#         tempVariance = matchFilter.calcFeatures(img_path + insect['image'], insect['x1'], insect['y1'], insect['x2'], insect['y2'])
#         if create:
#             #print(img_path+insect['image'])
#             img = cv2.imread(img_path+insect['image'])
#             w = (insect['w'] + border*2)
#             h = (insect['h'] + border*2)
#             if h > w: 
#                 wh2 = int(round(h/2))
#             else:
#                 wh2 = int(round(w/2))
            
#             #x1 = insect['x1']-border
#             x1 = insect['xc']-wh2
#             if x1 < 0:
#                x1 = 0
#             #x2 = insect['x2']+border
#             x2 = insect['xc']+wh2
#             if x2 >= width:
#                 x2 = width-1
#             #y1 = insect['y1']-border
#             y1 = insect['yc']-wh2
#             if y1 < 0: 
#                 y1 = 0
#             #y2 = insect['y2']+border
#             y2 = insect['yc']+wh2
#             if y2 >= height:
#                 y2 = height-1
#             imgCrop = img[y1:y2, x1:x2,  :]
#             if tempVariance < threshold: # Background image if low variance (between 300 - 600)
#                 cv2.imwrite(crops_dic_back + imgName, imgCrop)
#                 countBack += 1
#             else:
#                 cv2.imwrite(crops_dic_insect + imgName, imgCrop)
#                 countInsect += 1
                 
#     return countInsect

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
    #for insect in predictions:
    
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
    
# def create_movie(movie_name, image_dic, predictions, fps=2, size = (1920,1080) ):
        
#     movie_writer = cv2.VideoWriter(movie_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
#     print(movie_name)
    
#     lastFilename = "noname.jpg"
#     for predict in predictions:
#         pathFilename = predict['image']
#         filename = predict['name']
#         if lastFilename != filename:
#             found, objects = findObjectsInImage(filename, predictions)
#             if found:
#                 img = cv2.imread(image_dic+pathFilename)
#                 for insect in objects:
#                     cv2.rectangle(img,(insect['x1'],insect['y1']-10),(insect['x2'],insect['y2']), (255,255,255), 8)
#                     insectName = labelNames[insect['class']-1] + '(' + str(insect['prob'])+ ')'
#                     y = int(round(insect['y1']-20))
#                     cv2.putText(img, insectName,(insect['x1'],y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 4, cv2.LINE_AA)
#                 cv2.putText(img, pathFilename, (40,80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 4, cv2.LINE_AA)
#                 print(pathFilename)
#                 #cv2.imshow('image',img)
#                 #v2.waitKey(0)
#                 #cv2.destroyAllWindows()
#                 img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
#                 movie_writer.write(img)
#             lastFilename = filename
            
#     movie_writer.release()

# def plot_detected_boxes(img, objects, downsize):

#     img = cv2.resize(img, (0,0), fx=1/downsize, fy=1/downsize) #, cv2.INTER_CUBIC)
#     for insect in objects:
#         x1 = int(round(insect['x1']/downsize))
#         y1 = int(round((insect['y1']-10)/downsize))
#         x2 = int(round(insect['x2']/downsize))
#         y2 = int(round(insect['y2']/downsize))
#         cv2.rectangle(img,(x1,y1),(x2,y2), (0,0,255), 4)
#         insectName = labelNames[insect['class']-1] + '(' + str(insect['prob'])+ ')'
#         y = int(round((insect['y1']-20)/downsize))
#         cv2.putText(img, insectName, (x1,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
#     return img

# # Show predictions found in predictions list
# # Use key=s to save empty lable txt file for background image with wrong predictions       
# def show_predictions(image_dic, predictions, downsize=1.5):
    
#     count = 0
#     for filename in os.listdir(image_dic):
#         if filename.endswith('.JPG') or filename.endswith('.jpg'):
#             found, objects = findObjectsInImage(filename, predictions)
#             if found:
#                 img = cv2.imread(image_dic+'/'+filename)
#                 img = plot_detected_boxes(img, objects, downsize)
#                 cv2.imshow('image',img)
#                 #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#                 #cv2.resizeWindow('image', w, h)
#                 #cv2.waitKey(0)
#                 key = cv2.waitKey(0)
#                 if key == ord('s'):
#                     name = filename.split('.')
#                     txtfilename = image_dic+'/'+name[0]+'.txt'
#                     print('Create lable file:', txtfilename)
#                     open(txtfilename, 'a').close()
#                     count = count + 1
#                 if key == ord('e'):
#                     print('Exit')
#                     cv2.destroyAllWindows()
#                     break
#                 cv2.destroyAllWindows()
    
#     return count

# # Show predictions found in predictions list where also lable txt file exist
# # Use key=s to delete lable txt file for if wrong background image        
# def show_labled_predictions(image_dic, predictions, downsize=1.5):
    
#     count = 0
#     for filename in os.listdir(image_dic):
#         if filename.endswith('.JPG') or filename.endswith('.jpg'):
#             name = filename.split('.')
#             txtfilename = image_dic+'/'+name[0]+'.txt'
#             found, objects = findObjectsInImage(filename, predictions)
#             if found and os.path.isfile(txtfilename):
#                 img = cv2.imread(image_dic+'/'+filename)
#                 img = plot_detected_boxes(img, objects, downsize)
#                 cv2.imshow('image',img)
#                 key = cv2.waitKey(0)
#                 if key == ord('s'):
#                     print('Delete lable file:', txtfilename)
#                     os.remove(txtfilename)
#                 else:
#                     count = count + 1
#                 if key == ord('e'):
#                     print('Exit')
#                     cv2.destroyAllWindows()
#                     break
#                 cv2.destroyAllWindows()
                
#     return count

# def show_xy_histogram3(predictions, area_name, width = imgWidth, height = imgHeight):
    
#     # Create data
#     g1xc = []
#     g1yc = []
#     g2xc = []
#     g2yc = []
#     g3xc = []
#     g3yc = []
#     g4xc = []
#     g4yc = []
#     g5xc = []
#     g5yc = []
#     g6xc = []
#     g6yc = []
#     g7xc = []
#     g7yc = []
#     for obj in predictions:
#         if obj['class'] == 1:
#             g1xc.append(obj['xc'])
#             g1yc.append(obj['yc'])
#         if obj['class'] == 2: # Honningbi
#             g2xc.append(obj['xc'])
#             g2yc.append(obj['yc'])
#         if obj['class'] == 3 or obj['class'] == 4: # Havehumle og sten/jordhumle
#             g3xc.append(obj['xc'])
#             g3yc.append(obj['yc'])
#         if obj['class'] == 5 or obj['class'] == 6: # Svirrefluer hvid+gul
#             g4xc.append(obj['xc'])
#             g4yc.append(obj['yc'])
#         if obj['class'] == 7: # Sommerfugl
#             g5xc.append(obj['xc'])
#             g5yc.append(obj['yc'])
#         if obj['class'] == 8: # Hveps
#             g6xc.append(obj['xc'])
#             g6yc.append(obj['yc'])
#         if obj['class'] == 9: # Droneflue
#             g7xc.append(obj['xc'])
#             g7yc.append(obj['yc'])
    
#     dataxc = (g1xc, g2xc, g3xc, g4xc, g5xc, g6xc, g7xc)
#     datayc = (g1yc, g2yc, g3yc, g4yc, g5yc, g6yc, g7yc)
#     colors = ("red", "green", "blue", "grey", "black", "yellow", "orange")
#     groups = ("Mariehøne", "Honningbi", "Humlebi", "Svirreflue", "Sommerfugl", "Hveps", "Droneflue")
    
#     # Create plot
#     fig = plt.figure(figsize=(15,10))
#     ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    
#     for dataxc, datayc, color, group in zip(dataxc, datayc, colors, groups):
#         ax.scatter(dataxc, datayc, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
    
#     plt.title('Scatter plot of insect locations in: ' + area_name)
#     plt.legend(loc=2)
#     plt.show()
#     fig.savefig(area_name)   
#     plt.close(fig)   

# # Creates a scatter plot of positions of insects found in images
# def areaScatterPlots(path, system_name, model_name, csv_file='', filterTime=15, show_scatter=False, make_movie=False, make_thumpnails=False, threshold=[50,50,50,50,50,50]):
    
#     directory = path + system_name + '/'
#     if csv_file == '':
#         result_file = directory + system_name +'-' + model_name + '.csv'
#     else:
#         result_file = directory + csv_file

#     countPredictions = []
#     dirNames = []
#     for img_dir in os.listdir(directory):
#         if not img_dir.endswith('txt') and not img_dir.endswith('csv'):
#             imgPath = img_dir.split('/')
#             dateCamera = imgPath[0]
#             #thumpnails_name = 'crops/' + system_name + '-'  + dateCamera + '-' + model_name + '.avi'
#             movie_name = 'movies/' + system_name + '-'  + dateCamera + '-' + model_name + '.avi'
#             area_name = 'areas/' + system_name + '-'  + dateCamera + '-' + model_name + '.jpg'
#             predictions = load_predictions(result_file, selection=dateCamera, filterTime=filterTime, threshold=threshold) 
#             if show_scatter and len(predictions) > 0:
#                 show_xy_histogram3(predictions, area_name)
#             countPredictions.append(countPredictClasses(predictions))
#             dirNames.append(img_dir)
#             if make_movie and len(predictions) > 0:
#                 print("Creating movie as:", movie_name)
#                 image_dic = directory + dateCamera + '/'
#                 create_movie(movie_name, image_dic, predictions) # Create movie with bounding box
#             if make_thumpnails and len(predictions) > 0:
#                 image_dic = directory + dateCamera + '/'
#                 print("Creating thumpnails images from:", image_dic)
#                 #crops_dic = 'predict'
#                 #create_crops(crops_dic, system_name, dateCamera, image_dic, predictions) # Create movie with bounding box
#                 #print("Creating thumpnails movie as:", thumpnails_name)
#                 #create_thumpnails(thumpnails_name, image_dic, predictions) # Create movie with bounding box
    
#     print("System:", system_name, "Model:", model_name)
#     print(labelNames)
#     print("-------------------------------------------------------------------------------")
#     i = 0
#     for dic in dirNames: 
#         print(dic, countPredictions[i])
#         i += 1
#     print("==============================")
    
#     countTotal = []
#     for i in range(len(labelNames)):
#         countTotal.append(0);
#     for i in range(len(countPredictions)):
#         for j in range(len(labelNames)):
#             countTotal[j] += countPredictions[i][j]      
            
#     return countTotal

# # Creates a movie with images of bounding boxes for insects found in directory "dateCamera"
# def createMovie(path, predictions, result_file, video_path, create=True):

#     movie_name = video_path + result_file.split('.')[0] + '.jpg'
#     show_xy_histogram3(predictions, movie_name)
#     if create:
#         image_dic = path
#         movie_name = video_path + result_file.split('.')[0] + '.avi'
#         create_movie(movie_name, image_dic, predictions) # Create movie with bounding boxes

# # Used to show predictions in images with possibility to create background images for training
# def showPredictions(path, system_name, model_name, dateCamera):

#     directory = path + system_name + '/'
#     result_file = directory + system_name +'-' + model_name + '.csv'

#     image_dic = directory + dateCamera + '/'
#     predictions = load_predictions(result_file, selection = dateCamera) 
#     count = show_predictions(image_dic, predictions) # Show images with bounding boxes, possible to save 's' empty lable txt file
#     #count = show_labled_predictions(image_dic, predictions) # Show images with bounding boxes, where label txt file exist, possible to delete 's' wrong background image 
#     print('Counted labels:', count);

# # Plots number of bees and svirrefluer as function of dates where insects found
# def plotInsectsDate(path, system_name, model_name, camera, csv_file='', filterTime=15, threshold=[50,50,50,50,50,50,50,50]):
    
#     directory = path + system_name + '/'
#     if csv_file == '':
#         result_file = directory + system_name +'-' + model_name + '.csv'
#     else:
#         result_file = csv_file
#     predictions = load_predictions(result_file, selection = 'All', filterTime = filterTime, threshold=threshold) 
    
#     currDate = 0
#     monthArray = []
#     marie = []
#     bees = []
#     humle = []
#     svirre = []
#     dayArray = []
#     idx = -1
#     for predict in predictions:
#         if camera == predict['camera']:
#             if currDate != predict['date']:
#                 currDate = predict['date']
#                 monthArray.append(getMonthDay(currDate))
#                 marie.append(0)
#                 bees.append(0)
#                 humle.appen(0)
#                 svirre.append(0)
#                 idx += 1
#                 dayArray.append(idx)
#             classObj = predict['class']
#             if classObj == 1: #mariehøne (1)
#                 marie[idx] += 1
#             if classObj == 2: #honnigbi (2)
#                 bees[idx] += 1
#             if classObj >= 3 and classObj <= 4: #stenhumle (3), jordhumle (4)
#                 humle[idx] += 1
#             if classObj >= 5 and classObj <= 6: #svirrehvid (5), svirregul (6)
#                 svirre[idx] += 1
  
#     fig = plt.figure(figsize=(17,15))
#     ax = fig.add_subplot(2, 1, 1, axisbg="1.0")         
#     ax.plot(dayArray, marie, 'ro', label='Mariehøne')
#     ax.plot(dayArray, bees, 'go', label='Honningbi')
#     ax.plot(dayArray, humle, 'bo', label='Humlebi')
#     ax.plot(dayArray, svirre, 'yo', label='Svirreflue')
#     ax.legend(loc=2)
#     ax.set_ylim(0, 500)
#     ax.set_xlabel('Dage')
#     ax.set_ylabel('Antal')
#     ax.set_title('Insekter fra ' + str(system_name) + ' Camera ' + str(camera))
#     ax.grid(True)
#     fig.tight_layout()
#     plt.show()
    
#     #fig = plt.figure(figsize=(15,15))
#     #ax = fig.add_subplot(2, 1, 1, axisbg="1.0")         
#     #ax.plot(dayArray, svirre, 'bo')
#     #ax.set_ylim(0, 500)
#     #ax.set_xlabel('Dage')
#     #ax.set_ylabel('Svirrefluer')
#     #ax.set_title('Svirrefluer fra ' + str(system_name) + ' Camera ' + str(camera))
#     #ax.grid(True)   
#     #fig.tight_layout()
#     #plt.show()

#     print("Dates:", monthArray) 
    
# # Create an array with month and day for whole periode
# def createPeriode(periode):
    
#     monthDayArray = []
#     currDate = periode[0]
#     while currDate <= periode[1]+1:
#         monthDayArray.append(currDate)
#         currDate += 1
#         month = getMonth(currDate)
#         if getDay(currDate) == 31 and (month == 6 or month == 9): # June and September 30 days
#             currDate += (100-30);
#         if getDay(currDate) == 32 and (month == 5 or month == 7 or month == 8): # May, July and August 31 days
#             currDate += (100-31);
            
#     return monthDayArray

# # Get index that belongs to date
# def getDateIdx(currMonthDay, monthDayArray):
    
#     for idx in range(len(monthDayArray)):
#         if currMonthDay == monthDayArray[idx]:
#             return idx

#     return 0

# # Fundtion to create format of x-axis
# globalMonthDayArray = createPeriode([624, 1030])

# @ticker.FuncFormatter
# def major_formatter(x, pos):
#     day = int(globalMonthDayArray[int(x)] % 100)
#     month = int(globalMonthDayArray[int(x)] / 100)
#     string =  "{}/{}-2019"
#     return string.format(day, month) #"%d" % day

            
# # Plots number of bees and svirrefluer as function of periode with all dates
# def plotInsectsPeriode(path, system_name, model_name, periode, camera, filterTime=15, threshold=[50,50,50,50,50,5,50,50]):
    
#     directory = path + system_name + '/'
#     result_file = directory + system_name +'-' + model_name + '.csv'
#     predictions = load_predictions(result_file, selection = 'All', filterTime = filterTime, threshold=threshold) 
    
#     #currDate = 0
#     monthArray = createPeriode(periode)
#     length = len(monthArray)
#     marie = np.zeros((length,), dtype=int)
#     bees = np.zeros((length,), dtype=int)
#     humle = np.zeros((length,), dtype=int)
#     svirre = np.zeros((length,), dtype=int)
#     dayArray = range(length)
#     idx = -1
#     for predict in predictions:
#         if camera == predict['camera']:
#             #if currDate != predict['date']:
#             #    currDate = predict['date']
#             #    monthArray.append(getMonthDay(currDate))
#             #    bees.append(0)
#             #    svirre.append(0)
#             #    idx += 1
#             #    dayArray.append(idx)
#             idx = getDateIdx(getMonthDay(predict['date']), monthArray)
#             classObj = predict['class']
#             if classObj == 1: #mariehøne (1)
#                 marie[idx] += 1
#             if classObj == 2: #honnigbi (2)
#                 bees[idx] += 1
#             if classObj >= 3 and classObj <= 4: #stenhumle (3), jordhumle (4)
#                 humle[idx] += 1
#             if classObj >= 5 and classObj <= 6: #svirrehvid (5), svirregul (6)
#                 svirre[idx] += 1
  
#     fig = plt.figure(figsize=(20,20))
#     ax = fig.add_subplot(2, 1, 1, axisbg="1.0")         
#     ax.plot(dayArray, marie, 'r', label='Mariehøns')
#     ax.plot(dayArray, bees, 'g', label='Honningbier')
#     ax.plot(dayArray, humle, 'b', label='Humlebier')
#     ax.plot(dayArray, svirre, 'y', label='Svirrefluer')
#     ax.legend(loc=2)
#     ax.xaxis.set_major_formatter(major_formatter)
#     ax.set_ylim(0, 500)
#     ax.set_xlabel('Dato')
#     ax.set_ylabel('Antal')
#     ax.set_title('Insekter fra ' + str(system_name) + ' kamera ' + str(camera))
#     ax.grid(True)
#     fig.tight_layout()
#     plt.show()
#     fig.savefig('insects/' + str(system_name) + '_' + str(camera) + '-' + str(model_name) + '.jpg')   
#     plt.close(fig)   
    
#     return [dayArray, monthArray, marie, bees, humle, svirre]

# # Plots the sum of all bees and svirre from cameras in list
# def plotAllInsectsPeriode(cameras, model_name):
    
#     first = True
#     for camera in cameras:
#         dayArray = camera[0]
#         #monthArray = camera[1]
#         marie = camera[2]
#         bees = camera[3]
#         humle = camera[4]
#         svirre = camera[5]
#         if first:
#             marieTotal = marie
#             beesTotal = bees
#             humleTotal = humle
#             svirreTotal = svirre
#             first = False
#         else:
#             for idx in dayArray:
#                 marieTotal[idx] += marie[idx]
#                 beesTotal[idx] += bees[idx]
#                 humleTotal[idx] += humle[idx]
#                 svirreTotal[idx] += svirre[idx]
            
#     fig = plt.figure(figsize=(20,20))
#     ax = fig.add_subplot(2, 1, 1, axisbg="1.0")         
#     ax.plot(dayArray, marieTotal, 'r', label='Mariehøns')
#     ax.plot(dayArray, beesTotal, 'g', label='Honningbier')
#     ax.plot(dayArray, humleTotal, 'b', label='Humlebier')
#     ax.plot(dayArray, svirreTotal, 'y', label='Svirrefluer')
#     ax.legend(loc=2)
#     ax.xaxis.set_major_formatter(major_formatter)
#     #ax.set_ylim(0, 600)
#     ax.set_xlabel('Dato')
#     ax.set_ylabel('Antal')
#     ax.set_title('Insekter fra 10 kameraer (1,05 m2)')
#     ax.grid(True)
#     fig.tight_layout()
#     plt.show()   
#     #print("Dates:", monthArray) 
#     fig.savefig("insects/TotalAllCameras-" + model_name + ".jpg")   
#     plt.close(fig)      
    
                
if __name__=='__main__': 
    
    
    #path = 'O:/Tech_TTH-KBE/NI_2/'
    path = '/mnt/Dfs/Tech_TTH-KBE/NI_2/'
    #dir_csv = path +'Data_YOLOv5m6_21/'
    #dir_csv = path +'Data_YOLOv5m6_21/part1/'
    #dir_csv = path + 'Data_YOLOv5m6_21/Data_2020-csv/testSlides/'
    dir_csv = path + 'Data_CSV_NI21/2021-csv-MIE/'
    #dir_csv = path +'Data_YOLOv5m6_21/Data_2021-csv/'
    pathInsectCrop = './Data_2021_19cls_MIE/' # insect images
    pathBackCrop = './Data_2021_19cls_MIE/Background/' # background images
    
    threshold = [0, 0, 0, 0, 0, 0, 0, 0, 0] 
   
    #modelFile = "./models_save/EfficientNetB4-ADAM-72"
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
            #count_predict += create_filtered_crops(path + data_year + '/', file_dir.split('.')[0], pathInsectCrop, pathBackCrop, predictions, create=True)
            count, countMatch, countNoMove, predictions = filter_predict_crops(classifier, path + data_year + '/', file_dir.split('.')[0], pathInsectCrop, pathBackCrop, predictions, create=True)
            count_insects += count
            count_noMove += countNoMove
            count_Match += countMatch
            save_predictions(predictions, pathInsectCrop + 'result-2021-MIE.csv')
            #createMovie(path + data_year + '/', predictions, file_dir, 'F:/NI_2/YOLOV5crop/Video-I/', create=False)
    
    print("No movement, Match background, Insects", count_noMove, count_Match, count_insects)
 
    
    
 
