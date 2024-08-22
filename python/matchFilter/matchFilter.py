# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:55:16 2022

Class to filter detections using match with temporal images

@author: Kim Bjerge
"""

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

class MatchFilter:
    
    def __init__(self, timeWindow):
        self.timeWindow = timeWindow 

    def plotCrops(self, imgCrops, imgDiff, value):
        
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        ax1.imshow(imgCrops[0]) #, cmap='gray')
        ax1.set_title('n-2')
        ax2.imshow(imgCrops[1]) #, cmap='gray')
        ax2.set_title('n-1')
        ax3.imshow(imgCrops[2]) #, cmap='gray')
        ax4.imshow(imgDiff) #, cmap='gray')
        plt.text(2, 10, f'{value:0.1f}', fontsize = 10, bbox = dict(facecolor = 'white', alpha = 0.8))       
        plt.show()
        
    def backgroundVariance(self, imgCrops):
        
        imgBack = imgCrops[0]/2 + imgCrops[1]/2  
        #imgDiff = imgCrops[2] - imgBack
        imgDiff = imgBack - imgCrops[2]
        imgDiffArea = imgDiff.shape[0]*imgDiff.shape[1]
        imgDiffAvg = np.sum(imgDiff)/imgDiffArea
        print(imgDiffAvg, np.var(imgDiff))
        
        variance = np.var(imgDiff)

        self.plotCrops(imgCrops, imgDiff, variance)
        
        return variance
     
    def backgroundNormDiff(self, imgCrops):
        
        #imgCrops[0] = cv2.blur(imgCrops[0], (3,3))
        #imgCrops[1] = cv2.blur(imgCrops[1], (3,3))
        #imgCrops[2] = cv2.blur(imgCrops[2], (3,3))
        
        imgBack = imgCrops[0]/2 + imgCrops[1]/2  
        imgDiff = imgBack - imgCrops[2]
        
        # Template maching using normalized sum of squared difference
        # https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be
        RxyNumerator = sum(sum(imgDiff ** 2))
        RxyDenominator = math.sqrt( sum(sum(imgBack ** 2)) * sum(sum(imgCrops[2] ** 2)) )
        Rxy = RxyNumerator / RxyDenominator
        
        #self.plotCrops(imgCrops, imgDiff ** 2, Rxy)
        
        return Rxy
    
    def calcFeatures(self, filename, x1, y1, x2, y2):
        
        img_path, filename = os.path.split(filename)
        #print(img_path, filename)
        
        filenumber = filename[4:8]
        #print(filenumber)
        number = int(filenumber)
        imgCrops = []
        for i in range(self.timeWindow):
            filenumber = number - int(self.timeWindow) + i + 1
            nextfilename = '/' + filename[0:4] + f'{filenumber:04d}' + '.JPG'
            if os.path.exists(img_path+nextfilename):
                #print(nextfilename)
                img = cv2.imread(img_path+nextfilename)
                imgCrop = img[y1:y2, x1:x2,  :]
                #imgHSV = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV_FULL)
                #imgGray = imgHSV[:,:,1] + imgHSV[:,:,2]
                imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                # Improves NormDiff when used
                imgHist = cv2.equalizeHist(imgGray)
                imgCrops.append(imgHist)
            else:
                print("Priovius image missing", filename)
                return 1000 # No previous images, return value for insect
        
        
        #return self.backgroundVariance(imgCrops)
        return self.backgroundNormDiff(imgCrops)
        
        