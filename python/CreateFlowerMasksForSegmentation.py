# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:43:41 2023

@author: Kim Bjerge
"""

import os
import cv2 as cv
import glob
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from scipy import ndimage as ndi
#from skimage import io

#from skimage.morphology import disk
#from skimage.segmentation import watershed
#from skimage.filters import rank
#from skimage.util import img_as_ubyte
#from skimage.color import rgb2gray, rgb2hsv
#from skimage import data
#from skimage.filters import threshold_otsu, threshold_local
#from skimage.util import img_as_ubyte
    
def cvMaskFlowers(pathImage, fileName, maskType='RGB', showMasks=False):
    
    img = cv.imread(pathImage)
    #dim = (704, 396) # Width, Height Reduced by a factor of 6
    #img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    if maskType=='HSV':
    
        # HSV colors           Hue,  Sat, Value
        lower_yellow = np.array([0,  180, 200]) # Value higher 180
        #upper_yellow = np.array([29, 255, 255])
        upper_yellow = np.array([30, 255, 255])
        lower_white = np.array([0,   0, 200]) #230
        upper_white = np.array([360, 50, 255]) #25
        lower_red = np.array([125,  65, 165])
        upper_red = np.array([205,  165, 255])

        #convert the BGR image to HSV colour space
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        # used to find limits using a sample image
        # hsvmean = np.mean(hsv[:,:,0])
        # hsvstd = np.std(hsv[:,:,0])
        # print("Hue min/max", hsvmean-2*hsvstd, hsvmean+2*hsvstd)
        # hsvmean = np.mean(hsv[:,:,1])
        # hsvstd = np.std(hsv[:,:,1])
        # print("Saturation min/max", hsvmean-2*hsvstd, hsvmean+2*hsvstd)
        # hsvmean = np.mean(hsv[:,:,2])
        # hsvstd = np.std(hsv[:,:,2])
        # print("Value min/max", hsvmean-2*hsvstd, hsvmean+2*hsvstd)
 
        #set the lower and upper bounds for the green hue
        #lower_green = np.array([50,100,50])
        #upper_green = np.array([70,255,255])
        mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
        mask_white = cv.inRange(hsv, lower_white, upper_white)
        mask_red = cv.inRange(hsv, lower_red, upper_red)
        
    else:
        # BGR colors
        lower_yellow = np.array([0,180,200])
        upper_yellow = np.array([10,255,255])
        lower_white = np.array([230,240,200])
        upper_white = np.array([255,255,255])
    
        #create a mask for green colour using inRange function
        #mask = cv.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv.inRange(img, lower_yellow, upper_yellow)
        mask_white = cv.inRange(img, lower_white, upper_white)
    
    # Computes releative area of yellow, white and all flowers 
    yellow_count = np.count_nonzero(mask_yellow)
    white_count = np.count_nonzero(mask_white)
    red_count = np.count_nonzero(mask_red)
    area_flowers = yellow_count + white_count + red_count
    area_total = mask_yellow.shape[0] * mask_yellow.shape[1]
    yellow_percentage = yellow_count/area_total
    white_percentage = white_count/area_total
    red_percentage = red_count/area_total
    flowers_percentage = area_flowers/area_total
    
    #perform bitwise and on the original image arrays using the mask
    resWhite = cv.bitwise_and(img, img, mask=mask_white)
    resYellow = cv.bitwise_and(img, img, mask=mask_yellow)
    resRed = cv.bitwise_and(img, img, mask=mask_red)
    resWY = cv.bitwise_or(resWhite, resYellow) 
    res = cv.bitwise_or(resWY, resRed)
    
    #display the images
    if showMasks == True:
        #create resizable windows for displaying the images
        cv.namedWindow("res", cv.WINDOW_NORMAL)
        cv.namedWindow("maskY", cv.WINDOW_NORMAL)
        cv.namedWindow("maskW", cv.WINDOW_NORMAL)
        cv.imshow("maskY", mask_yellow)
        cv.imshow("maskW", mask_white)
    
        if maskType=='HSV':
            cv.namedWindow("hsv", cv.WINDOW_NORMAL)
            cv.imshow("hsv", img)
        else:
            cv.namedWindow("bgr", cv.WINDOW_NORMAL)
            cv.imshow("bgr", img)
        
        cv.imshow("res", res)

        if cv.waitKey(0):
            cv.destroyAllWindows()   
    
    # Plot original image and flower mask
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24, 24),
                             sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    ax[0].set_title(pathImage)

    ax[1].imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
    flowersStr = "{:.2f}".format(flowers_percentage*100)
    ax[1].set_title('Flowers ' + fileName + ' (' + flowersStr + '%)' )    
        
    #for a in ax:
    #    a.axis('off')
    
    fig.tight_layout()
    #KBE??? 
    fig.savefig("../imagesTest/Data_2021_Dataset/ImgMskRed/" + fileName)
    plt.show()
    
    #cv.imwrite("../imagesTest/Data_2021_Dataset/ImagesFlowRed/" + fileName, img)
    greyMask = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    ret, blackWhiteImg = cv.threshold(greyMask, 50, 255, cv.THRESH_BINARY)
    cv.imwrite("../imagesTest/Data_2021_Dataset/MasksFlowRed/" + fileName.replace('JPG', 'PNG'), blackWhiteImg)

    return yellow_percentage, white_percentage, red_percentage, flowers_percentage
    
    
def generateFlowerMasks(path, showMasks=False):
    
    for directory_path in glob.glob(path):
        label = directory_path.split("\\")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.JPG")):
            print(img_path)     
            
            fileName = img_path.split('\\')[1]
            print(fileName)
            yellow, white, red, flowers = cvMaskFlowers(img_path, fileName, maskType='HSV', showMasks=showMasks)
            print(yellow, white, red, flowers)
            #img = io.imread(img_path)
            #plt.imshow(img)
            #plt.title(img_path)
            #plt.show()


def expMA(data, alpha=0.4):
#Expontentiel moving average filter    
    y = []
    for i in range(len(data)):
        if i > 0:
            ynew = alpha*data[i] + (1-alpha)*y[i-1]
        else:
            ynew = alpha*data[i]          
        y.append(ynew)   
    return y

def plotFlowers(filePath):
    
    imageFlowers = np.load(filePath)
    
    dates = []
    flowering = []
    white = []
    yellow = []
    camera = 'Unknown'
    for flowers in imageFlowers:
        
        if camera == 'Unknown':
            camera = flowers[0]

        if camera != flowers[0]:
            plt.plot(flowering, 'r-o')
            plt.plot(expMA(flowering), 'b-o')
            plt.title(camera)
            plt.xlabel("Day")
            plt.ylabel("Percentage flowers")
            plt.savefig("../Data_2021_Plots/" + camera + "_" + "flowers" + ".jpg")
            plt.show()
            #plt.plot(white, 'bo')
            #plt.show()
            #plt.plot(yellow, 'yo')
            #plt.show()
            camera = flowers[0]        
            dates = []
            flowering = []
            white = []
            yellow = []  
            
        dates.append(flowers[2][5:10])
        yellowPct = float(flowers[4])*100  
        whitePct = float(flowers[5])*100
        flowerPct = float(flowers[6])*100
        flowering.append(flowerPct)
        white.append(whitePct)
        yellow.append(yellowPct)
    
                
#%% MAIN
if __name__=='__main__':
    

    #img_path = "./RedSmall.jpg"
    #fileName = "RedSmall.jpg"
    #yellow, white, red, flowers = cvMaskFlowers(img_path, fileName, maskType='HSV',showMasks=False)
    #print(yellow, white, red, flowers)
    
    path = "O:/Tech_TTH-KBE/NI_2/Kim/imagesTest/Data_2021_Dataset/ImagesFlow/"
    generateFlowerMasks(path)

    

