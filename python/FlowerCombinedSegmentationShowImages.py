# -*- coding: utf-8 -*-
"""
Created on Thu July  8 11:04:02 2023

@author: Kim Bjerge
"""

import os
import cv2 as cv
import glob
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms


validCameras2021 =  [
    "ECOS01",
    "ECOS02",
    "ECOS03",
    #"ECOS04", # Tilovers
    "ECOS05",
    "ECOS06",
    "ECOS07",
    "ECOS08",
    "ECOS09",
    "ECOS10",

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
    "NAIM21",
    "NAIM22",
    "NAIM23",
    "NAIM24",
    "NAIM25",
    "NAIM26",
    "NAIM27",
    "NAIM28",
    "NAIM29",
    "NAIM30",
    "NAIM31",
    "NAIM32",
    "NAIM33",
    "NAIM34",
    "NAIM35",
    "NAIM36",
    "NAIM37",
    "NAIM38",
    "NAIM39",
    "NAIM40",
    
    #"NAIM57", # Ekstra mindegade
    #"NAIM58",
    #"NAIM59",
    #"NAIM60"
    ]

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

# saveCameras = [ "NAIM01",
#                 "NAIM08",
#                 "NAIM18",
#                 "NAIM21",
#                 "NAIM25",
#                 "NAIM34"
#                 #"NAIM30",
#                 #"NAIM40",
#                 #"NAIM57"
#                 #"NAIM33",
#                 #"NAIM34",
#                 #"NAIM35",
#                 #"NAIM36",
#                 #"NAIM37",
#                 #"NAIM38",
#                 #"NAIM39",
#                 #"NAIM40",
#                 #"NAIM49",
#                 #"NAIM57",
#                 #"NAIM58",
#                 #"NAIM59",
#                 #"NAIM60"
#                 ]

def colorMaskFlowers(pathImage, imgSize, showMasks=False):
    
    img = cv.imread(pathImage)
    img = cv.resize(img, imgSize, interpolation=cv.INTER_AREA)
  
    # HSV colors           Hue,  Sat, Value
    lower_yellow = np.array([0,  180, 200]) # Value higher 180
    upper_yellow = np.array([30, 255, 255])
    lower_white = np.array([0,   0, 200]) 
    upper_white = np.array([360, 50, 255]) 
    lower_red = np.array([125,  65, 165])
    upper_red = np.array([205,  165, 255])

    #convert the BGR image to HSV colour space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv.inRange(hsv, lower_white, upper_white)
    mask_red = cv.inRange(hsv, lower_red, upper_red)

    # Computes releative area of yellow, white and all flowers 
    yellow_count = np.count_nonzero(mask_yellow)
    white_count = np.count_nonzero(mask_white)
    red_count = np.count_nonzero(mask_red)
    area_flowers = yellow_count + white_count + red_count
    area_total = mask_yellow.shape[0] * mask_yellow.shape[1]
    #yellow_percentage = yellow_count/area_total
    #white_percentage = white_count/area_total
    #red_percentage = red_count/area_total
    flowers_percentage = area_flowers/area_total
        
    #perform bitwise and on the original image arrays using the mask
    resWhite = cv.bitwise_and(img, img, mask=mask_white)
    resYellow = cv.bitwise_and(img, img, mask=mask_yellow)
    resRed = cv.bitwise_and(img, img, mask=mask_red)
    resWY = cv.bitwise_or(resWhite, resYellow) 
    maskedImage = cv.bitwise_or(resWY, resRed)

    # Plot original image and flower mask
    if showMasks:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24, 24),
                                 sharex=True, sharey=True)
        ax = axes.ravel()
        
        ax[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax[0].set_title(pathImage)
    
        ax[1].imshow(cv.cvtColor(maskedImage, cv.COLOR_BGR2RGB))
        flowersStr = "{:.2f}".format(flowers_percentage*100)
        ax[1].set_title('Flowers ' + pathImage + ' (' + flowersStr + '%)' )    
    
    return mask_red, mask_white, mask_yellow, flowers_percentage, img, maskedImage

def SegmentImageMethodPIL(filename, model, camera, date, imgSize, use2021data, threshold=0.3, plotResult=True):
    
    if os.path.exists(filename) == False:
        return 0, 0, 0, 0
    
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    input_image = input_image.resize(imgSize)
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Imagenet
   ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model /255?

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_image = output.squeeze(0).cpu()
    mask_image = output_image > threshold

    # Computes releative area of yellow, white and all flowers 
    area_flowers = mask_image.numpy().sum()
    area_total = imgSize[0] * imgSize[1]
    flowers_percentage = area_flowers/area_total
    #print(area_flowers, flowers_percentage)
    
    # Mask color segmented images with otput of segmantic segmentation
    mask_img = mask_image.numpy().astype(np.uint8)
    mask_red, mask_white, mask_yellow, color_flow_percentage, imgCV, maskedImage = colorMaskFlowers(filename, imgSize=imgSize)
    
    mask_and_red = cv.bitwise_and(mask_img, mask_red)
    mask_and_white = cv.bitwise_and(mask_img, mask_white)
    mask_and_yellow = cv.bitwise_and(mask_img, mask_yellow)
    red_count = np.count_nonzero(mask_and_red)
    white_count = np.count_nonzero(mask_and_white)
    yellow_count = np.count_nonzero(mask_and_yellow)
    yellow_percentage = yellow_count/area_total
    white_percentage = white_count/area_total
    red_percentage = red_count/area_total    
    combined_flow_percentage = (red_count + white_count + yellow_count)/area_total
    
    #perform bitwise and on the original image arrays using the mask
    resWhite = cv.bitwise_and(imgCV, imgCV, mask=mask_and_red)
    resYellow = cv.bitwise_and(imgCV, imgCV, mask=mask_and_white)
    resRed = cv.bitwise_and(imgCV, imgCV, mask=mask_and_yellow)
    resWY = cv.bitwise_or(resWhite, resYellow) 
    maskedCombinedImage = cv.bitwise_or(resWY, resRed)
     
    if use2021data:
        validCameras = validCameras2021
    else:
        validCameras = validCameras2020
        
    # Plot original image and flower mask
    #if camera in validCameras2021 and plotResult:
    if camera in validCameras and plotResult:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(24, 24), # nrows=4
                                 sharex=True, sharey=True)
        ax = axes.ravel()
        
        ax[0].imshow(input_image)
        ax[0].set_title(filename)
    
        # ax[1].imshow(mask_image)
        # flowersStr = "{:.2f}".format(flowers_percentage*100)
        # ax[1].set_title('Flowers ' + date + ' (' + flowersStr + '%)' )    

        # ax[2].imshow(cv.cvtColor(maskedImage, cv.COLOR_BGR2RGB))
        # flowersStr = "{:.2f}".format(color_flow_percentage*100)
        # ax[2].set_title('Flowers ' + date + ' (' + flowersStr + '%)' )   

        # ax[3].imshow(cv.cvtColor(maskedCombinedImage, cv.COLOR_BGR2RGB))
        # flowersStr = "{:.2f}".format(combined_flow_percentage*100)
        # ax[3].set_title('Flowers ' + date + ' (' + flowersStr + '%)' )           

        ax[1].imshow(cv.cvtColor(maskedCombinedImage, cv.COLOR_BGR2RGB))
        flowersStr = "{:.2f}".format(combined_flow_percentage*100)
        ax[1].set_title('Flowers ' + camera + ' ' + date + ' (' + flowersStr + '%)' )   
        
        fig.tight_layout()
        nameSplit = filename.split('/')
        imgName = nameSplit[4] + '-' + nameSplit[5] + '-'  + nameSplit[6] + '-'  + nameSplit[7].split('.')[0] + '-' + flowersStr.replace('.', '_') + '.jpg'
        if use2021data:
            fig.savefig("./Data_2021_SegCombined/" + imgName)
        else:
            fig.savefig("./Data_2020_SegCombined/" + imgName)
            
        plt.show()
        fig.clear()
        plt.close()
        plt.cla()
        plt.clf()

    return yellow_percentage, white_percentage, red_percentage, combined_flow_percentage
    
    
def browseTestImages(path, model, imgSize=(704, 396)):
    
    for directory_path in glob.glob(path):
        label = directory_path.split("\\")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.JPG")):
            print(img_path)     
            
            yellow, white, red, flowers = SegmentImageMethodPIL(img_path, model, "NAIM01", "2021:08:03", imgSize=imgSize, use2021data=True)
            print(yellow, white, red, flowers)


# Get list of flower images in usedCameras, pick random images if random=True
def getUsedCameras(images, usedCameras, selectRandom=False, numbers=500):
    

    dfImages = pd.DataFrame(images, columns =['camera', 'file', 'date', 'time'])
    dfUsedImages = dfImages[dfImages['time'].str.contains('12:00:')]
    dfUsedImages = dfUsedImages[dfUsedImages['camera'].isin(usedCameras)] 

    # usedImages = []
    # for imageData in images:
    #     camera = imageData[0]
    #     if camera in usedCameras:
    #         usedImages.append(imageData)

    length = len(dfUsedImages)
    usedImages = dfUsedImages
    print("Number of used flower images", length)
    if selectRandom and length > numbers:
        #Generate random numbers between 0 and length
        randomlist = random.sample(range(0, length), numbers)
        print(randomlist)       
        usedImages = dfUsedImages.iloc[randomlist]

    return usedImages.values.tolist()

# Browse all images based on meta data in O:\Tech_TTH-KBE\NI_2\Data_2021\*csv files
# Pick all images with time 12:00 
# if analyseImages == False then 
#       images with flowers stored to imagesTest/Data_2021f/
#       images without flowers stored to imagesTest/Data_2021n/
# if analyseImages == True then
#       image analyzed for flowers data appended an stored in FlowersInImages_HSV_1200_new.npy
def browseImage(path, model, use2021data, analyzeImages=True, imgSize=(704, 396)):
    
    # Temp file should be placed on local drive to make loading faster
    if use2021data:
        savedSortedImagesPath = "./Sorted_images_2021.npy" # Faster to save and load from local drive 2020
    else:
        savedSortedImagesPath = "./Sorted_images_2020.npy" # Faster to save and load from local drive 2020
    
    if os.path.exists(savedSortedImagesPath):
        print("Uses saved sorted image list from", savedSortedImagesPath)
        images = np.load(savedSortedImagesPath)
    else:
        images = []
        for file_dir in sorted(os.listdir(path)):
            if file_dir.endswith('csv'):
                print(file_dir)
                df = pd.read_csv(path+file_dir)
                #print(df)
                #sourceFiles = df.SourceFile
                #dateTime = df.DateTimeOriginal
                for index, row in df.iterrows():
                    sourceFile = row['SourceFile']
                    #camera = sourceFile.split('/')[5] # Camera from path, not always correct
                    camera = row['MakerNoteUnknownText'].split(':')[3].rstrip() # Camera from file meta data
                    print(camera, row['DateTimeOriginal'], row['SourceFile'])
                    date = row['DateTimeOriginal'].split(' ')[0]
                    time = row['DateTimeOriginal'].split(' ')[1]
                    images.append([camera, row['SourceFile'], date, time])
    
        images = sorted(images)
        np.save(savedSortedImagesPath, images)        
        #print(images)
    
    # Call this function to get used cameras and/or pick random images
    images = getUsedCameras(images, validCameras2020, selectRandom=True, numbers=505)
    #images = getUsedCameras(images, validCameras2021, selectRandom=True, numbers=500)
    
    imageFlowers = []
    skipCount = 0
    for imageData in images:
        #print(imageData[3][0:5])
        #if imageData[0] == 'BOR02':
        #    break
        camera = imageData[0]
        if imageData[3][0:5] == '12:00': #and camera in saveCameras: # '19:00'
            sourceFile = imageData[1]
            date = imageData[2]
            time = imageData[3]
            
            if analyzeImages == False:
                skipCount += 1
                if skipCount % 2 == 0: # Each 7th day or each 2th day
                    dstPathf = "../imagesTestSeg/Data_2021f/"
                    dstPathn = "../imagesTestSeg/Data_2021n/"
                    splitSrcFile = sourceFile.split("/")
                    dstFile = splitSrcFile[4] + '_' + splitSrcFile[5] + '_' + splitSrcFile[6] + '_' + splitSrcFile[7] 
                    #print(dstFile)
                    yellow, white, red, flowers = SegmentImageMethodPIL(sourceFile, model, camera, date, imgSize=imgSize, use2021data=use2021data)
                    if flowers > 0.02: # No flowers if less 2.0% or 1.0%
                        print("Flowers ", dstFile)
                        shutil.copyfile(sourceFile, dstPathf+dstFile)
                    else:
                        print("No flowers ", dstFile)
                        shutil.copyfile(sourceFile, dstPathn+dstFile)
                        
            else:
                # Analyze images
                print(imageData)
                yellow, white, red, flowers = SegmentImageMethodPIL(sourceFile, model, camera, date, imgSize=imgSize, use2021data=use2021data)
                print("Flowers %.2f (Yellow %.2f White %.2f Red %.2f)" % (flowers*100, yellow*100, white*100, red*100) )
                imageFlowers.append([camera, sourceFile, date, time, yellow, white, red, flowers])
            
    if use2021data:
        np.save("FlowersInImages_Combined_2021_1200_new.npy", imageFlowers)
    else:
        np.save("FlowersInImages_Combined_2020_1200_new.npy", imageFlowers)
        

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

def plotFlowers(filePath, use2021data):
    
    imageFlowers = np.load(filePath)
    
    dates = []
    flowering = []
    white = []
    yellow = []
    red = []
    camera = 'Unknown'
    for flowers in imageFlowers:
        
        if camera == 'Unknown':
            camera = flowers[0]

        if camera != flowers[0]:
            plt.plot(flowering, 'k-o', label="Flowers")
            plt.plot(white, 'c', label="White")
            plt.plot(yellow, 'y', label="Yellow")
            plt.plot(red, 'r', label="Red")
            plt.legend(loc="upper right")
            #plt.plot(expMA(flowering), 'b-o')
            plt.title(camera)
            plt.xlabel("Day")
            plt.ylabel("Percentage of flowers")
            if use2021data:
                plt.savefig("./Data_2021_PlotsCombined/" + camera + "_" + "flowers" + ".jpg")
            else:
                plt.savefig("./Data_2020_PlotsCombined/" + camera + "_" + "flowers" + ".jpg")
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
            red = []
            
        dates.append(flowers[2][5:10])
        yellowPct = float(flowers[4])*100  
        whitePct = float(flowers[5])*100
        redPct = float(flowers[6])*100
        flowerPct = float(flowers[7])*100
        flowering.append(flowerPct)
        white.append(whitePct)
        red.append(redPct)
        yellow.append(yellowPct)
    
# Computes the precision for flower area in images 
def calcFlowerPrecision(path):
    
    # Sum of all true positive flower pct above 0.1%
    totalTPFlowerPct = 0
    for fileName in os.listdir(path):
        if fileName.endswith('jpg'):
            print(fileName)
            fileNameSplit = fileName.split('-')
            flowerPct = float(fileNameSplit[4].split('.')[0].replace('_','.'))
            #if flowerPct >= 0.1:
            print(flowerPct)
            totalTPFlowerPct += flowerPct

    print("Evaluating false positive")
    # Sum of all false positive flower pct
    totalFPFlowerPct = 0
    for fileName in os.listdir(path + 'FP'):
        if fileName.endswith('jpg'):
            print(fileName)
            fileNameSplit = fileName.split('-')
            flowerPct = float(fileNameSplit[4].split('.')[0].replace('_','.'))
            print(flowerPct)
            totalFPFlowerPct += flowerPct

    precisionFLowers = totalTPFlowerPct / (totalTPFlowerPct + totalFPFlowerPct)
    
    return totalTPFlowerPct, totalFPFlowerPct, precisionFLowers
                                
#%% MAIN
if __name__=='__main__':
    
    use2021data = True
    analyseImages = False # Set to True when original raw camera images should be analysed, else plotting flower cover for each camera
    
    #print(calcFlowerPrecision('O:/Tech_TTH-KBE/NI_2/Kim/Data_2021_SegCombined_1200/'))
    #print(calcFlowerPrecision('O:/Tech_TTH-KBE/NI_2/Kim/Data_2020_SegCombined_1200/'))
    
    if analyseImages:
 
        #modelName = "C:/IHAK/SemanticSegmentation/weightsFlowers30NormBackRedV2b.pt" # Faster to load from local drive
        modelName = "weightsFlowers30NormBackRedV2b.pt" 
        print("Loading red, white and yellow flowers DeepLabV3 Normalized model", modelName)
        model = torch.load(modelName, map_location=torch.device('cpu'))
        model.eval()
    
        if use2021data:
            path = "O:/Tech_TTH-KBE/NI_2/Data_2021/" # Path to all images from all traps monitored in 2021
            #path = "/mnt/Dfs/Tech_TTH-KBE/NI_2/Data_2021/"
        else:
            path = "O:/Tech_TTH-KBE/NI_2/Data_2020/" # Path to all images from all traps monitored in 2020
            #path = "/mnt/Dfs/Tech_TTH-KBE/NI_2/Data_2020/"
            
        browseImage(path, model, use2021data, analyzeImages=True)
    else:
    
        if use2021data:
            fileName = "FlowersInImages_Combined_2021_1200_30_0_3v2b_Final.npy" # Same dataset as below, but camera fixed 2020
        else:
            fileName = "FlowersInImages_Combined_2020_1200_30_0_3v2b_Final.npy" # Same dataset as below, but camera fixed 2020
            
        plotFlowers(fileName, use2021data)
    
