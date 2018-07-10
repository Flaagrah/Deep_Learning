from skimage.transform import resize
from skimage.io import imread

import numpy as np
import os

import mainmodel.Normalization as Normalization
import mainmodel.DataAugmentation as DataAugmentation

from mainmodel import BOX_HEIGHT as BOX_HEIGHT
from mainmodel import BOX_WIDTH as BOX_WIDTH
from mainmodel import IMAGE_HEIGHT as IMAGE_HEIGHT
from mainmodel import IMAGE_WIDTH as IMAGE_WIDTH



dataURL = '../Data/stage1_train/'
imagesDir = 'images/'
masksDir = 'masks/'
compressionLoc = '../Data/tmp.png'
testURL = '../Data/stage2_test/stage2_test_final/'

#Given the compressed pixels of the mask, get the relevant details of the mask.
def maskDetails(images, first):
    maskInfo = np.zeros((len(images), 4))
    for i in range(0, len(images)):
        img = images[i]
        maskindices = np.nonzero(img)
            
        minHeight = 0
        maxHeight = 0
        rightMost = 0
        leftMost = 0
        if maskindices[0].size > 0:
            #Get a list of all of the boxes.
            minHeight = min(maskindices[0])
            maxHeight = max(maskindices[0])
            leftMost = min(maskindices[1])
            rightMost = max(maskindices[1])
        
        height = maxHeight - minHeight
        width = rightMost - leftMost
        centerX = (rightMost+leftMost)/2
        centerY = (maxHeight+minHeight)/2
        
        maskInfo[i][0] = height
        maskInfo[i][1] = width
        maskInfo[i][2] = centerY
        maskInfo[i][3] = centerX
    return maskInfo

def compress(img):
    img = resize(img, (BOX_HEIGHT, BOX_WIDTH))
    return img

#Given a list of masks and the coordinates, returns a bounding box for each yolo box.
#This represents the y label for a single image example in the dataset.
#Return labels for segment in format [flag, y, x, h, w]
def trainLabels(maskInfo):
    boxRows = int(BOX_HEIGHT/BOX_HEIGHT)
    boxColumns = int(BOX_WIDTH/BOX_WIDTH)
    boxes = np.zeros((boxRows, boxColumns, 5))
    for h in range(0, boxRows):
        minH = h*BOX_HEIGHT
        maxH = minH+BOX_HEIGHT
        for w in range(0, boxColumns):
            minW = w*BOX_WIDTH
            maxW = minW+BOX_WIDTH
            found = False
            for i in range(0, len(maskInfo)):
                y = maskInfo[i][2]
                x = maskInfo[i][3]
                if y < maxH and y >= minH and x < maxW and x >= minW and (maxH-minH) > 0 and (maxW-minW) > 0:
                    found = True
                    boxes[h][w][0]=1.0
                    boxes[h][w][1]=float((y-minH)/BOX_HEIGHT)
                    boxes[h][w][2]=float((x-minW)/BOX_WIDTH)
                    boxes[h][w][3]=float(maskInfo[i][0]/BOX_HEIGHT)
                    boxes[h][w][4]=float(maskInfo[i][1]/BOX_WIDTH)
                if found == True:    
                    break
            if found == False :
                boxes[h][w][0]=0.0
                boxes[h][w][1]=0.0
                boxes[h][w][2]=0.0
                boxes[h][w][3]=0.0
                boxes[h][w][4]=0.0
    return boxes

#Creates the input for the model. In format [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]
def createInput(isTrainingInput):
    allLabels = []
    allImages = []
    allDims = []
    allImageNames = []
    first = True
    url = dataURL
    if not isTrainingInput:
        url = testURL
    for filename in os.listdir(url):
        imdir = url+filename+'/'+imagesDir
        immasks = url+filename+'/'+masksDir
        img = imread(imdir+os.listdir(imdir)[0])
        allDims.append((img.shape[0], img.shape[1], 3))
        #Drop the 4th dimension if it exists. Only need RGB
        if (len(img.shape) == 3 and img.shape[2] == 4):
            img = img[:, :, 0:3]
        #Add 2nd and 3rd dimension if black and white
        elif len(img.shape) == 2:
            tmp = np.reshape(img, (img.shape[0], img.shape[1], 1))
            img = np.concatenate((tmp, tmp), axis=2)
            img = np.concatenate((img, tmp), axis=2)
        
        #Compress image to standard size. (IMAGE_HEIGHT, IMAGE_WIDTH))
        img = compress(img)
        allImages.append(img)
        allImageNames.append(filename)
        masks = []
        #If it's the training input, also create the labels.
        if isTrainingInput:
            for m in os.listdir(immasks):
                mask = imread(immasks+m)
                mask = compress(mask)
                masks.append(mask)
        
            masksInfo = maskDetails(masks, first)
            trainingLabels = trainLabels(masksInfo)
            flattenedLabels = trainingLabels.flatten()
            processed = []
            for n in range(0, len(flattenedLabels)):
                processed.append(flattenedLabels[n])
            allLabels.append(processed)
            first=False
        #Drop the 4th dimension. https://www.kaggle.com/c/data-science-bowl-2018/discussion/47750
        #Maybe need to keep it later.
    #Normalize the labels.
    if not allLabels == [] :
        l = Normalization.NormalizeWidthHeightForAll(allLabels)
        allLabels = l
    
    normalizedImages = np.asarray(allImages).astype(np.float32)
    allDims = np.asarray(allDims).astype(np.int32)
    allLabels = np.asarray(allLabels).astype(np.float32)
    #Save training input in file
    if (isTrainingInput):
        imgFileName = 'imagesTrain'
        labFileName = 'labelsTrain'
        dimFileName = 'dimsTrain' 
        np.save(imgFileName, normalizedImages)
        np.save(dimFileName, allDims)
        np.save(labFileName, allLabels)
        totalInput, totalLabels, totalDims = DataAugmentation.returnAugmentationForList(normalizedImages, allLabels, allDims)
        
        np.save(imgFileName+'Total', totalInput)
        np.save(dimFileName+'Total', totalDims)
        np.save(labFileName+'Total', totalLabels)
    else:
        imgFileName = 'imagesTest'
        nameFileName = 'imagesNamesTest'
        dimFileName = 'dimsTest'
        np.save(imgFileName, normalizedImages)
        np.save(dimFileName, allDims)
        np.save(nameFileName, allImageNames)
        