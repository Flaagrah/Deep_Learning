from skimage.transform import resize
from skimage.io import imread

import numpy as np
import os

trainImagesFolder = "../trainImages/"
trainMasksFolder = "../trainMasks/"
testImagesFolder = "../testImages/"

def createTestInput():
    allImages = []
    for filename in os.listdir(testImagesFolder):
        imdir = testImagesFolder+filename
        img = imread(imdir)
        print("test img dim:")
        print(img.shape)
        allImages.append(img)

#Creates the input for the model. In format [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]
def createTrainingInput():
    allImages = []
    allLabels = []
    print("training input")
    for filename in os.listdir(trainImagesFolder):
        imdir = trainImagesFolder+filename
        immask = trainMasksFolder+filename
        img = np.array(imread(imdir)).astype(np.float32)
        mask = np.array(imread(immask)).astype(np.float32)
        
        allImages.append(img)
        print("img dim:")
        print(img.shape)
        print(img)
        allLabels.append(mask)
        print("label dim:")
        print(mask.shape)
        print(mask)

        #Drop the 4th dimension. https://www.kaggle.com/c/data-science-bowl-2018/discussion/47750
        #Maybe need to keep it later.
    #Normalize the labels.
    return np.asarray(allImages).astype(np.float32), np.asarray(allLabels).astype(np.float32)
