from skimage.transform import resize
from skimage.io import imread

import numpy as np
import os
from mainmodel import IMAGE_HEIGHT, IMAGE_WIDTH

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
        img = imread(imdir).astype(np.float32)
        mask = imread(immask).astype(np.float32)
        
        #Assumption is that each pixel ranges from white to black.
        img = img[:, :, 0]
        img = np.reshape(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        img = np.divide(img, 256.0).astype(np.float32)
        mask = [[0.0 if (element == 0.0) else 1.0 for element in row] for row in mask]
        mask = np.asarray(mask).astype(np.float32)
        
        allImages.append(img)
        allLabels.append(mask)

        #Drop the 4th dimension. https://www.kaggle.com/c/data-science-bowl-2018/discussion/47750
        #Maybe need to keep it later.
    #Normalize the labels.
    return np.asarray(allImages).astype(np.float32), np.asarray(allLabels).astype(np.float32)
