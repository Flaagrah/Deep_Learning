from skimage.transform import resize
from skimage.io import imread

import numpy as np
import os
from mainmodel import IMAGE_HEIGHT, IMAGE_WIDTH
from mainmodel import testImagesFolder, trainImagesFolder, trainMasksFolder


def shapeImage(img):
    img = img[:, :, 0]
    img = np.reshape(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return np.divide(img, 255.0).astype(np.float64)


def createTestInput():
    allImages = {}
    for filename in os.listdir(testImagesFolder):
        imdir = testImagesFolder+filename
        img = imread(imdir).astype(np.float64)
        img = shapeImage(img)
        name = filename[0:(len(filename)-4)]
        allImages[name] = np.array(img).astype(np.float64)
    return allImages

#Creates the input for the model. In format [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]
def createTrainingInput():
    allImages = []
    allLabels = []
    print("training input")
    for filename in os.listdir(trainImagesFolder):
        imdir = trainImagesFolder+filename
        immask = trainMasksFolder+filename
        img = imread(imdir).astype(np.float64)
        mask = imread(immask).astype(np.float64)
        
        #Assumption is that each pixel ranges from white to black.
        img = shapeImage(img)
        mask = [[0.0 if (element == 0.0) else 1.0 for element in row] for row in mask]
        mask = np.asarray(mask).astype(np.float64)
        
        allImages.append(img)
        allLabels.append(mask)

        #Drop the 4th dimension. https://www.kaggle.com/c/data-science-bowl-2018/discussion/47750
        #Maybe need to keep it later.
    #Normalize the labels.
    print(len(allImages))
    return np.asarray(allImages).astype(np.float64), np.asarray(allLabels).astype(np.float64)
