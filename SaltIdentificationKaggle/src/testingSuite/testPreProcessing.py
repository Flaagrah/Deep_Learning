'''
Created on Jul 30, 2018

@author: Bhargava
'''
import os
import numpy as np
from skimage.io import imread
from mainmodel import testImagesFolder, trainImagesFolder, trainMasksFolder, IMAGE_HEIGHT, preProcessing, IMAGE_WIDTH


def testTrainingInput():
    filename = os.listdir(trainImagesFolder)[1]
    #filename = filename[0:(len(filename)-4)]
    print("________________________________")
    print("Testing training input creation:")
    print(filename)
    imdir = trainImagesFolder+filename
    img = imread(imdir).astype(np.float64)
    print(img.shape)
    print("IMAGE")
    for i in range(0, 5):
        print(img[i])
    
    immask = trainMasksFolder+filename
    mask = imread(immask).astype(np.float64)
    print("MASK")
    for j in range(0, 5):
        print(mask[j])
    
    imgs, labels = preProcessing.createTrainingInput()
    itemp = imgs[1]
    ltemp = labels[1]
    print(ltemp[0])
    for j in range(0, IMAGE_HEIGHT):
        for k in range(0, IMAGE_WIDTH):
            if not float(img[j][k][0]/255.0) == itemp[j][k]:
                print("ERROR in testTrainingInput image pixel mismatch: 0 "+str(j)+" "+str(k))
            if not ((mask[j][k]==0.0 and ltemp[j][k]==0.0) or ltemp[j][k]==1.0) :
                print("ERROR in testTrainingInput mask pixel mismatch: 0 "+str(j)+" "+str(k))
    
def testTestingInput():
    filename = os.listdir(testImagesFolder)[0]
    print("___________________________________")
    print("Testing testing input creation:")
    print(filename)
    imdir = testImagesFolder+filename
    img = imread(imdir).astype(np.float64)
    print(img.shape)
    print("IMAGE")
    for i in range(0, 5):
        print(img[i])
    
    imgs = preProcessing.createTestInput()
    itemp = imgs["0005bb9630"]
    print("hello")
    print(itemp.shape)
    print(img[:, :, 0:1].shape)
    if (not itemp.shape == (101, 101)):
        print("Error, input is the wrong shape")
    for j in range(0, IMAGE_HEIGHT):
        for k in range(0, IMAGE_WIDTH):
            if not (float(img[j][k][0]/255.0) == itemp[j][k]):
                print(str(img[j][k][0])+" "+str(img[j][k][0]/255.0)+" "+str(itemp[j][k]))
                print("ERROR in testTestingInput image pixel mismatch: 0 "+str(j)+" "+str(k))
    

def main():
    testTrainingInput()
    testTestingInput()
       
main()