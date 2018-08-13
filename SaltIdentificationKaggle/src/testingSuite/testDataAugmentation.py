import os
import numpy as np
from skimage.io import imread
from mainmodel import testImagesFolder, trainImagesFolder, trainMasksFolder, IMAGE_HEIGHT, postProcessing, IMAGE_WIDTH, DataAugmentation

def checkForOnes(images):
    print(images.shape)
    for i in range(0, len(images)):
        img = images[i]
        for j in range(0, IMAGE_HEIGHT):
            for k in range(0, IMAGE_WIDTH):
                if img[j][k] == 1.0:
                    indexStr = str(i) + ": " + str(j) + ", " + str(k)
                    print(indexStr)
            

def testShiftVertical(img):
    print("TEST VERTICAL SHIFT")
    print("______________________")
    vert1 = DataAugmentation.shiftVertical(img, 1)
    vert2 = DataAugmentation.shiftVertical(img, 2)
    vert3 = DataAugmentation.shiftVertical(img, -1)
    vert4 = DataAugmentation.shiftVertical(img, -2)
    return vert1, vert2, vert3, vert4

def testShiftHorizontal(img):
    print("TEST HORIZONTAL SHIFT")
    print("______________________")
    hori1 = DataAugmentation.shiftHorizontal(img, 1)
    hori2 = DataAugmentation.shiftHorizontal(img, 2)
    hori3 = DataAugmentation.shiftHorizontal(img, -1)
    hori4 = DataAugmentation.shiftHorizontal(img, -2)
    return hori1, hori2, hori3, hori4

def testTurnCounterClockwise(img):
    print("TURN COUNTER CLOCKWISE")
    print("________________________")
    turn1 = DataAugmentation.turnCounterClockWise(img)
    turn2 = DataAugmentation.turnCounterClockWise(turn1)
    turn3 = DataAugmentation.turnCounterClockWise(turn2)
    turn4 = DataAugmentation.turnCounterClockWise(turn3)
    return turn1, turn2, turn3, turn4

def main():
    images = np.zeros((2, IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float64)
    images[0][50][40] = 1.0
    images[0][50][41] = 1.0
    images[1][51][40] = 1.0
    images[1][52][40] = 1.0
    checkForOnes(images)
    vert1, vert2, vert3, vert4 = testShiftVertical(images)
    print("CHECK 1")
    checkForOnes(vert1)
    print("CHECK 2")
    checkForOnes(vert2)
    print("CHECK 3")
    checkForOnes(vert3)
    print("CHECK 4")
    checkForOnes(vert4)
    
    hori1, hori2, hori3, hori4 = testShiftHorizontal(images)
    print("CHECK 1")
    checkForOnes(hori1)
    print("CHECK 2")
    checkForOnes(hori2)
    print("CHECK 3")
    checkForOnes(hori3)
    print("CHECK 4")
    checkForOnes(hori4)
    
    t1, t2, t3, t4 = testTurnCounterClockwise(images)
    print("CHECK 1")
    checkForOnes(t1)
    print("CHECK 2")
    checkForOnes(t2)
    print("CHECK 3")
    checkForOnes(t3)
    print("CHECK 4")
    checkForOnes(t4)
    
main()