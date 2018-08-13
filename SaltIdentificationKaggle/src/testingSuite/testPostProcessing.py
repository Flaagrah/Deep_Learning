import os
import numpy as np
from skimage.io import imread
from mainmodel import testImagesFolder, trainImagesFolder, trainMasksFolder, IMAGE_HEIGHT, postProcessing, IMAGE_WIDTH


def testProcessResults():
    results = np.zeros((2, IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float64)
    results[0][3][40] = 0.6
    results[0][3][41] = 0.4
    results[0][4][40] = 0.9
    results[1][3][30] = 0.5
    results[1][3][31] = 0.65
    results[1][4][30] = 0.45
    processed = postProcessing.processResults(results)
    processed = np.reshape(processed, (2, IMAGE_HEIGHT, IMAGE_WIDTH))
    print("TEST PROCESS RESULTS")
    print("_____________________")
    print(processed[0][0][0])
    print(processed[0][3][40])
    print(processed[0][3][41])
    print(processed[0][4][40])
    print(processed[1][3][30])
    print(processed[1][3][31])
    print(processed[1][4][30])
    
    
def testGenerateOutput():
    print("TEST GENERATE OUTPUT")
    print("_____________________")
    masks = np.zeros((2, IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float64)
    #for i in range(0, 5):
    masks[0][40][50:55] = np.ones((5), np.float64)
    #for j in range(0, 9):
    masks[1][45][55:64] = np.ones((9), np.float64)
    
    masks = np.reshape(masks, (2, IMAGE_HEIGHT*IMAGE_WIDTH))
    output = postProcessing.generateOutput(masks)
    print(output)











def main():
    testProcessResults()
    testGenerateOutput()
       
main()