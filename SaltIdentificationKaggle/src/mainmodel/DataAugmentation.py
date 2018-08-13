import numpy as np
from mainmodel import IMAGE_HEIGHT, IMAGE_WIDTH
from nltk.app.nemo_app import images

shiftUnitSize = 10

def shiftVertical(img, shiftBy):
    inp = np.zeros(img.shape).astype(np.float32)
    divIndex = int(shiftUnitSize*shiftBy)
    print(divIndex)
    if shiftBy > 0:
        upSide = np.copy(img[:, 0:(IMAGE_HEIGHT-divIndex), :])
        downSide = np.copy(img[:, (IMAGE_HEIGHT-divIndex):IMAGE_HEIGHT, :])
        inp[:, 0:divIndex, :] = downSide
        inp[:, divIndex:IMAGE_HEIGHT, :] = upSide
    else:
        divIndex = divIndex*-1
        upSide = np.copy(img[:, 0:divIndex, :])
        downSide = np.copy(img[:, divIndex:IMAGE_HEIGHT, :])
        inp[:, 0:(IMAGE_HEIGHT-divIndex), :] = downSide
        inp[:, (IMAGE_HEIGHT-divIndex):IMAGE_HEIGHT, :] = upSide
    return inp
    
def shiftHorizontal(img, shiftBy):
    inp = np.zeros(img.shape).astype(np.float32)
    divIndex = int(shiftUnitSize*shiftBy)
    if shiftBy > 0:
        leftSide = np.copy(img[:, :, 0:(IMAGE_WIDTH-divIndex)])
        rightSide = np.copy(img[:, :, (IMAGE_WIDTH-divIndex):IMAGE_WIDTH])
        inp[:, :, 0:divIndex] = rightSide
        inp[:, :, divIndex:IMAGE_WIDTH] = leftSide
    else:
        divIndex = divIndex*-1
        upSide = np.copy(img[:, :, 0:divIndex])
        downSide = np.copy(img[:, :, divIndex:IMAGE_HEIGHT])
        inp[:, :, 0:(IMAGE_HEIGHT-divIndex)] = downSide
        inp[:, :, (IMAGE_HEIGHT-divIndex):IMAGE_HEIGHT] = upSide
    return inp
   
def flipVertically(img):
    return np.flip(img, 2)
    
def turnCounterClockWise(images):
    inp = np.zeros(images.shape)
    for i in range(0, len(images)):
        img = images[i]
        for j in range(0, IMAGE_HEIGHT):
            inp[i][ :, j] = np.reshape(np.flip(img[j, :], 0), (IMAGE_HEIGHT))
    return np.reshape(np.asarray(inp).astype(np.float64), (-1, IMAGE_HEIGHT, IMAGE_WIDTH))
    

def mutateData(data, labels, rotation, shiftVertical, shiftHorizontal):
    reshapedData = np.reshape(data, (-1, IMAGE_HEIGHT, IMAGE_WIDTH))
    reshapedLabels = np.reshape(labels, (-1, IMAGE_HEIGHT, IMAGE_WIDTH))
    
    