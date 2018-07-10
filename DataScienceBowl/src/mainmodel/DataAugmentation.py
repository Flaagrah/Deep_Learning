from skimage.transform import resize

import numpy as np
import tensorflow as tensorflow
import imageio
from mainmodel import BOX_HEIGHT, BOX_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH

#Darken the image.
def darken(flattenedInput, factor):
    return flattenedInput*factor

#Lighten the image.
def lighten(flattenedInput, factor):
    space = 1.0-flattenedInput
    newInput = flattenedInput + (space*factor)
    return newInput

#Shift the image.
def shift(flattenedInput, flattenedLabel, shiftHorizontal, shiftVertical):
    print()
    rows = int(IMAGE_HEIGHT/BOX_HEIGHT)
    columns = int(IMAGE_WIDTH/BOX_WIDTH)
    
    label = np.reshape(flattenedLabel, (-1,rows,columns,5))
    input = np.reshape(flattenedInput, (-1,IMAGE_HEIGHT,IMAGE_WIDTH,3))
    
    shiftLabel = 0
    shiftInput = 0
    #Shift right by specified number of units (unit = BOX_WIDTH)
    if shiftHorizontal > 0:
        #Shift image right.
        shiftLabel = label[:, :, 0:columns-shiftHorizontal, :]
        label[:, :, shiftHorizontal:columns, :] = shiftLabel
        #Fill left most space with zeros.
        label[:, :, 0:shiftHorizontal, :] = np.full((label.shape[0], rows, shiftHorizontal, 5), 0)
        #Shift Input right.
        shiftInput = input[:, :, 0:(IMAGE_WIDTH-int(BOX_WIDTH*shiftHorizontal)), :]
        input[:, :, int(BOX_WIDTH*shiftHorizontal):IMAGE_WIDTH, :] = shiftInput
        #Fill leftmost space with random colours.
        input[:, :, 0:int(BOX_WIDTH*shiftHorizontal), :] = np.random.rand(input.shape[0], IMAGE_HEIGHT, int(BOX_WIDTH*shiftHorizontal), 3)
        
    elif shiftHorizontal < 0:
        #Shift image left.
        shiftLabel = label[:, :, (shiftHorizontal*-1):columns, :]
        label[:, :, 0:(columns+shiftHorizontal), :] = shiftLabel
        #Fill rightmost space with zeros.
        label[:, :, (columns+shiftHorizontal):columns, :] = np.full((label.shape[0], rows, int(shiftHorizontal*-1), 5), 0)
        #Shift image left.
        shiftInput = input[:, :, int(BOX_WIDTH*(-1*shiftHorizontal)):IMAGE_WIDTH, :]
        input[:, :, 0:(IMAGE_WIDTH+int(BOX_WIDTH*shiftHorizontal)), :] = shiftInput
        #Fill rightmost space with random colours.
        input[:, :, (IMAGE_WIDTH+int(BOX_WIDTH*shiftHorizontal)):IMAGE_WIDTH, :] = np.random.rand(input.shape[0], IMAGE_HEIGHT, int(-BOX_WIDTH*shiftHorizontal), 3)
    
    #Shift down
    if shiftVertical > 0:
        shiftLabel = label[:, 0:rows-shiftVertical, :, :]
        label[:, shiftVertical:rows, :, :] = shiftLabel
        label[:, 0:shiftVertical, :, :] = np.full((label.shape[0], shiftVertical, columns, 5), 0)
        shiftInput = input[:, 0:(IMAGE_HEIGHT-int(BOX_HEIGHT*shiftVertical)), :, :]
        input[:, int(BOX_HEIGHT*shiftVertical):IMAGE_HEIGHT, :, :] = shiftInput
        input[:, 0:int(BOX_HEIGHT*shiftVertical), :, :] = np.random.rand(input.shape[0], int(BOX_HEIGHT*shiftVertical), IMAGE_WIDTH, 3)
    #Shift up
    elif shiftVertical < 0:
        shiftLabel = label[:, (shiftVertical*-1):rows, :, :]
        label[:, 0:(rows+shiftVertical), :, :] = shiftLabel
        label[:, (rows+shiftVertical):rows, :, :] = np.full((label.shape[0], (shiftVertical*-1), columns, 5), 0)
        shiftInput = input[:, BOX_HEIGHT*(-1*shiftVertical):IMAGE_HEIGHT, :, :]
        input[:, 0:(IMAGE_HEIGHT+int(BOX_HEIGHT*shiftVertical)), :, :] = shiftInput
        input[:, (IMAGE_HEIGHT+int(BOX_HEIGHT*shiftVertical)):IMAGE_HEIGHT, :, :] = np.random.rand(input.shape[0], int(-BOX_HEIGHT*shiftVertical), IMAGE_WIDTH, 3)
        
    return input, label

#Creates a data augmentation.   
def returnAugmentationForList(originalInput, originalLabel, originalDims):
    originalInput = np.reshape(originalInput, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    originalLabel = np.reshape(originalLabel, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
    totalInput = np.copy(originalInput)
    totalLabels = np.copy(originalLabel)
    totalDims = np.copy(originalDims)
    #Create vertical and horizontal shifts.
    for i in range(0, 4):
        horiInput, horiLabel = shift(np.copy(originalInput), np.copy(originalLabel), i, 0)
        totalInput = np.append(totalInput, horiInput, 0)
        totalLabels = np.append(totalLabels, horiLabel, 0)
        totalDims = np.append(totalDims, np.copy(originalDims), 0)
    
    origL = np.reshape(originalInput, (-1,IMAGE_HEIGHT,IMAGE_WIDTH,3))
    for i in range(0, 4):
        vertInput, vertLabel = shift(np.copy(originalInput), np.copy(originalLabel), 0, i)
        totalInput = np.append(totalInput, vertInput, 0)
        totalLabels = np.append(totalLabels, vertLabel, 0)
        totalDims = np.append(totalDims, np.copy(originalDims), 0)

    #Light and darken the image.
    print(vertInput[0][0][0][1])
    dark = darken(np.copy(originalInput), 0.5)
    totalInput = np.append(totalInput, dark, 0)
    totalLabels = np.append(totalLabels, np.copy(originalLabel), 0)
    totalDims = np.append(totalDims, np.copy(originalDims), 0)
    light = lighten(np.copy(originalInput), 0.5)
    totalInput = np.append(totalInput, light, 0)
    totalLabels = np.append(totalLabels, np.copy(originalLabel), 0)
    totalDims = np.append(totalDims, np.copy(originalDims), 0)
    
    return totalInput, totalLabels, totalDims

#Compress image to fixed size of IMAGE_HEIGHT, IMAGE_WIDTH
def compress(img):
    img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return img
    
#def main(unused_argv):
    #images = (np.load('images.npy'))
    #labels = np.load('labels.npy')
    #dims = np.load('dims.npy')
    
    #firstTwoImages = [compress(images[0]), compress(images[1])]
    #images = np.reshape(firstTwoImages, (-1, 256*256*3))
    #labels = labels[0:2]
    #dims = dims[0:2]
    #totalInput, totalLabels, totalDims = returnAugmentationForList(images, labels, dims)
    #print(totalInput.shape)
    #np.save('imagesTotal', totalInput.flatten())
    #np.save('labelsTotal', totalLabels.flatten())
    #np.save('dimsTotal', totalDims.flatten())
    
    
#tensorflow.app.run(main)