from skimage.transform import resize

import numpy as np
import tensorflow as tensorflow
import imageio

def darken(flattenedInput, factor):
    return flattenedInput*factor
    
def lighten(flattenedInput, factor):
    space = 1.0-flattenedInput
    newInput = flattenedInput + (space*factor)
    return newInput
    
def shift(flattenedInput, flattenedLabel, shiftHorizontal, shiftVertical):
    print()
    label = np.reshape(flattenedLabel, (-1,16,16,5))
    input = np.reshape(flattenedInput, (-1,256,256,3))
       
    shiftLabel = 0
    shiftInput = 0
    if shiftHorizontal > 0:
        shiftLabel = label[:, :, 0:16-shiftHorizontal, :]
        label[:, :, shiftHorizontal:16, :] = shiftLabel
        label[:, :, 0:shiftHorizontal, :] = np.full((label.shape[0], 16, shiftHorizontal, 5), 0)
        shiftInput = input[:, :, 0:(256-16*shiftHorizontal), :]
        input[:, :, 16*shiftHorizontal:256, :] = shiftInput
        input[:, :, 0:16*shiftHorizontal, :] = np.random.rand(input.shape[0], 256, 16*shiftHorizontal, 3)
        
    elif shiftHorizontal < 0:
        shiftLabel = label[:, :, (shiftHorizontal*-1):16, :]
        label[:, :, 0:(16+shiftHorizontal), :] = shiftLabel
        label[:, :, (16+shiftHorizontal):16, :] = np.full((label.shape[0], 16, (shiftHorizontal*-1), 5), 0)
        shiftInput = input[:, :, 16*(-1*shiftHorizontal):256, :]
        input[:, :, 0:(256+16*shiftHorizontal), :] = shiftInput
        input[:, :, (256+16*shiftHorizontal):256, :] = np.random.rand(input.shape[0], 256, -16*shiftHorizontal, 3)
    
    if shiftVertical > 0:
        shiftLabel = label[:, 0:16-shiftVertical, :, :]
        label[:, shiftVertical:16, :, :] = shiftLabel
        label[:, 0:shiftVertical, :, :] = np.full((label.shape[0], shiftVertical, 16, 5), 0)
        shiftInput = input[:, 0:(256-16*shiftVertical), :, :]
        input[:, 16*shiftVertical:256, :, :] = shiftInput
        input[:, 0:16*shiftVertical, :, :] = np.random.rand(input.shape[0], 16*shiftVertical, 256, 3)
        
    elif shiftVertical < 0:
        shiftLabel = label[:, (shiftVertical*-1):16, :, :]
        label[:, 0:(16+shiftVertical), :, :] = shiftLabel
        label[:, (16+shiftVertical):16, :, :] = np.full((label.shape[0], (shiftVertical*-1), 16, 5), 0)
        shiftInput = input[:, 16*(-1*shiftVertical):256, :, :]
        input[:, 0:(256+16*shiftVertical), :, :] = shiftInput
        input[:, (256+16*shiftVertical):256, :, :] = np.random.rand(input.shape[0], -16*shiftVertical, 256, 3)
        
    return input, label
        
def returnAugmentationForList(originalInput, originalLabel, originalDims):
    originalInput = np.reshape(originalInput, (-1, 256, 256, 3))
    originalLabel = np.reshape(originalLabel, (-1, 16, 16, 5))
    totalInput = np.copy(originalInput)
    totalLabels = np.copy(originalLabel)
    totalDims = np.copy(originalDims)
    for i in range(0, 4):
        horiInput, horiLabel = shift(np.copy(originalInput), np.copy(originalLabel), i, 0)
        totalInput = np.append(totalInput, horiInput, 0)
        totalLabels = np.append(totalLabels, horiLabel, 0)
        totalDims = np.append(totalDims, np.copy(originalDims), 0)
    
    #imageio.imwrite('imgBefore.png', np.reshape(horiInput[1], (256, 256, 3)))
    origL = np.reshape(originalInput, (-1,256,256,3))
    for i in range(0, 4):
        vertInput, vertLabel = shift(np.copy(originalInput), np.copy(originalLabel), 0, i)
        totalInput = np.append(totalInput, vertInput, 0)
        totalLabels = np.append(totalLabels, vertLabel, 0)
        totalDims = np.append(totalDims, np.copy(originalDims), 0)
    #imageio.imwrite('imgAfter.png', np.reshape(vertInput[1], (256, 256, 3)))
    #print(vertLabel)
    print("after")
    print(vertInput[0][0][0][1])
    dark = darken(np.copy(originalInput), 0.5)
    totalInput = np.append(totalInput, dark, 0)
    totalLabels = np.append(totalLabels, np.copy(originalLabel), 0)
    totalDims = np.append(totalDims, np.copy(originalDims), 0)
    #imageio.imwrite('imageDarken.png', np.reshape(np.copy(vertInput[1]), (256, 256, 3)))
    light = lighten(np.copy(originalInput), 0.5)
    totalInput = np.append(totalInput, light, 0)
    totalLabels = np.append(totalLabels, np.copy(originalLabel), 0)
    totalDims = np.append(totalDims, np.copy(originalDims), 0)
    
    return totalInput, totalLabels, totalDims
    #imageio.imwrite('imageLighten.png', np.reshape(np.copy(vertInput[1]), (256, 256, 3)))

    
def compress(img):
    img = resize(img, (256, 256))
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