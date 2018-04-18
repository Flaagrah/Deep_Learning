import tensorflow as tf
import os
import numpy as np
import pandas
from . import model as model

def NormalizeWidthHeight(labels):
    #1) Find the mean of all width/heights.
    #2) Find the standard deviation.
    #3) Calculate sd for each width/height.
    
    rLabels = np.reshape(labels, (-1, int(model.IMAGE_HEIGHT/model.BOX_HEIGHT), int(model.IMAGE_WIDTH/model.BOX_WIDTH), 5))
    widthHeight = rLabels[:,:,:,3:]
    otherLabels = rLabels[:,:,:,0:3]
    
    widthHeight = np.sqrt(widthHeight)
    
    normalizedVars = np.concatenate([otherLabels, widthHeight], axis = -1)
    normalizedVars = normalizedVars.flatten()
    normalizedVars = np.asarray(normalizedVars)
    return normalizedVars

def NormalizeWidthHeightForAll(allLabels):
    normLabels = []
    normalized = None
    for i in range(0, len(allLabels)):
        normalized = NormalizeWidthHeight(allLabels[i])
        normLabels.append(normalized)
    
    return np.asarray(normLabels).astype(np.float32)

def unNormalize(labels):
    widthHeight = labels[:,:,3:]
    otherLabels = labels[:,:,0:3]
    widthHeight = np.multiply(widthHeight, widthHeight)
    unNormalLabels = np.concatenate([otherLabels, widthHeight], axis = -1)
    unNormalLabels = unNormalLabels.flatten()
    unNormalLabels = np.asarray(unNormalLabels)
    return unNormalLabels

def unNormalizeAll(labels):
    normLabels = []
    for i in range(0, len(labels)):
        normLabels.append(unNormalize(labels[i]))
    return normLabels