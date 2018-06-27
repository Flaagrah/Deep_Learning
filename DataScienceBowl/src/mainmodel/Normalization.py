import tensorflow as tf
import os
import numpy as np
import pandas

from mainmodel import BOX_HEIGHT as BOX_HEIGHT
from mainmodel import BOX_WIDTH as BOX_WIDTH
from mainmodel import IMAGE_HEIGHT as IMAGE_HEIGHT
from mainmodel import IMAGE_WIDTH as IMAGE_WIDTH

#Normalize the width and height by square rooting. The purpose is to make smaller values more visible.
def NormalizeWidthHeight(labels):
    
    rLabels = np.reshape(labels, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
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

#Undo normalization.
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