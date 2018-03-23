import imageio
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.transform import resize

dataURL = '../Data/stage1_train/'
imagesDir = 'images/'
masksDir = 'masks/'
compressionLoc = '../Data/tmp.png'
testURL = '../Data/stage1_test/'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

def compress(img):
    img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return img

def normalizeInput(inputlist):
    normalizedList = []
    for img in inputlist:
        normalized = img/(256.0)
        normalizedList.append(normalized)
    return normalizedList

def trainLabels(rawmasks):
    for masklist in rawmasks:
        for mask in masklist:
            #Get a list of all of the boxes.
            print()

def main(unused_argv):
    #for filename in os.listdir(testURL):
     #   imdir = testURL+filename+'/'+imagesDir
      #  print(imread(imdir+os.listdir(imdir)[0]).shape)
    
    allimages = []
    allmasks = []
    alldims = []
    for filename in os.listdir(dataURL):
        imdir = dataURL+filename+'/'+imagesDir
        immasks = dataURL+filename+'/'+masksDir
        #imagefile = imageio.imread(imdir+os.listdir(imdir)[0])
        img = imread(imdir+os.listdir(imdir)[0])
        alldims.append(img.shape)
        img = compress(img)
        allimages.append(img)
        masks = []
        for m in os.listdir(immasks):
            #mask = imageio.imread(immasks+m)
            mask = imread(immasks+m)
            mask = compress(mask)
            maskindices = np.nonzero(mask)
            masks.append(maskindices)
            
        allmasks.append(masks)
        #Drop the 4th dimension. https://www.kaggle.com/c/data-science-bowl-2018/discussion/47750
        #Maybe need to keep it later.
    normalizedImages = normalizeInput(allimages)
    
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)