import imageio
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from bokeh.layouts import column
from absl.logging import info

dataURL = '../Data/stage1_train/'
imagesDir = 'images/'
masksDir = 'masks/'
compressionLoc = '../Data/tmp.png'
testURL = '../Data/stage1_test/'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

BOX_HEIGHT = 16
BOX_WIDTH = 16

def compress(img):
    img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return img

#perhaps normalize by mean and variance using tensorflow batch normalization?
def normalizeInput(inputlist):
    normalizedList = []
    for img in inputlist:
        normalized = img/(255.0)
        normalizedList.append(normalized)
    return normalizedList

#Given the compressed pixels of the mask, get the relevant details of the mask.
def maskDetails(images, first):
    maskInfo = np.zeros((len(images), 4))
    for i in range(0, len(images)):
        img = images[i]
        if (first==True):
            #print("img in mask:")
            print(img.shape)
        maskindices = np.nonzero(img)
        if (first==True):
            #print("maskindices")
            print(maskindices)
            #print("min maskindices[0]")
            print(min(maskindices[0]))
            print(max(maskindices[0]))
            print(min(maskindices[1]))
            print(max(maskindices[1]))
            
        minHeight = 0
        maxHeight = 0
        rightMost = 0
        leftMost = 0
        if maskindices[0].size > 0:
            #Get a list of all of the boxes.
            minHeight = min(maskindices[0])
            maxHeight = max(maskindices[0])
            leftMost = min(maskindices[1])
            rightMost = max(maskindices[1])
        
        height = maxHeight - minHeight
        width = rightMost - leftMost
        centerX = (rightMost+leftMost)/2
        centerY = (maxHeight+minHeight)/2
        
        maskInfo[i][0] = height
        maskInfo[i][1] = width
        maskInfo[i][2] = centerY
        maskInfo[i][3] = centerX
    return maskInfo

#Given a list of masks and the coordinates, returns a bounding box for each yolo box.
#This represents the y label for a single image example in the dataset.
def trainLabels(maskInfo):
    boxRows = int(IMAGE_HEIGHT/BOX_HEIGHT)
    boxColumns = int(IMAGE_WIDTH/BOX_WIDTH)
    boxes = np.zeros((boxRows, boxColumns, 5))
    for h in range(0, boxRows):
        minH = h*BOX_HEIGHT
        maxH = minH+BOX_HEIGHT
        for w in range(0, boxColumns):
            minW = w*BOX_WIDTH
            maxW = minW+BOX_WIDTH
            found = False
            for i in range(0, len(maskInfo)):
                y = maskInfo[i][2]
                x = maskInfo[i][3]
                if y < maxH and y >= minH and x < maxW and x >= minW and (maxH-minH) > 0 and (maxW-minW) > 0:
                    found = True
                    boxes[h][w][0]=1.0
                    boxes[h][w][1]=float((y-minH)/BOX_HEIGHT)
                    boxes[h][w][2]=float((x-minW)/BOX_WIDTH)
                    boxes[h][w][3]=float(maskInfo[i][0]/IMAGE_HEIGHT)
                    boxes[h][w][4]=float(maskInfo[i][1]/IMAGE_WIDTH)
                if found == True:    
                    break
    return boxes

def combineMasks(rawmasks):
    mergedMasks = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    for i in range(0, len(rawmasks)):
        mergedMasks = mergedMasks + rawmasks[i]
    return mergedMasks

def trainModel(unused_argv):
    allImages = []
    allLabels = []
    alldims = []
    first = True
    for filename in os.listdir(dataURL):
        print(filename)
        imdir = dataURL+filename+'/'+imagesDir
        immasks = dataURL+filename+'/'+masksDir
        #imagefile = imageio.imread(imdir+os.listdir(imdir)[0])
        img = imread(imdir+os.listdir(imdir)[0])
        alldims.append(img.shape)
        img = compress(img)
        allImages.append(img.flatten())
        masks = []
        for m in os.listdir(immasks):
            #mask = imageio.imread(immasks+m)
            mask = imread(immasks+m)
            mask = compress(mask)
            masks.append(mask)
            if (first):
                print("mask:"+m)
                if (len(masks)==1):
                    print("Testing inside mask loop:")
                    #print(np.nonzero(imread(immasks+m)))
                    #animg = Image.fromarray(imread(immasks+m), 'L')
                    #animg.save(compressionLoc)
                    
            #maskindices = np.nonzero(mask)
            #masks.append(maskindices)
        
        masksInfo = maskDetails(masks, first)
        if (first):
            print(filename)
            print("mask info")
            print(masksInfo[0])
        trainingLabels = trainLabels(masksInfo)
        if (first):
            print("trainingLabels")
            print(trainingLabels[0])
        flattenedLabels = trainingLabels.flatten()#lambda l: [lab for rows in trainingLabels for columns in rows for lab in column]
        if (first):
            print("flattenedLabels")
            print(flattenedLabels[0])
            print(flattenedLabels[1])
            print(flattenedLabels[2])
            print(flattenedLabels[3])
            print(flattenedLabels[4])
            print("after flattened")
        allLabels.append(flattenedLabels)
        first=False
        #Drop the 4th dimension. https://www.kaggle.com/c/data-science-bowl-2018/discussion/47750
        #Maybe need to keep it later.
    normalizedImages = normalizeInput(allImages)
    print(normalizedImages[0])
    

def main(unused_argv):
    trainModel(unused_argv)
    
    
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)