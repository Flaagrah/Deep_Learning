import tensorflow as tf
import os
import numpy as np
import pandas
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from bokeh.layouts import column
from absl.logging import info
from astropy.wcs.docstrings import row
from _ast import Num
from pathlib import Path
from tensorflow.contrib.specs.python.specs_ops import Flat

from mainmodel import Normalization
from sympy.physics.quantum.gate import normalized

dataURL = '../Data/stage1_train/'
imagesDir = 'images/'
masksDir = 'masks/'
compressionLoc = '../Data/tmp.png'
testURL = '../Data/stage1_test/'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

BOX_HEIGHT = 16
BOX_WIDTH = 16

CERTAINTY_THRESHOLD = 0.6

def compress(img):
    img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return img

#perhaps normalize by mean and variance using tensorflow batch normalization?
def reduceInput(inputlist):
    normalizedList = []
    for img in inputlist:
        normalized = img/(255.0)
        normalizedList.append(normalized)
    return np.asarray(normalizedList, np.float32)

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
            if found == False :
                boxes[h][w][0]=0.0
                boxes[h][w][1]=0.0
                boxes[h][w][2]=0.0
                boxes[h][w][3]=0.0
                boxes[h][w][4]=0.0
    return boxes

def combineMasks(rawmasks):
    mergedMasks = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    for i in range(0, len(rawmasks)):
        mergedMasks = mergedMasks + rawmasks[i]
    return mergedMasks

def convertOutput(output):
    outputArray= []
    
    def addToArray(element):
        outputArray.append(element)
    print(("here"))
    tf.map_fn(addToArray, output)
    print("there")
    length = len(outputArray)
    boxes = []
    print(outputArray[0])
    for i in range(0, length):
        if ((i % 5) == 0 and outputArray[i] > 0.5):
            row = outputArray[i+1]
            column = outputArray[i+2]
            height = outputArray[i+3]
            width = outputArray[i+4]
            box = []
            box.append(row)
            box.append(column)
            box.append(height)
            box.append(width)
            boxes.append(box)
    print(boxes[0])
    return tf.convert_to_tensor(boxes)
    
def createModel(features, labels, mode):
    #HEIGHT*WIDTH*4
    input_layer = tf.reshape(features["x"], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    
    #HEIGHT*WIDTH*32
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size=[2,2],
        padding="same",
        activation=tf.nn.relu)
    
    #avg_pool1 =tf.layers.average_pooling2d(inputs = conv1, pool_size=[4,4], strides=[4,4]) 
    max_pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[2,2], strides=2)
    
    #pool1 = tf.concat([avg_pool1, max_pool1], -1)
    #HEIGHT/4*WIDTH/4*128
    conv2 = tf.layers.conv2d(
        inputs = max_pool1,
        filters = 64,
        kernel_size=[2,2],
        padding="same",
        activation=tf.nn.relu)
    
    #avg_pool2 =tf.layers.average_pooling2d(inputs = conv2, pool_size=[4,4], strides=[4,4])
    #HEIGHT/16*WIDTH/16*128
    max_pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
    
    
    conv3 = tf.layers.conv2d(
        inputs = max_pool2,
        filters = 128,
        kernel_size=[2,2],
        padding="same",
        activation=tf.nn.relu)

    max_pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)

    conv4 = tf.layers.conv2d(
        inputs = max_pool3,
        filters = 256,
        kernel_size=[2,2],
        padding="same",
        activation=tf.nn.relu)
    
    max_pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=2)
    
    conv5 = tf.layers.conv2d(
        inputs = max_pool4,
        filters = 512,
        kernel_size=[2,2],
        padding="same",
        activation=tf.nn.relu)
    
    max_pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2,2], strides=2)
    
    conv6 = tf.layers.conv2d(
        inputs = max_pool5,
        filters = 1024,
        kernel_size=[2,2],
        padding="same",
        activation=tf.nn.relu)
    
    max_pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2,2], strides=2)
    
    conv7 = tf.layers.conv2d(
        inputs = max_pool6,
        filters = 2048,
        kernel_size=[2,2],
        padding="same",
        activation=tf.nn.relu)
    
    max_pool7 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2,2], strides=2)
    
    conv8 = tf.layers.conv2d(
        inputs = max_pool7,
        filters = 4096,
        kernel_size=[2,2],
        padding="same",
        activation=tf.nn.relu)
    
    max_pool8 = tf.layers.max_pooling2d(inputs=conv8, pool_size=[2,2], strides=2)
    
    pool_flat = tf.reshape(max_pool8, [-1, 4096 ])
    
    dense = tf.layers.dense(inputs=pool_flat, units=4096, activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    preds = tf.layers.dense(inputs=dropout, units = int( (IMAGE_HEIGHT/BOX_HEIGHT) * (IMAGE_WIDTH/BOX_WIDTH) * 5 ), activation=tf.nn.sigmoid )
    
    predictions = {
        "preds": preds,
        #"boxes": convertOutput(logits)
        }
    
    if mode == tf.estimator.ModeKeys.PREDICT :
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    print("label shape:")
    print(labels.shape)
    print("logit shape:")
    print(preds.shape)
    #loss = tf.losses.mean_squared_error(labels=labels, predictions=preds)
    #How are the preds reshaped.
    reshapedPreds = tf.reshape(preds, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
    reshapedLabels = tf.reshape(labels, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
    #Cost calculation taken from https://stackoverflow.com/questions/48938120/make-tensorflow-ignore-values
    #This excludes bounding boxes that are 
    mask = tf.tile(reshapedLabels[:, :, :, 0:1], [1, 1, 1, 5]) #repeating the first item 5 times
    mask_first = tf.tile(reshapedLabels[:, :, :, 0:1], [1, 1, 1, 1]) #repeating the first item 1 time

    mask_of_ones = tf.ones(tf.shape(mask_first))

    full_mask = tf.concat([tf.to_float(mask_of_ones), tf.to_float(mask[:, :, :, 1:])], 3)

    loss = tf.reduce_mean(tf.square(tf.multiply(full_mask, tf.to_float(tf.subtract(reshapedLabels, reshapedPreds, name="loss")))))
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["logits"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def processResults(predictionsForOneImage, mode = 'MINIMUM_THRESHOLD'):
    #number of rows
    refBoxRows = IMAGE_HEIGHT/BOX_HEIGHT
    #number of columns
    refBoxColumns = IMAGE_WIDTH/BOX_WIDTH
    boxes = []
    
    largest = -1.0
    largestIndex = 0       
    
    for i in range(0, int(len(predictionsForOneImage)/5)):
        boxIndex = i*5
        boxFlag = predictionsForOneImage[boxIndex]
        if boxFlag > largest:
            largest = boxFlag
            largestIndex = boxIndex
        if  (mode == 'MINIMUM_THRESHOLD' and boxFlag > CERTAINTY_THRESHOLD) or (mode == 'LARGEST' and (boxIndex + 6 > len(predictionsForOneImage))) :
            boxVerLoc = predictionsForOneImage[boxIndex+1]
            boxHorLoc = predictionsForOneImage[boxIndex+2]
            boxHeight = predictionsForOneImage[boxIndex+3]
            boxWidth = predictionsForOneImage[boxIndex+4]
            
            row = int(i/refBoxColumns)
            column = int(i % refBoxColumns)
            
            #Get the actual coordinates and height/width in 256*256 version of image.
            verCoord = ((row +boxVerLoc) * BOX_HEIGHT)
            horCoord = ((column+boxHorLoc) * BOX_WIDTH)
            height = boxHeight * IMAGE_HEIGHT
            width = boxWidth * IMAGE_WIDTH
            
            #Making sure that the boundaries are within the image.
            if verCoord < 0.0:
                verCoord = 0.0
            if horCoord < 0.0:
                horCoord = 0.0
            if (verCoord - (height / 2.0)) < 0.0:
                height = verCoord * 2.0
            if (horCoord - (width / 2.0)) < 0.0:
                width = horCoord * 2.0
            if (verCoord + (height / 2.0)) > 255.0:
                height = (255.0 - verCoord) * 2.0
            if (horCoord + (width / 2.0)) > 255.0:
                width = (255.0 - horCoord) * 2.0
            
            boxVals = []
            boxVals.append(verCoord)
            boxVals.append(horCoord)
            boxVals.append(height)
            boxVals.append(width)
            boxes.append(boxVals)
    
    return boxes

def generateOutput(imgNames, imgPreds, testDims):
    imgStrs = []
    for i in range(0, len(imgPreds)):
        img = imgPreds[i]
        dims = testDims[i]
        name = imgNames[i]
        
        boxResults = processResults(img)
        if len(boxResults) == 0:
            boxResults = processResults(img, 'LARGEST')
        print(name)
        print(len(boxResults))
        verDim = dims[0]
        horDim = dims[1]
        
        horMultiple = horDim/IMAGE_WIDTH
        verMultiple = verDim/IMAGE_HEIGHT
        
        traversedPixels = []
        if (len(boxResults) == 0):
            imgStrs.append([name, ''])
        else :
            #Process each mask using the boxes.
            for j in range(0, len(boxResults)):
                runLength = ''
                box = boxResults[j]
                verCoord = int(box[0] * verMultiple)
                horCoord = int(box[1] * horMultiple)
                height = int(box[2] * verMultiple)
                width = int(box[3] * horMultiple)
                
                leftSide = horCoord - int(width/2)
                rightSide = horCoord + int(width/2)
                top = verCoord - int(height/2)
                bottom = verCoord + int(height/2)
                
                newAdditions = []
                #Encode the pixels for each mask
                for w in range(leftSide, rightSide):
                    topPoint = int(w * verDim) + top
                    bottomPoint = topPoint + height -1
                    pair = [topPoint, bottomPoint]
                    addSegment(pair, traversedPixels, newAdditions)
                for a in range(0, len(newAdditions)):
                    newAdd = newAdditions[a]
                    newTop = newAdd[0]
                    newBottom = newAdd[1]
                    height = newBottom - newTop + 1
                    runLength += ' ' + str(newTop) + ' ' + str(height)
                                            
                if len(runLength) > 1:
                    runLength = runLength[1:]
                elif len(runLength) == 1:
                    runLength = ''
                imgStrs.append([name, runLength])
        

    return imgStrs

#Given top and bottom of segment, returns all segmentations.
def addSegment(segmentPair, traversedPixels, newAdditions, recurse = 0):
    topPoint = segmentPair[0]
    bottomPoint = segmentPair[1]
    if topPoint >= bottomPoint:
        return
    for t in range(0, len(traversedPixels)):
        pixelPair = traversedPixels[t]
        tTop = pixelPair[0]
        tBottom = pixelPair[1]
        #Changing the top and bottom point to avoid duplicating pixels.
        #If top point is among pixels already masked, move it beyond the already masked segment.
        if (topPoint >= tTop and topPoint <= tBottom):
            topPoint = tBottom + 1
            aSeg = [topPoint, bottomPoint]
            addSegment(aSeg, traversedPixels, newAdditions, recurse+1)
            return
        #If the bottom point is among pixels already masked, move it before the already masked segment.
        if (bottomPoint >= tTop and bottomPoint <= tBottom):
            bottomPoint = tTop - 1
            aSeg = [topPoint, bottomPoint]
            addSegment(aSeg, traversedPixels, newAdditions, recurse+1)
            return
        if (topPoint <= tTop and bottomPoint >= tBottom):
            aBottom = tTop - 1
            aTop = tBottom + 1
            seg1 = [aTop, bottomPoint]
            seg2 = [topPoint, aBottom]
            addSegment(seg1, traversedPixels, newAdditions, recurse+1)
            addSegment(seg2, traversedPixels, newAdditions, recurse+1)
            return
    newAdditions.append([topPoint, bottomPoint]) 
    traversedPixels.append([topPoint, bottomPoint])       

def trainModel(train = False, test = False):
    allImages = []
    allLabels = []
    alldims = []
    allTestDims = []
    allTestImages = []
    allTestImageNames = []
    first = True
    skip = False
    skip2 = False
    boxRows = int(IMAGE_HEIGHT/BOX_HEIGHT)
    boxColumns = int(IMAGE_WIDTH/BOX_WIDTH)
    mean = None
    sd = None
    nucleus_detector = tf.estimator.Estimator(
        model_fn = createModel, model_dir="/tmp/DataScienceBowl")
    
    normalizedImages = []
    if (train):
        if (Path('images.npy')).exists():
            print('hello')
            normalized = np.load('images.npy')
            allLabels = np.load('labels.npy')
            alldims = np.load('dims.npy')
            meanAndSD = np.load('meanAndSD.npy')
            print('bye')
            mean = meanAndSD[0]
            sd = meanAndSD[1]
            
        else : 
            for filename in os.listdir(dataURL):
                print(filename)
                imdir = dataURL+filename+'/'+imagesDir
                immasks = dataURL+filename+'/'+masksDir
                #imagefile = imageio.imread(imdir+os.listdir(imdir)[0])
                img = imread(imdir+os.listdir(imdir)[0])
                if (img.shape[2] == 4):
                    img = img[:, :, 0:3]
                img = compress(img)
                allImages.append(img)
                alldims.append(img.shape)
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
                flattenedLabels = trainingLabels.flatten()
                #lambda l: [lab for rows in trainingLabels for columns in rows for lab in column]
                if (first):
                    print("flattenedLabels")
                    print(flattenedLabels[0])
                    print(flattenedLabels[1])
                    print(flattenedLabels[2])
                    print(flattenedLabels[3])
                    print(flattenedLabels[4])
                    print("after flattened")
                print("Flatten")
                processed = []
                for n in range(0, len(flattenedLabels)):
                    processed.append(flattenedLabels[n])
                print(processed)
                allLabels.append(processed)
                first=False
                #Drop the 4th dimension. https://www.kaggle.com/c/data-science-bowl-2018/discussion/47750
                #Maybe need to keep it later.
            
            m, s, l = Normalization.NormalizeWidthHeightForAll(allLabels)
            mean = m
            sd = s
            allLabels = l
            normalizedImages = reduceInput(allImages)
            
            np.save('images', np.asarray(normalizedImages))
            np.save('dims', np.asarray(alldims))
            np.save('labels', np.asarray(allLabels))
            np.save('meanAndSD', np.asarray([mean, sd]))
            
        tensors_to_log = {}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)
        theLabels = np.asarray(allLabels).astype(np.float32)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": normalizedImages},
            y=theLabels,
            batch_size=1,
            num_epochs=10,
            shuffle=False)

        nucleus_detector.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook])
     
    if (test):   
        for filename in os.listdir(testURL):
            print(filename)
            imdir = testURL+filename+'/'+imagesDir
            img = imread(imdir+os.listdir(imdir)[0])
            allTestDims.append(img.shape)
            if (img.shape[2] == 4):
                img = img[:, :, 0:3]
            allTestImageNames.append(os.listdir(imdir)[0])
            img = compress(img)
            allTestImages.append(img)
        
        allTestImages = reduceInput(allTestImages)
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
           x={"x": np.asarray(allTestImages).astype(np.float32)},
            shuffle=False)
        preds = nucleus_detector.predict(test_input_fn, checkpoint_path=None)
        predList = []
        for single_prediction in preds:
            predList.append(list(single_prediction['preds']))
        predList = np.asarray(predList)
        print(predList)
        reshapePreds = np.reshape(predList, (-1, boxRows, boxColumns, 5))
        unNormal = Normalization.unNormalizeAll(reshapePreds, mean, sd)
        unNormal = np.reshape(unNormal, (-1, boxRows*boxColumns*5))
        imgStrs = generateOutput(allTestImageNames, unNormal, allTestDims)
        print("Hello")
        print(len(imgStrs))
        for i in range(0, len(imgStrs)):
            fname = imgStrs[i][0]
            extIndex = len(fname) - 4
            imgStrs[i][0] = (fname[0:extIndex])
            print(imgStrs[i][0])
        df = pandas.read_csv('submission.csv')
        for index, row in df.iterrows():
            imgID = row['ImageId']
            for i in range(0, len(imgStrs)):
                if imgID == imgStrs[i][0]:
                    row['EncodedPixels'] = imgStrs[i][1]
                    print("here")
                    print(row['EncodedPixels'])
                    break
            
        df.to_csv(path_or_buf = 'submission.csv',
                                 header=True,
                                 index=False)
   # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={"x": testData},
     #   y=testLabels,
      #  num_epochs=1,
       # shuffle=False)
    
    #eval_results = mnist_small_classifier.evaluate(input_fn=eval_input_fn)
    #print(eval_results)

def main(unused_argv):
    trainModel(True, True)
    #trainModel('TRAIN')
    #test()
    img = [1.0, 0.4, 0.3, 0.2, 0.1,
           1.0, 0.5, 0.5, 0.1, 0.1,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.1, 0.2, 0.15, 0.3,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.2, 0.2, 0.1, 0.2,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.5, 0.5, 0.1, 0.1,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.15, 0.2, 0.15, 0.3,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.4, 0.3, 0.1, 0.2,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0
           ]
    img2 = [1.0, 0.4, 0.3, 0.2, 0.1,
           1.0, 0.5, 0.5, 0.1, 0.1,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.1, 0.2, 0.15, 0.3,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.2, 0.2, 0.1, 0.2,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.5, 0.5, 0.1, 0.1,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.15, 0.2, 0.15, 0.3,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.4, 0.3, 0.1, 0.2,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0
           ]
    img3 = [0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.4, 0.4, 0.1, 0.2,
           1.0, 0.75, 0.0, 0.1, 0.2,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.0, 0.0, 0.1, 0.2,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.5, 0.5, 0.1, 0.1,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           1.0, 0.5, 0.5, 0.1, 0.1,
           ]
    imgs = []
    imgs.append(img)
    imgs.append(img2)
    imgs.append(img3)
    testDims = []
    testDims.append((1024, 512))
    testDims.append((1024, 512))
    testDims.append((512, 256))
    print(processResults(img3))
    print(generateOutput(['ImageName', 'ImageName2', 'ImageName3'], imgs, testDims))
    
    
    
    
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)