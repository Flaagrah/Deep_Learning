import tensorflow
import os
import numpy as np
import pandas
import imageio
import codecs, json
from skimage.io import imread
from skimage.transform import resize
from pathlib import Path

import mainmodel.Normalization as Normalization

dataURL = '../Data/stage1_train/'
imagesDir = 'images/'
masksDir = 'masks/'
compressionLoc = '../Data/tmp.png'
testURL = '../Data/stage1_test/'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

BOX_HEIGHT = 16
BOX_WIDTH = 16

CERTAINTY_THRESHOLD = 0.5

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
    
from tensorflow.python.ops import array_ops
def conv2d_3x3(filters):
    return tensorflow.layers.Conv2D(filters, kernel_size=(3,3), activation=tensorflow.nn.relu, padding='same')

def max_pool():
    return tensorflow.layers.MaxPooling2D((2,2), strides=2, padding='same') 

def conv2d_transpose_2x2(filters):
    return tensorflow.layers.Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')

def concatenate(branches):
    return array_ops.concat(branches, 3)

    
def createModel(features, labels, mode):
    #HEIGHT*WIDTH*4
    input_layer = tensorflow.reshape(features["x"], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    #Model taken from:
    #https://www.kaggle.com/piotrczapla/tensorflow-u-net-starter-lb-0-34
    c1 = conv2d_3x3(8) (input_layer)
    c1 = conv2d_3x3(8) (c1)
    p1 = max_pool() (c1)

    c2 = conv2d_3x3(16) (p1)
    c2 = conv2d_3x3(16) (c2)
    p2 = max_pool() (c2)

    c3 = conv2d_3x3(32) (p2)
    c3 = conv2d_3x3(32) (c3)
    p3 = max_pool() (c3)

    c4 = conv2d_3x3(64) (p3)
    c4 = conv2d_3x3(64) (c4)
    p4 = max_pool() (c4)

    c5 = conv2d_3x3(128) (p4)
    c5 = conv2d_3x3(128) (c5)

    u6 = conv2d_transpose_2x2(64) (c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_3x3(64) (u6)
    c6 = conv2d_3x3(64) (c6)

    u7 = conv2d_transpose_2x2(32) (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_3x3(32) (u7)
    c7 = conv2d_3x3(32) (c7)

    u8 = conv2d_transpose_2x2(16) (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_3x3(16) (u8)
    c8 = conv2d_3x3(16) (c8)

    u9 = conv2d_transpose_2x2(8) (c8)
    u9 = concatenate([u9, c1])
    c9 = conv2d_3x3(8) (u9)
    c9 = conv2d_3x3(8) (c9)

    c10 = max_pool() (c9)
    c11 = max_pool() (c10)
    c12 = max_pool() (c11)
    c13 = max_pool() (c12)
    c14 = max_pool() (c13)
    c15 = max_pool() (c14)
    c16 = max_pool() (c15)
    c17 = max_pool() (c16)
    
    dense = tensorflow.layers.dense(inputs=c17, units = 1280)
    
    dropout = tensorflow.layers.dropout(inputs=dense, rate=0.2, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
    
    preds = tensorflow.layers.dense(inputs=dropout, units = int( (IMAGE_HEIGHT/BOX_HEIGHT) * (IMAGE_WIDTH/BOX_WIDTH) * 5 ), activation=tensorflow.nn.sigmoid, kernel_initializer=tensorflow.contrib.layers.xavier_initializer() )
    print("h")
    print(preds.shape)
    print('J')
    predictions = {
        "preds": preds,
        #"boxes": convertOutput(logits)
        }
    
    if mode == tensorflow.estimator.ModeKeys.PREDICT :
        return tensorflow.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    print("label shape:")
    print(labels.shape)
    print("logit shape:")
    print(preds.shape)
    #loss = tensorflow.losses.mean_squared_error(labels=labels, predictions=preds)
    #How are the preds reshaped.
    reshapedPreds = tensorflow.reshape(preds, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
    reshapedLabels = tensorflow.reshape(labels, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
    #Cost calculation taken from https://stackoverflow.com/questions/48938120/make-tensorflow-ignore-values
    #This excludes bounding boxes that are 
    mask = tensorflow.tile(reshapedLabels[:, :, :, 0:1], [1, 1, 1, 5]) #repeating the first item 5 times
    mask_first = tensorflow.tile(reshapedLabels[:, :, :, 0:1], [1, 1, 1, 1]) #repeating the first item 1 time

    mask_of_ones = tensorflow.ones(tensorflow.shape(mask_first))

    full_mask = tensorflow.concat([tensorflow.to_float(mask_of_ones), tensorflow.to_float(mask[:, :, :, 1:])], 3)

    terms = tensorflow.multiply(full_mask, tensorflow.to_float(tensorflow.subtract(reshapedLabels, reshapedPreds, name="loss")))
    
    non_zeros = tensorflow.cast(tensorflow.count_nonzero(full_mask), dtype=tensorflow.float32)

    loss = tensorflow.div((tensorflow.reduce_sum(tensorflow.square(terms))), non_zeros, "loss_calc")
    #loss = tensorflow.reduce_sum(tensorflow.square(terms))
    
    if mode == tensorflow.estimator.ModeKeys.TRAIN:
        #optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=1.0)
        optimizer = tensorflow.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tensorflow.train.get_global_step())
        return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    eval_metric_ops = {
        "accuracy": tensorflow.metrics.accuracy(
        labels=labels, predictions=predictions["logits"])}
    return tensorflow.estimator.EstimatorSpec(
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
    
    moddir = "/tmp/DataScienceBowl"
    if Path("/output").exists() :
        moddir = "/output/"
    nucleus_detector = tensorflow.estimator.Estimator(
        model_fn = createModel, model_dir=moddir)
    
    normalizedImages = []
    if (train):
        if (Path('imagesTotal.npy')).exists():
            normalizedImages = np.reshape(np.asarray(np.load('imagesTotal.npy')).astype(np.float32), (-1, 256, 256, 3))
            print(normalizedImages.shape)
            allLabels = np.reshape(np.asarray(np.load('labelsTotal.npy')).astype(np.float32), (-1, 16,16, 5))
            print(allLabels.shape)
            alldims = np.reshape(np.asarray(np.load('dimsTotal.npy')).astype(np.int32), (-1, 3))
            print(alldims.shape)
        else : 
            for filename in os.listdir(dataURL):
                print(filename)
                imdir = dataURL+filename+'/'+imagesDir
                immasks = dataURL+filename+'/'+masksDir
                #imagefile = imageio.imread(imdir+os.listdir(imdir)[0])
                img = imread(imdir+os.listdir(imdir)[0])
                alldims.append((img.shape[0], img.shape[1], 3))
                if (img.shape[2] == 4):
                    img = img[:, :, 0:3]
                img = compress(img)
                allImages.append(img)
                masks = []
                for m in os.listdir(immasks):
                    #mask = imageio.imread(immasks+m)
                    mask = imread(immasks+m)
                    mask = compress(mask)
                    masks.append(mask)
                
                masksInfo = maskDetails(masks, first)
                trainingLabels = trainLabels(masksInfo)
                flattenedLabels = trainingLabels.flatten()
                #lambda l: [lab for rows in trainingLabels for columns in rows for lab in column]
                processed = []
                for n in range(0, len(flattenedLabels)):
                    processed.append(flattenedLabels[n])
                print(processed)
                allLabels.append(processed)
                first=False
                #Drop the 4th dimension. https://www.kaggle.com/c/data-science-bowl-2018/discussion/47750
                #Maybe need to keep it later.
            
            l = Normalization.NormalizeWidthHeightForAll(allLabels)
            allLabels = l
            print("hello")
            normalizedImages = allImages
            print(normalizedImages[1][0])
            normalizedImages = np.asarray(normalizedImages).astype(np.float32)
            alldims = np.asarray(alldims).astype(np.int32)
            allLabels = np.asarray(allLabels).astype(np.float32)
            
            np.save('images', normalizedImages)
            np.save('dims', alldims)
            np.save('labels', allLabels)
            
        tensors_to_log = {}
        logging_hook = tensorflow.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)
        train_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
            x={"x": normalizedImages},
            y=allLabels,
            batch_size=40,
            num_epochs=None,
            shuffle=False)

        nucleus_detector.train(
            input_fn=train_input_fn,
            steps=200000,
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
        test_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
           x={"x": np.asarray(allTestImages).astype(np.float32)},
            shuffle=False)
        preds = nucleus_detector.predict(test_input_fn, checkpoint_path=None)
        predList = []
        for single_prediction in preds:
            predList.append(list(single_prediction['preds']))
        predList = np.asarray(predList)
        print(predList)
        reshapePreds = np.reshape(predList, (-1, boxRows, boxColumns, 5))
        unNormal = Normalization.unNormalizeAll(reshapePreds)
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


def main(unused_argv):
    images = (np.load('images.npy'))
    labels = np.load('labels.npy')
    dims = np.load('dims.npy')
    
    images = np.reshape(images, (-1, IMAGE_HEIGHT*IMAGE_WIDTH*3))
    
    #json.dump(images.tolist(), codecs.open('images.json', 'w', encoding='utf-8'), separators=(',', ':'), indent=4)
    #json.dump(labels.tolist(), codecs.open('labels.json', 'w', encoding='utf-8'), separators=(',', ':'), indent=4)
    #json.dump(dims.tolist(), codecs.open('dims.json', 'w', encoding='utf-8'), separators=(',', ':'), indent=4)
    
    #images = np.reshape(images, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

   # df.to_csv(path_or_buf = 'data.csv', header=True, index=True)
    trainModel(True, False)

    imageio.imwrite('img.png', images[1])
    filenames = os.listdir(dataURL)
    rUnNorm = np.reshape(labels, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
    print("UnNormalize")
    print(rUnNorm[0][0])
    rUnNorm = Normalization.unNormalizeAll(rUnNorm)
    rUnNorm = np.reshape(rUnNorm, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
    print(rUnNorm[0][0])
    print("Normalize")
    normed = Normalization.NormalizeWidthHeightForAll(rUnNorm)
    normed = np.reshape(normed, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
    print(normed[0][0])
    #rUnNorm = np.reshape(rUnNorm, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))

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
    
    
    
    
    

print("hello")
tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
print("hello")
tensorflow.app.run(main)