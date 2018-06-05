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
import mainmodel.DataAugmentation as DataAugmentation

dataURL = '../Data/stage1_train/'
imagesDir = 'images/'
masksDir = 'masks/'
compressionLoc = '../Data/tmp.png'
testURL = '../Data/stage2_test/stage2_test_final/'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

BOX_HEIGHT = 16
BOX_WIDTH = 16

CERTAINTY_THRESHOLD = 0.95

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
    input_layer = tensorflow.layers.AveragePooling2D((4, 4), strides=(4, 4))(input_layer)
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

    c15 = tensorflow.layers.Conv2D(1, (1, 1)) (c9)
    c15 = tensorflow.layers.Flatten()(c15)
    dense = tensorflow.layers.Dense(units = 1280)(c15)
    print(dense.shape)
    #, training=mode == tensorflow.estimator.ModeKeys.TRAIN
    dropout = tensorflow.layers.Dropout(rate=0.2)(dense)
    
    preds = tensorflow.layers.Dense(units = int( (IMAGE_HEIGHT/BOX_HEIGHT) * (IMAGE_WIDTH/BOX_WIDTH) * 5 ), activation=tensorflow.nn.sigmoid, kernel_initializer=tensorflow.contrib.layers.xavier_initializer() )(dropout)
    print("h")
    print(preds.shape)
    print('J')
    predictions = {
        "preds": preds,
        #"boxes": convertOutput(logits)
        }
    
    if mode == tensorflow.estimator.ModeKeys.PREDICT :
        return tensorflow.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    #loss = tensorflow.losses.mean_squared_error(labels=labels, predictions=preds)
    #How are the preds reshaped.
    reshapedPreds = tensorflow.reshape(preds, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
    reshapedLabels = tensorflow.reshape(labels, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
    print("label shape:")
    print(reshapedLabels.shape)
    print("logit shape:")
    print(reshapedPreds.shape)
    #Cost calculation taken from https://stackoverflow.com/questions/48938120/make-tensorflow-ignore-values
    #This excludes bounding boxes that are 
    mask = tensorflow.tile(reshapedLabels[:, :, :, 0:1], [1, 1, 1, 5]) #repeating the first item 5 times
    mask_first = tensorflow.tile(reshapedLabels[:, :, :, 0:1], [1, 1, 1, 1]) #repeating the first item 1 time

    mask_of_ones = tensorflow.ones(tensorflow.shape(mask_first))

    full_mask = tensorflow.concat([tensorflow.to_float(mask_of_ones), tensorflow.to_float(mask[:, :, :, 1:])], 3)

    terms = tensorflow.multiply(full_mask, tensorflow.to_float(tensorflow.subtract(reshapedLabels, reshapedPreds, name="loss")))
    
    #non_zeros = tensorflow.cast(tensorflow.count_nonzero(full_mask), dtype=tensorflow.float32)

    #loss = tensorflow.div((tensorflow.reduce_sum(tensorflow.square(terms))), non_zeros, "loss_calc")
    loss = tensorflow.reduce_sum(tensorflow.square(terms))
    
    if mode == tensorflow.estimator.ModeKeys.TRAIN:
        #optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=1.0)
        optimizer = tensorflow.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tensorflow.train.get_global_step())
        return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    eval_metric_ops = {
        "accuracy": tensorflow.metrics.accuracy(
        labels=labels, predictions=predictions["logits"])}
    return tensorflow.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def hitOrMiss(flag):
    if flag > CERTAINTY_THRESHOLD:
        return True
    return False

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
        if  (mode == 'MINIMUM_THRESHOLD' and hitOrMiss(boxFlag)) or (mode == 'LARGEST' and (boxIndex + 6 > len(predictionsForOneImage))) :
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
            boxVals.append(boxFlag)
            boxVals.append(verCoord)
            boxVals.append(horCoord)
            boxVals.append(height)
            boxVals.append(width)
            boxes.append(boxVals)
    
    def coords(box):
        x1 = box[2] - (box[4]/2)
        x2 = box[2] + (box[4]/2)
        y1 = box[1] - (box[3]/2)
        y2 = box[1] + (box[3]/2)
        return x1, x2, y1, y2
    
    def getOverLap(start1, end1, start2, end2):
        startOverLap = -1
        endOverLap = -1
        if start1 >= start2 and start1 <= end2:
            startOverLap = start1
        elif start2 >= start1 and start2 <= end1:
            startOverLap = start2
        if startOverLap == -1:
            return 0
        if end1 <= end2:
            endOverLap = end1
        else:
            endOverLap = end2
        return endOverLap - startOverLap
    
    boxResults = []
    removeIndices = []
    checkedIndices = []
    for i in range(0, len(boxes)):
        removeIndices.append(False)
        checkedIndices.append(False)
    while True:
        numBoxes = len(boxes)
        checked = True
        maxBox = None
        maxBoxIndex = 0
        for i in range(0, numBoxes):
            box1 = boxes[i]
            if (maxBox is None or box1[0] > maxBox[0]) and removeIndices[i] == False and checkedIndices[i] == False:
                maxBox = box1
                print("DFHIODHFIOHDI")
                maxBoxIndex = i
                checked = False
                
        if checked == True:
            break
        checkedIndices[maxBoxIndex] = True
        x11, x12, y11, y12 = coords(maxBox)
        print(maxBox[0])
        print(x11)
        print(x12)
        print(y11)
        print(y12)
        
        i = 0
        for i in range(0, numBoxes):
            if i == maxBoxIndex or removeIndices[i] == True or checkedIndices[i] == True:
                continue
            print("here")
            box1 = boxes[i]
            print(box1[0])
            print(maxBox[0])
            if box1[0] < maxBox[0]:
                print("there")
                x21, x22, y21, y22 = coords(box1)
                print(maxBox[0])
                print(x21)
                print(x22)
                print(y21)
                print(y22)
                xLap = getOverLap(x11, x12, x21, x22)
                yLap = getOverLap(y11, y12, y21, y22)
                lap = xLap*yLap
                IOU = lap / (maxBox[3]*maxBox[4] + box1[3]*box1[4] - lap)
                print("lap")
                print(maxBox[3]*maxBox[4])
                print(box1[3]*box1[4])
                print(xLap)
                print(yLap)
                print(lap)
                print("IOU")
                print(IOU)
                if IOU > 0.1:
                    removeIndices[i] = True
                    
    for i in range(0, len(boxes)):
        if not removeIndices[i]:
            box = boxes[i]
            box = box[1:]
            boxResults.append(box)
            
    return boxResults

def generateOutput(imgNames, imgPreds, testDims):
    names = []
    encoding = []
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
        added = False
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
                topPoint = int(w * verDim) + top + 1
                if topPoint < 1:
                    topPoint = 1
                bottomPoint = topPoint + height - 1
                lastPixel = horDim*verDim
                if bottomPoint > lastPixel:
                    bottomPoint = lastPixel
                
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
            elif len(runLength) == 1 or len(runLength)==0:
                runLength = ''
            if not runLength == '':
                names.append(name)
                encoding.append(runLength)
                added = True
        if not added:
            names.append(name)
            encoding.append('1 1')
    return names, encoding

#Given top and bottom of segment, returns all segmentations.
def addSegment(segmentPair, traversedPixels, newAdditions):
    topPoint = segmentPair[0]
    bottomPoint = segmentPair[1]
    if topPoint > bottomPoint:
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
            addSegment(aSeg, traversedPixels, newAdditions)
            return
        #If the bottom point is among pixels already masked, move it before the already masked segment.
        if (bottomPoint >= tTop and bottomPoint <= tBottom):
            bottomPoint = tTop - 1
            aSeg = [topPoint, bottomPoint]
            addSegment(aSeg, traversedPixels, newAdditions)
            return
        if (topPoint <= tTop and bottomPoint >= tBottom):
            aBottom = tTop - 1
            aTop = tBottom + 1
            seg1 = [topPoint, aBottom]
            seg2 = [aTop, bottomPoint]
            addSegment(seg1, traversedPixels, newAdditions)
            addSegment(seg2, traversedPixels, newAdditions)
            return
    newAdditions.append([topPoint, bottomPoint]) 
    traversedPixels.append([topPoint, bottomPoint])       

def createInput(isTrainingInput):
    allLabels = []
    allImages = []
    allDims = []
    allImageNames = []
    first = True
    url = dataURL
    if not isTrainingInput:
        url = testURL
    for filename in os.listdir(url):
        imdir = url+filename+'/'+imagesDir
        immasks = url+filename+'/'+masksDir
        #imagefile = imageio.imread(imdir+os.listdir(imdir)[0])
        img = imread(imdir+os.listdir(imdir)[0])
        allDims.append((img.shape[0], img.shape[1], 3))
        if (len(img.shape) == 3 and img.shape[2] == 4):
            img = img[:, :, 0:3]
        elif len(img.shape) == 2:
            print("here")
            tmp = np.reshape(img, (img.shape[0], img.shape[1], 1))
            img = np.concatenate((tmp, tmp), axis=2)
            img = np.concatenate((img, tmp), axis=2)
            print(img.shape)
           
        img = compress(img)
        allImages.append(img)
        allImageNames.append(filename)
        masks = []
        if isTrainingInput:
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
            allLabels.append(processed)
            first=False
        #Drop the 4th dimension. https://www.kaggle.com/c/data-science-bowl-2018/discussion/47750
        #Maybe need to keep it later.
    
    if not allLabels == [] :
        l = Normalization.NormalizeWidthHeightForAll(allLabels)
        allLabels = l
        print("hello")
    
    normalizedImages = np.asarray(allImages).astype(np.float32)
    allDims = np.asarray(allDims).astype(np.int32)
    allLabels = np.asarray(allLabels).astype(np.float32)
    
    if (isTrainingInput):
        imgFileName = 'imagesTrain'
        labFileName = 'labelsTrain'
        dimFileName = 'dimsTrain' 
        np.save(imgFileName, normalizedImages)
        np.save(dimFileName, allDims)
        np.save(labFileName, allLabels)
        totalInput, totalLabels, totalDims = DataAugmentation.returnAugmentationForList(normalizedImages, allLabels, allDims)
        #imgFileName.append('Total')
        #dimFileName.append('Total')
        #labFileName.append('Total')
        np.save(imgFileName+'Total', totalInput)
        np.save(dimFileName+'Total', totalDims)
        np.save(labFileName+'Total', totalLabels)
    else:
        imgFileName = 'imagesTest'
        nameFileName = 'imagesNamesTest'
        dimFileName = 'dimsTest'
        np.save(imgFileName, normalizedImages)
        np.save(dimFileName, allDims)
        np.save(nameFileName, allImageNames)
        

def trainModel(train = False, test = False):
    allImages = []
    allLabels = []
    allDims = []
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
        if not Path('imagesTrainTotal.npy').exists():
            createInput(True)
        normalizedImages = np.reshape(np.asarray(np.load('imagesTrainTotal.npy')).astype(np.float32), (-1, 256, 256, 3))
        print(normalizedImages.shape)
        allLabels = np.reshape(np.asarray(np.load('labelsTrainTotal.npy')).astype(np.float32), (-1, 16,16, 5))
        print(allLabels.shape)
        allDims = np.reshape(np.asarray(np.load('dimsTrainTotal.npy')).astype(np.int32), (-1, 3))
        print(allDims.shape)
            
            
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
        if not Path('imagesTest.npy').exists():
            createInput(False)
        
        allTestImages = np.reshape(np.asarray(np.load('imagesTest.npy')).astype(np.float32), (-1, 256, 256, 3))
        print(allTestImages.shape)
        allTestImageNames = np.reshape(np.asarray(np.load('imagesNamesTest.npy')), (-1))
        print(allTestImageNames.shape)
        print(allTestImageNames[0])
        allTestDims = np.reshape(np.asarray(np.load('dimsTest.npy')).astype(np.int32), (-1, 3))
        print(allTestDims.shape)
        #for filename in os.listdir(testURL):
         #   print(filename)
          #  imdir = testURL+filename+'/'+imagesDir
           # img = imread(imdir+os.listdir(imdir)[0])
            #allTestDims.append(img.shape)
            #if (img.shape[2] == 4):
            #    img = img[:, :, 0:3]
            #allTestImageNames.append(os.listdir(imdir)[0])
            #img = compress(img)
            #allTestImages.append(img)
        
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
        names, encoding = generateOutput(allTestImageNames, unNormal, allTestDims)
       # for i in range(0, len(names)):
        #    fname = names[i]
         #   extIndex = len(fname) - 4
          #  names[i] = (fname[0:extIndex])
        print(names[0])
        print(encoding[0])
        df = pandas.read_csv('stage2_sample_submission_final.csv')
        #sub = pandas.DataFrame()

        #sub['ImageId'] = names
        #sub['EncodedPixels'] = encoding
        
        #for index, row in df.iterrows():
           #imgID = row['ImageId']
        for i in range(0, len(names)):
           df.loc[i] = [names[i], encoding[i]]
            
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
    #trainModel(True, False)
    #trainModel(False, True)

    #imageio.imwrite('img.png', images[1])
    filenames = os.listdir(dataURL)
    rUnNorm = np.reshape(labels, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
    rUnNorm = Normalization.unNormalizeAll(rUnNorm)
    rUnNorm = np.reshape(rUnNorm, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
    normed = Normalization.NormalizeWidthHeightForAll(rUnNorm)
    normed = np.reshape(normed, (-1, int(IMAGE_HEIGHT/BOX_HEIGHT), int(IMAGE_WIDTH/BOX_WIDTH), 5))
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
           0.99, 0.75, 0.0, 0.1, 0.2,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0,
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
    #print(generateOutput(['ImageName', 'ImageName2', 'ImageName3'], imgs, testDims))
    
    
    
    
    

print("hello")
tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
print("hello")
tensorflow.app.run(main)

            
                