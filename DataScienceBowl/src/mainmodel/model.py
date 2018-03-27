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
    print(labels.shape)
    #HEIGHT*WIDTH*4
    input_layer = tf.reshape(features["x"], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 4])
    
    #HEIGHT*WIDTH*32
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size=[4,4],
        padding="same",
        activation=tf.nn.relu)
    
    #avg_pool1 =tf.layers.average_pooling2d(inputs = conv1, pool_size=[4,4], strides=[4,4]) 
    max_pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[4,4], strides=[4,4])
    
    #pool1 = tf.concat([avg_pool1, max_pool1], -1)
    #HEIGHT/4*WIDTH/4*128
    conv2 = tf.layers.conv2d(
        inputs = max_pool1,
        filters = 128,
        kernel_size=[4,4],
        padding="same",
        activation=tf.nn.relu)
    
    #avg_pool2 =tf.layers.average_pooling2d(inputs = conv2, pool_size=[4,4], strides=[4,4])
    #HEIGHT/16*WIDTH/16*128
    max_pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[4,4], strides=[4,4])
    
    #pool2 = tf.concat([avg_pool2, max_pool2], -1)
    
    pool2_flat = tf.reshape(max_pool2, [-1, int( int(IMAGE_HEIGHT/16) * int(IMAGE_WIDTH/16) * 128) ])
    
    dense = tf.layers.dense(inputs=pool2_flat, units=1280, activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    preds = tf.layers.dense(inputs=dropout, units = int( (IMAGE_HEIGHT/BOX_HEIGHT) * (IMAGE_WIDTH/BOX_WIDTH) * 5 ) )
    
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
    loss = tf.losses.mean_squared_error(labels=labels, predictions=preds)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["logits"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
def trainModel(unused_argv):
    allImages = []
    allLabels = []
    alldims = []
    allTestDims = []
    allTestImages = []
    first = True
    skip = False
    skip2 = False
    boxRows = int(IMAGE_HEIGHT/BOX_HEIGHT)
    boxColumns = int(IMAGE_WIDTH/BOX_WIDTH)
    for filename in os.listdir(testURL):
        print(filename)
        if skip2 == True :
            break
        elif skip==True:
            skip2=True
        else:
            skip=True
        imdir = testURL+filename+'/'+imagesDir
        img = imread(imdir+os.listdir(imdir)[0])
        allTestDims.append(img.shape)
        img = compress(img)
        allTestImages.append(img)
    
    skip = False
    skip2 = False
    for filename in os.listdir(dataURL):
        print(filename)
        if skip2 == True :
            break
        elif skip==True:
            skip2=True
        else:
            skip=True
        imdir = dataURL+filename+'/'+imagesDir
        immasks = dataURL+filename+'/'+masksDir
        #imagefile = imageio.imread(imdir+os.listdir(imdir)[0])
        img = imread(imdir+os.listdir(imdir)[0])
        alldims.append(img.shape)
        img = compress(img)
        allImages.append(img)
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
    normalizedImages = normalizeInput(allImages)
    #allTestImages = normalizeInput(allTestImages)
    
    print("normalized image 0:")
    print(normalizedImages[0])
    print("normalized image 1:")
    #print(normalizedImages[1])
    print("label 0:")
    print(allLabels[0])
    print("label 1:")
    #print(allLabels[1])
    
    nucleus_detector = tf.estimator.Estimator(
    model_fn = createModel, model_dir="/tmp/DataScienceBowl")
    
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    print("Hello")
    theLabels = np.asarray(allLabels).astype(np.float32)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": normalizedImages},
        y=theLabels,
        batch_size=1,
        num_epochs=None,
        shuffle=False)
    
    nucleus_detector.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
    
    #test_input_fn = tf.estimator.inputs.numpy_input_fn(
     #   x={"x": np.asarray(allTestImages).astype(np.float32)},
      #     batch_size=1,
       #    num_epochs=None,
        #   shuffle=False)
    
    #preds = nucleus_detector.predict(test_input_fn, hooks=logging_hook, checkpoint_path=None)
    #print(preds)
    
   # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={"x": testData},
     #   y=testLabels,
      #  num_epochs=1,
       # shuffle=False)
    
    #eval_results = mnist_small_classifier.evaluate(input_fn=eval_input_fn)
    #print(eval_results)

    

def main(unused_argv):
    trainModel(unused_argv)
    
    
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)