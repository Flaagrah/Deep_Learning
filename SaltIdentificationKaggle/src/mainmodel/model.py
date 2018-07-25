import tensorflow
import os
import numpy as np
import pandas
import imageio
import codecs, json
from skimage.io import imread
from pathlib import Path

import mainmodel.preProcessing as preProcessing
import mainmodel.postProcessing as postProcessing

from mainmodel import IMAGE_HEIGHT as IMAGE_HEIGHT
from mainmodel import IMAGE_WIDTH as IMAGE_WIDTH

from tensorflow.python.ops import array_ops
def conv2d_3x3(filters):
    return tensorflow.layers.Conv2D(filters, kernel_size=(3,3), activation=tensorflow.nn.relu, padding='same')

def max_pool():
    return tensorflow.layers.MaxPooling2D((2,2), strides=2, padding='same') 

def conv2d_transpose_3x3(filters):
    return tensorflow.layers.Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='same')

def concatenate(branches):
    return array_ops.concat(branches, 3)

def createModel(features, labels, mode):
    input_layer = tensorflow.reshape(features["x"], [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    #Model taken from:
    #https://www.kaggle.com/piotrczapla/tensorflow-u-net-starter-lb-0-34
    c1 = conv2d_3x3(8) (input_layer)
    c1 = conv2d_3x3(8) (c1)
    p1 = max_pool() (c1)
    print(p1.shape)

    c2 = conv2d_3x3(16) (p1)
    c2 = conv2d_3x3(16) (c2)
    p2 = max_pool() (c2)
    print(p2.shape)

    c3 = conv2d_3x3(32) (p2)
    c3 = conv2d_3x3(32) (c3)
    p3 = max_pool() (c3)
    print(p3.shape)
    
    c4 = conv2d_3x3(64) (p3)
    c4 = conv2d_3x3(64) (c4)
    print(c4.shape)
    #p4 = max_pool() (c4)

    #c5 = conv2d_3x3(128) (p4)
    #c5 = conv2d_3x3(128) (c5)

    #u6 = conv2d_transpose_2x2(64) (c5)
    #u6 = concatenate([u6, c4])
    #c6 = conv2d_3x3(64) (u6)
    #c6 = conv2d_3x3(64) (c6)

    u7 = conv2d_transpose_3x3(32) (c4)
    print("u7")
    print(u7)
    print("c3")
    print(c3)
    u7 = concatenate([u7, c3])
    c7 = conv2d_3x3(32) (u7)
    c7 = conv2d_3x3(32) (c7)

    u8 = conv2d_transpose_3x3(16) (c7)
    print("u8")
    u8 = tensorflow.slice(u8, [0, 0, 0, 0], [10, 51, 51, 16])
    print(u8)
    print("c2")
    print(c2)
    u8 = concatenate([u8, c2]) 
    c8 = conv2d_3x3(16) (u8)
    c8 = conv2d_3x3(16) (c8)

    u9 = conv2d_transpose_3x3(8) (c8)
    print("u9")
    print(u9)
    u9 = tensorflow.slice(u9, [0, 0, 0, 0], [10, 101, 101, 8])
    print("c1")
    print(c1)
    u9 = concatenate([u9, c1])
    c9 = conv2d_3x3(8) (u9)
    c9 = conv2d_3x3(8) (c9)

    c15 = tensorflow.layers.Conv2D(1, (1, 1)) (c9)
    c15 = tensorflow.layers.Flatten()(c15)
    dense = tensorflow.layers.Dense(units = 1280)(c15)
    dropout = tensorflow.layers.Dropout(rate=0.2)(dense)
    
    preds = tensorflow.layers.Dense(units = IMAGE_HEIGHT*IMAGE_WIDTH, activation=tensorflow.nn.sigmoid, kernel_initializer=tensorflow.contrib.layers.xavier_initializer() )(dropout)
    predictions = {
        "preds": preds,
        }
    
    if mode == tensorflow.estimator.ModeKeys.PREDICT :
        return tensorflow.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    #loss = tensorflow.losses.mean_squared_error(labels=labels, predictions=preds)
    #How are the preds reshaped.
    reshapedPreds = tensorflow.reshape(preds, (-1, IMAGE_HEIGHT, IMAGE_WIDTH))
    reshapedLabels = tensorflow.reshape(labels, (-1, IMAGE_HEIGHT, IMAGE_WIDTH))
    
    loss = tensorflow.losses.mean_squared_error(reshapedLabels, reshapedPreds)
    
    if mode == tensorflow.estimator.ModeKeys.TRAIN:
        optimizer = tensorflow.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tensorflow.train.get_global_step())
        return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
    eval_metric_ops = {
        "accuracy": tensorflow.metrics.accuracy(
        labels=labels, predictions=predictions["logits"])}
    return tensorflow.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



moddir = "../../SavedModels"

def testModel():
    nucleus_detector = tensorflow.estimator.Estimator(
        model_fn = createModel, model_dir=moddir)
    
    testImages = preProcessing.createTestInput()
    
    test_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
       x={"x": np.asarray(testImages).astype(np.float32)},
        shuffle=False)
    preds = nucleus_detector.predict(test_input_fn, checkpoint_path=None)
    predList = []
    for single_prediction in preds:
        predList.append(list(single_prediction['preds']))
    predList = np.asarray(predList)
    print(predList)
    names, encoding = postProcessing.generateOutput(testImages)

    df = pandas.read_csv('stage2_sample_submission_final.csv')

    for i in range(0, len(names)):
       df.loc[i] = [names[i], encoding[i]]
        
    df.to_csv(path_or_buf = 'submission.csv',
                             header=True,
                             index=False)
    

def trainModel():
    nucleus_detector = tensorflow.estimator.Estimator(
        model_fn = createModel, model_dir=moddir)
        
    saltImages, saltMasks = preProcessing.createTrainingInput()
    tensors_to_log = {}
    logging_hook = tensorflow.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    train_input_fn = tensorflow.estimator.inputs.numpy_input_fn(
        x={"x": saltImages.copy()},
        y=saltMasks.copy(),
        batch_size=10,
        num_epochs=None,
        shuffle=True)

    nucleus_detector.train(
       input_fn=train_input_fn,
       steps=200000,
       hooks=[logging_hook])

def main(unused_argv): 
    trainModel()
    
tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
tensorflow.app.run(main)