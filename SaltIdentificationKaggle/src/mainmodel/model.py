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

def createModel(features, labels, mode):
    print()

def testModel():
    moddir = "SavedModels"
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

    moddir = "SavedModels"
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

    #nucleus_detector.train(
     #   input_fn=train_input_fn,
      #  steps=200000,
       # hooks=[logging_hook])

def main(unused_argv): 
    trainModel()
    
tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
tensorflow.app.run(main)