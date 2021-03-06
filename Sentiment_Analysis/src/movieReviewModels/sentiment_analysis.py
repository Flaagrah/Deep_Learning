'''
Created on Apr 5, 2018

@author: Bhargava
'''

import tensorflow as tf
import os
import numpy as np
import pandas
import tarfile

from movieReviewModels import preprocess_text
from movieReviewModels import max_length
from movieReviewModels import w2v
from movieReviewModels import embedding_size
from dask.bag import text
from gensim.models.word2vec import LineSentence


#embedding = "movieReviewData/GoogleNews-vectors-negative300.bin"
scaledata = "scaledata/"

dennisName = "Dennis+Schwartz"
jamesName = "James+Berardinelli"
scottName = "Scott+Renshaw"
steveName = "Steve+Rhodes"

subject = "/subj."
identity = "/id."
rating = "/rating."
labelThree = "/label.3class."
labelFour = "/label.4class."

dennisIndex = 0
jamesIndex = 1
scottIndex = 2
steveIndex = 3

lineIndex = 0
idIndex = 1
ratingIndex = 2
threeIndex = 3
fourIndex = 4

b_size = 1

embedding_array = []

def getWordFromFile(file):
    c = file.read()
    content = c.decode("utf-8")  
    contentList = content.split('\n')
    contentList = contentList[0:len(contentList)-1]
    return contentList

#Get all of the information for the reviews of the author given his name.
def authorInfo(name):
    file = tarfile.open("../movieReviewData/scale_data.tar.gz", "r:gz")
    subjFile = file.extractfile(scaledata+name+subject+name)
    idFile = file.extractfile(scaledata+name+identity+name)
    ratingFile = file.extractfile(scaledata+name+rating+name)
    labelThreeFile=  file.extractfile(scaledata+name+labelThree+name)
    labelFourFile = file.extractfile(scaledata+name+labelFour+name)
    
    subjLines = []
    while True:
        line = subjFile.readline().decode("utf-8")
        if len(line)<5:
            break
        if line.endswith('\n'):
            line = line[0:len(line)-1]
        subjLines.append(line)
        
    idLines = getWordFromFile(idFile)
    ratingLines = getWordFromFile(ratingFile)
    labelThreeLines = getWordFromFile(labelThreeFile)
    labelFourLines = getWordFromFile(labelFourFile)
    
    authorInfo = [subjLines, idLines, ratingLines, labelThreeLines, labelFourLines]
    authorInfo[lineIndex] = subjLines
    authorInfo[idIndex] = idLines
    authorInfo[ratingIndex] = ratingLines
    authorInfo[threeIndex] = labelThreeLines
    authorInfo[fourIndex] = labelFourLines
    
    return authorInfo

def importData():
    authorsInfo = []
    authorsInfo.append(authorInfo(dennisName))
    authorsInfo.append(authorInfo(jamesName))
    authorsInfo.append(authorInfo(scottName))
    authorsInfo.append(authorInfo(steveName))
    
    return authorsInfo
    
def importEmbeddings():
    print()

def createModel(features, labels, mode):
    embedding_size = w2v.wv.vector_size
    W = tf.constant(embedding_array, name="W")
    embedding_vectors = tf.nn.embedding_lookup(W, tf.cast(features["x"], tf.int32))
    
    length = tf.cast(features["lengths"], tf.int32)
    
    #word_embeddings = tf.Variable(tf.random_uniform([len(w2v.wv.vocab), embedding_size], -1.0, 1.0))
    lstm_cell = tf.nn.rnn_cell.LSTMCell(embedding_size)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, embedding_vectors, sequence_length=length, dtype=tf.float32)
    dropout = tf.nn.dropout(final_state.h, keep_prob=0.9)
    logits = tf.layers.dense(inputs=dropout, units=3, activation=tf.nn.softmax)
    
    predictions = {
        "scores": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_result")}
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32), logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9999, beta2=0.999999)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["scores"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions["scores"])
    
    
#Get the specified column of info for all the reviews of all authors.
def getAuthorInfo(authorsInfo, i):
    allEntries = authorsInfo[0][i][:]
    allEntries = allEntries + authorsInfo[1][i][:]
    allEntries = allEntries + authorsInfo[2][i][:]
    allEntries = allEntries + authorsInfo[3][i][:]
    return allEntries

#Split the data into train/eval/test data  
def splitData(data):
    l = len(data)
    print(l)
    train_data = data[0:int(l * 0.8)]
    eval_data = data[int(l*0.8):int(l*0.9)]
    test_data = data[int(l*0.9):]
    train_data = train_data[0:4000]

    train_data = np.asarray(train_data)#convertToArray(train_data)
    eval_data = np.asarray(eval_data)#convertToArray(eval_data)
    test_data = np.asarray(test_data)#convertToArray(test_data)
    return train_data, eval_data, test_data

def convertToArray(list2DToConvert):
    newArray = np.asarray([])
    for entry in list2DToConvert:
        np.concatenate((newArray , np.asarray(entry)))
    return newArray

def model3(allStatements, allThreeLabels, statementLengths):
    moddir = "../../savedModels/MovieReviewSentiments"
        
    sentimentClassifier = tf.estimator.Estimator(model_fn = createModel, model_dir = moddir)
    
    train_statements, eval_statements, test_statements= splitData(allStatements)
    train_labels, eval_labels, test_labels = splitData(allThreeLabels)
    train_lengths, eval_lengths, test_lengths = splitData(statementLengths)
    tensors_to_log = {}
    
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_statements, "lengths": train_lengths},
        y=train_labels,
        batch_size=40,
        num_epochs=50,
        shuffle=True)
    
    sentimentClassifier.train(
        input_fn=train_input_fn,
        steps=5000,
        hooks=[logging_hook])
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_statements, "lengths": eval_lengths},
        y=eval_labels,
        shuffle=False,
        num_epochs=1)
    eval_results = sentimentClassifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    

def shuffleData(inputs, labels):
    num = len(inputs)
    numArray = np.arange(0, num)
    np.random.shuffle(numArray)
    
    shuffledInput = []
    shuffledLabels = []
    

    for i in range(0, num):
        shuffledInput.append(inputs[numArray[i]])
        shuffledLabels.append(labels[numArray[i]])
    
    shuffledInput = np.asarray(shuffledInput)
    shuffledLabels = np.asarray(shuffledLabels)
    
    return shuffledInput, shuffledLabels

def main(unused):
    s = [['unk']]
    w2v.build_vocab(sentences=s, min_count=1, update=True)
    #unknown words represented by a vector of 0's
    w2v.wv.syn0[w2v.wv.vocab['unk'].index] = np.zeros((100)).astype(dtype=np.float32)
    s = [['eos']]
    #end of sequence represented by a vector of 1's
    w2v.build_vocab(sentences=s, min_count=1, update=True)
    w2v.wv.syn0[w2v.wv.vocab['eos'].index] = np.ones((100)).astype(dtype=np.float32)

    global embedding_array
    vocab_size = len(w2v.wv.vocab)
    for i in range(0, vocab_size):
        word = w2v.wv.index2word[i]
        embedding_array.append(w2v.wv[word])
    embedding_array = np.array(embedding_array, np.float32)
    
    authorsInfo = importData()
    
    allStatements = np.asarray(getAuthorInfo(authorsInfo, lineIndex))
    allThreeLabels = np.asarray(getAuthorInfo(authorsInfo, threeIndex))
    
    allStatements, allThreeLabels = shuffleData(allStatements, allThreeLabels)
    
    allStatements, allLengths = preprocess_text.preProcessInput(allStatements)
    allThreeLabels = preprocess_text.preProcess3Labels(allThreeLabels)
    
    
    model3(allStatements, allThreeLabels, allLengths)
    
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)