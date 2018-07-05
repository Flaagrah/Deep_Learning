import numpy as np
from movieReviewModels import preprocess_text
from movieReviewModels import max_length
from movieReviewModels import w2v
from movieReviewModels import embedding_size

def preProcessInput(passages):
    #Split the words into an array
    passages = [passage.split(' ') for passage in passages]
    #Remove punctuation
    print(passages[0])
    #indices = [[word if (word in w2v.wv.vocab) else 'unk' for word in passage] for passage in passages]
    indices = [[w2v.wv.vocab[word].index if (word in w2v.wv.vocab) else w2v.wv.vocab['unk'].index for word in passage] for passage in passages]
    print(indices[0])
    return indices

def preProcess3Labels(labelList):
    newLabels = np.eye(3)[np.array(labelList).astype(np.int32)]
    return newLabels