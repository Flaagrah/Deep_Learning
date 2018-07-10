import numpy as np
from movieReviewModels import preprocess_text
from movieReviewModels import max_length
from movieReviewModels import w2v
from movieReviewModels import embedding_size

def preProcessInput(passages):
    #Split the words into an array
    passages = [passage.split(' ') for passage in passages]
    #Get the index of every word in the review. Unk represents words that are not in the vocabulary
    indices = [[w2v.wv.vocab[word].index if (word in w2v.wv.vocab) else w2v.wv.vocab['unk'].index for word in passage] for passage in passages]
    unknownIndex = w2v.wv.vocab['eos'].index
    
    #Pad the entry with eos tokens so that the length is 100. Clip each review to length 100 if already greater than 100.
    lengths = []
    for i in range(0, len(indices)):
        l = len(indices[i])
        if l < max_length:
            padding = np.ones(max_length-l, np.float32) * unknownIndex
            indices[i] = np.append(indices[i], padding)
        else:
            indices[i] = indices[i][0:max_length]
            l = max_length
        lengths.append(len(indices[i]))
    
    indices = np.array(indices)
    lengths = np.array(lengths)  
    return indices, lengths

def preProcess3Labels(labelList):    
    return np.array(labelList).astype(np.int32)