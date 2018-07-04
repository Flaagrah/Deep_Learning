import numpy as np
from movieReviewModels import preprocess_text
from movieReviewModels import max_length
from movieReviewModels import w2v
from movieReviewModels import embedding_size

def preProcessInput(passages):
    #Split the words into an array
    passages = [passage.split(' ') for passage in passages]
    #Remove punctuation
    passages = [[word for word in passage if (not len(word)==1) or word=='a' or word=='i'] for passage in passages]
    #Convert to matrices and randomize matrix for unknown words.
    #word_matrice = [[w2v.wv[word] if (word in w2v.wv.vocab) else np.random.uniform(-0.5,0.5,(len(passage), 100)) for word in passage ] for passage in passages]
    #word_matrice = [np.concatenate(passage, np.zeros((max_length-len(passage), 100), axis=0)) for passage in word_matrice if len(passage)<max_length]
    print("matrice")
    return passages

def preProcess3Labels(labelList):
    newLabels = np.eye(3)[np.array(labelList).astype(np.int32)]
    return newLabels