from gensim.models import Word2Vec
from nltk.corpus import movie_reviews

#maximum allowed length of a sequence.
max_length = 100
#the size of the embeddings.
embedding_size = 100
#the word2vec object that stores the vocabulary and embeddings.
w2v = Word2Vec(movie_reviews.sents())