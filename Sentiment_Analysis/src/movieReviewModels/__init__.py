from gensim.models import Word2Vec
from nltk.corpus import movie_reviews

max_length = 3000
embedding_size = 100
w2v = Word2Vec(movie_reviews.sents())