DISCLAIMER
Since the data is shuffled every time the model is run before splitting it into train/eval/test data, the training data will be different each time you run the model. Future improvements will store the results of the initial shuffle in a file to be read on future runs. 


Introduction

This model performs a 3-class sentiment classification on subjective statements in movie reviews.
Information on the dataset can be found here: movieReviewData/DataInfo.txt

Data

Information on the dataset can be found here:
movieReviewData/DataInfo.txt
	
Because of the small size of the dataset (5000 examples) means that it has to be iterated over many times and this will lead to overfitting.A maximum sequence size of 100 was chosen as a compromise between training time and amount of information for each review. This likely has a negative effect on accuracy because many of the reviews are too ambiguous in their first 100 words.

Model and Training

The model is a dynamic LSTM with a maximum sequence length of 100 (for now). A dropout layer is used as a regularizer.
The LSTM has 100 hidden units so that it is able to represent the entire embedding of each word. 
The Adam Optimizer's parameters need to be optimized because of the the large depth of the LSTM. 
Also, the LSTM has 100 hidden units so that it is able to represent a word vector This may be unnecessary and the model
may benefit from reducing the number of units because of the small size of the data.
	

Results and potential improvements

{'accuracy': 0.43512973, 'loss': 1.0959202, 'global_step': 5000}
	
The monkey score accuracy would be 0.333. Therefore, a 0.435 accuracy represents a significant improvement over the monkey score.
Of course, this is still very low and a usable sentiment classifier should have much higher accuracy,
but this does show how an LSTM could be used to perform sentiment classification.
	
Potential improvements might include finding a bigger dataset. A dataset of 5000 is far too low for training such a deep network.
Reducing the number of hidden units in the LSTM might be a good idea to reduce the "untrainable noise" in the network.
Iterating over the the dataset many more times with more regularization (eg, higher dropout rate) might be a good idea.
The Adam Optimizer needs to be modified so that the training loss can go below 0.5 (which it fails to do right now).