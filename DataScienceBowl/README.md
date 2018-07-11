Disclaimer:

	Set workspace to <workspace>/DataScienceBowl/src. This program trains a model indefinitely. In order to create a submission file,
	Change the line in model.py to trainModel(False, True) and run it. To continue training, change it to trainModel(True, False).
	Testing creates a submission.csv file under the src folder which can be submitted to https://www.kaggle.com/c/data-science-bowl-2018.
	
	THE DATA IS NOT INCLUDED IN GITHUB BECAUSE OF SIZE RESTRICTIONS.

Introduction:

	This project is based on the Kaggle competition "Data Science Bowl 2018". The goal of the competition was to detect nuclei within images.
	This model is an implementation of the yolo algorithm and is supplemented with width/height normalization and data augmentation
	by adjusting brightness and shifting the image in various directions. The images are compressed to 256x256 pixels and divided by
	16x16 segments such that each segment is 16x16 pixels. Each segment has a flag to indicate whether the segment has a nuclei in it.
	A description of the Yolo architecture can be found here: https://pjreddie.com/media/files/papers/yolo.pdf

Network Model:

	The model is a UNET architecture that is based on the following research paper: https://arxiv.org/pdf/1505.04597.pdf

	The implementation of the model was taken from: https://www.kaggle.com/piotrczapla/tensorflow-u-net-starter-lb-0-34
	
	The loss function takes into account the flag of every segment (which indicates that there is a hit within the segment) and the
	width, height, x, y of every segment that has a hit (ie, every segment with a nuclei in it). This ignores the width/height/x/y 
	of segments that don't have hits because they are not relevant to the accuracy.
	
	All resulting values in the output layer are between 0.0 and 1.0. The width and height of each segment are represented
	by the square root of the width/height of the image itself (ie, if a nuclei is 10% the height of the image height,
	then height is 0.1). The x and y coordinates are represented by the coordinates relative to the segment divided by the 
	segment width/height. For example, if the centre of the nuclei is in the center of the segment, then the x and y would be 0.5 
	and 0.5 respectively. If the nuclei is in the top left of the segment, then x and y would be 0.0 and 0.0 respectively. 
	
	The output of the model ends being 16X16X5 in size. 16X16 grid segments with each segment having a flag, width variable,
	height variable, x, and y. Output size is 1280.
	
Data:

	The data is stored in a file after it is initially preprocessed. This is because it takes a long time for the images 
	to be processed and so it speeds up testing if the processed data is stored in a file for future use.
	
Data Augmentation:

	Data augmentation is done by creating lighter and darker versions of each picture and shifting the picture in various directions.
	The manipulation of the brightness ensures the the network learns to deal with a wider array of brightness while shifting the
	picture ensures that the model is trained to recognize all segments.
	
Normalization:
	
	Normalization takes the square root of the width and height of the images (all values are between 0.0 and 0.1). This is magnifies
	the width/height error of smaller nuclei. The error in smaller nuclei is more significant than for bigger nuclei. For example, 
	an error of 2 pixels has a greater effect on a nuclei of size 10 (20% miss), than a nuclei that is 100 pixels (2% miss). 
	Therefore the model should be more sensitive to error on smaller nuclei.
	
Output of model:

	The model generates a submission.csv file that predicts bounding boxes for the test images. The format of the file 
	and the measurement of the result is described here: https://www.kaggle.com/c/data-science-bowl-2018#evaluation
	
Results and suggestions for improvement:

	As of right now, Kaggle reports a score of 0.043, which is extremely low. However this is not due to a flaw in the model
	as a quick eye test of the results and the test images reveal that the bounding boxes are converging towards the nuclei.
	
	One problem is likely that the loss is not weighted. The x and y centre of the bounding box should be considered the most 
	important variables for each segment because of it's influence on the IOU of the image. The normalization of the width/height
	renders them less important. Furthermore, the flag is the least important variable because it is effectively rounded to 0.0 or 1.0 anyway.
	
	It is possible that the model would also benefit from more training as improvements due to training have not plateaued. 
	The IOU threshold and the flag threshold should be modified and experimented with to find optimal values.
	
	Another potential improvement is to forego the YOLO model and have the output directly represent each pixel in the 256X256 image.
	YOLO might be superfluous because there is only one object class (nuclei). Since there is no need for classification, bounding
	boxes are unnecessary. Given the way that the kaggle scoreboard measures it, a lot of false positives can be avoided
	by directly selecting pixels rather than create a rectangular bounding box.
	