The first step in running my code would be to download the neccessary libraries. This includes nltk (pip install nltk), tensorflow (pip install tensorflow), NumPy (pip install numpy), and possibly keras to be safe 
(pip install keras). After that, you must download the Kaggle dataset from the link included in my report. Please update the file path in my evaluate_DATA641_HW3.py file to be the location of your IMDB dataset.
From there, you just need to make sure that all of my python files I included are downloaded and in the same folder, and run the evaluate_DATA641_HW3 file. The outputs will be the training accuracy, training
loss, validation accuracy, and validation loss over each epoch shown, along with the test accuracy, the F1 score, and the percentage of predictions in each class. These outputs come from the train_and_evaluate
function.
