# Purpose
The purpose of this project was to create a basic spam detection model and learn more about the intricacies
of the whole process of creating one.

# Files
## models
This is a folder containing the models saved by spam_model_train.py that are imported into spam_model_test.py

## vect_and_scale
This folder contains the TF-IDF vectorizers used for each sub-model, and additionally contains the scaler
used for scaling the engineered features. Saved by spam_model_train.py and imported into spam_model_test.py

## spam.csv
This CSV file contains the data for training the spam detector. The first column is the category (spam or not spam), while
the second column contains the message.

## spam_model_arch.py
This file houses the architechture for the model, as well as the engineered_features function for extracting
specific features of a message to use while training

## spam_model_test.py
This file tests specific messages a user can define in order to get a sense of the model and some of its characteristics

## spam_model_train.py
This file creates, trains, and saves the models that are used to create singlular predictions on whether a message
is spam or not
