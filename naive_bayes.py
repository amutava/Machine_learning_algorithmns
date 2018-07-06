#Import Library
import os
import scipy
import pandas as pd
from scipy.io import arff
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_multilabel_classification
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object model = GaussianNB() # there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link
# Train the model using the training sets and check score
model = GaussianNB()
data_path = os.path.join(os.path.pardir, 'Machine_learning_algorithmns', 'data')
data, path = arff.loadarff(os.path.join(data_path, 'yeast/yeast-train.arff'))

X, y = make_multilabel_classification(sparse = True, n_labels = 20,
return_indicator = 'sparse', allow_unlabeled = False)
# print(X, y)

# print(pd.DataFrame(data))
# print(data)
# model.fit(X, y)
# Predict Output
# predicted= model.predict(x_test)


# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

# train
classifier.fit(X, y)

# predict
# predictions = classifier.predict(X_test)