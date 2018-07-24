#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================
"""

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
import os, numpy, shutil
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.externals import joblib
import warnings
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
import time


warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

h = .02  # step size in the mesh
NEWLINE = '\n'
SKIP_FILES = {'.DS_STORE'}

path = os.getcwd()

#  READ ALL LINES IN ALL FILES IN ALL DIRECTORIES IN ROOT
def read_files(path):
    # each folders in the root directory
    for root, dir_names, file_names in os.walk(path):
        # each file
        for file_name in file_names:
            # ignore the .DS_Store file
            if file_name and not file_name.startswith("."):
                # create filepath for each file
                file_path = os.path.join(root, file_name)
                # if there is a file that exists in this file path
                if os.path.isfile(file_path):
                    # create an array called lines
                    lines = []
                    # open the file
                    f = open(file_path, encoding="latin-1")
                    # append each lines in that array
                    for line in f:
                        lines.append(line)
                        # print(line)
                    #     close the file
                    f.close()
                    # join all the lines
                    content = NEWLINE.join(lines)
                    # yield the file_path and content
                    yield file_path, content


# NOW CREATE A DATA FRAME FROM ALL THE INFO YOU COLLECTED
# body text in text and class in the other column. index is filename
def build_data_frame(path, classification):
    # Make an array for rows and one for index
    rows = []
    index = []
    # for each file name and text that read_files returns
    for file_name, text in read_files(path):
        # append new row with text label followed by text and class label followed by class
        rows.append({'text': text, 'class': classification})
        # append the index with the filename.
        index.append(file_name)

    # create a new data frame with the rows and the index arrays as parameters
    data_frame = DataFrame(rows, index=index)
    # send the dataframe object back
    return data_frame


ENG = "ENGLISH"
SPA = "SPANISH"
FRN = "FRENCH"

# classification - 'eng' and 'spa' are the folders in which the training documents are. ENG and SPA
# are the classifications.
SOURCES = [
    ('eng',    ENG),
    ('spa',    SPA),
    ('frn',    FRN)
]

# create a DataFrame object that is like a dictionary of arrays in rows
data = DataFrame({'text': [], 'class': []})
# data is going to be the data frame created in build-dataframe.
# This is going to read each file and append each files text, into a new row, along with the classification
# All files in 'eng' are going to be classified as ENG
# All files in 'spa' are going to be classified as SPA
for path, classification in SOURCES:
    # we get the path and classification from SOURCES
    data = data.append(build_data_frame(path, classification))
#     shuffles the data apparently
data = data.reindex(numpy.random.permutation(data.index))


# ======== COUNT VECTORIZER AND CLASSIFIER ====================
# # Creates the object that will count the words in a document
# count_vectorizer = CountVectorizer()
# # returns the count value for the text values in data's text column
# X = count_vectorizer.fit_transform(data['text'].values)
#
# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         # "Naive Bayes", "QDA"]
#
classifiers = [
    MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(13, 13, 13), learning_rate='constant', learning_rate_init=0.001, max_iter=500, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=None, shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False),
    AdaBoostClassifier(),
    MultinomialNB(),
    RidgeClassifier(tol=1e-2, solver="lsqr"),
    PassiveAggressiveClassifier(n_iter=50),
    SGDClassifier(alpha=.0001, n_iter=50),
    RandomForestClassifier(n_estimators=100)]

start_time = time.time()
print("Starting time now")
#============== CONFUSION ============================

pipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  RandomForestClassifier(n_estimators=100) )])

k_fold = KFold(n=len(data), n_folds=4)
scores = []
confusion = numpy.array([[0, 0], [0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    X_train = data.iloc[train_indices]['text'].values
    y_train = data.iloc[train_indices]['class'].values
#
    X_test = data.iloc[test_indices]['text'].values
    y_test = data.iloc[test_indices]['class'].values
    #
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print("x train data:", X_train.shape, y_train.shape)
    print("")
    print("x test data:", X_test.shape, y_test.shape)
    confusion = confusion_matrix(y_test, predictions)
    score = f1_score(y_test, predictions, average=None)
    scores.append(score)

print("Total time: ", (time.time()-start_time)/60, "mins")
# print('Total documents classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)

from matplotlib.colors import LogNorm
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

norm_conf = []
for i in confusion:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.viridis, interpolation='nearest')
# Hi(tler) -Jimmy
width, height = confusion.shape

for x in range(width):
    for y in range(height):
        ax.annotate(str(confusion[x][y]), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
alphabet = ('ENG', 'SPA', 'FRN')
plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])
plt.show()
