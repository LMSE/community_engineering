#!/usr/bin/python
# -*- coding: utf-8 -*-



import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

"""
Trains and evaluates a RandomForestClassifier using a subset of the data defined by command-line arguments. It preprocesses 
the target variable, fits the model on the training data, predicts on the test data, and appends the model's accuracy to a text file. The 
process is iterated by increasing the amount of training data and evaluating on subsequent test data sets.

"""
print(__doc__)





import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import DyMMMSettings as settings



communityDir=communitiesDir=settings.simSettings["communitiesDir"]+"/error/"+sys.argv[1]+"/"


X_train=pd.read_csv(communityDir+"X_train.csv")
FEATURE_NAMES = X_train.columns.tolist()
y_train=pd.read_csv(communityDir+"y_train.csv")
print(y_train.shape)

print("-----------------------------------------------------------")
print(X_train.shape)
print(y_train.shape)


y_train.loc[y_train['CSI'] < 0.9] = 0 
y_train.loc[y_train['CSI'] >= 0.9] = 1
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
y_train=encoded_Y

import numpy as np
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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

h = .02  # step size in the mesh

# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]

names = ["Random Forrest"]

classifiers = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    #DecisionTreeClassifier(max_depth=2,random_state=0),
    RandomForestClassifier(max_depth=2, n_estimators=100, max_features=X_train.shape[1], random_state=0),
    # MLPClassifier(alpha=1, max_iter=1000),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()
    ]

trainRows=int(sys.argv[2])
testRows=int(sys.argv[3])
print("----------------------------------------------------------------------------")
        # iterate over classifiers
for name, clf in zip(names, classifiers):
    print(name)
    X_trainrows=X_train[:trainRows]
    y_trainrows=y_train[:trainRows]
    score = cross_val_score(clf, X_trainrows, y_trainrows, cv=int(X_trainrows.shape[0]/500))
    clf.fit(X_trainrows, y_trainrows)
    X_testrows=X_train[trainRows:trainRows+testRows]
    y_testrows=y_train[trainRows:trainRows+testRows]
    print("----------------------------------------------------------------------------")
    print(trainRows)
    print(trainRows+testRows)
    print(X_train.shape)
    print(y_train.shape)
    print(X_testrows)
    print(y_testrows)
    y_pred=clf.predict(X_testrows)
    y_pred_prb=clf.predict_proba(X_testrows)
    accuracy=accuracy_score(y_testrows, y_pred, normalize=True)
    file1 = open(communityDir+"/modelError.txt", "a")  # append mode
    file1.write("Accuracy for %s: %0.1f%% %0.1f%% \n" % (name, score.mean()*100 , accuracy * 100))
    file1.close()        
    trainRows+=testRows

