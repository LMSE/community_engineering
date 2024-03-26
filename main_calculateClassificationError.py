#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier Error
=====================

Uses RandomForrest to calculate classification Error

"""
print(__doc__)


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import DyMMMSettings as settings

"""
This code uses a RandomForestClassifier to predict binary outcomes based on a threshold applied to a continuous 
target variable. 
"""

def generateRangesScalar(paramsRangeFile):
    paramsRangeFileDf=pd.read_csv(paramsRangeFile)
    minValueRange=paramsRangeFileDf['MinValue'].tolist()
    maxValueRange=paramsRangeFileDf['MaxValue'].tolist()
    scaler=[MinMaxScaler() for i in range(len(minValueRange))]
    [scaler[i].fit([[minValueRange[i]], [maxValueRange[i]]]) for i in range(len(minValueRange))]
    return minValueRange, maxValueRange, scaler, paramsRangeFileDf

def applyScaler(X, scaler):
    X_n=np.copy(X)
    for i in range(X.shape[1]):
        X_n[:,i]=scaler[i].transform([X.iloc[:,i].to_numpy()])
    return(X_n)


analysisDir=settings.simSettings["analysisDir"]
minValueRange, maxValueRange, scaler, paramsRangeDf = generateRangesScalar(analysisDir+"\screening_inputparams.csv")

X_train=pd.read_csv(analysisDir+"/X_train.csv")
FEATURE_NAMES = X_train.columns.tolist()
print(X_train)
X_train=applyScaler(X_train, scaler)
print(X_train)
y_train=pd.read_csv(analysisDir+"/y_train.csv")
print(y_train)

y_train.loc[y_train['CSI'] < 0.9] = 0 
y_train.loc[y_train['CSI'] >= 0.9] = 1
print(y_train)
X_test=pd.read_csv(analysisDir+"/X_test.csv")
print(X_test)
X_test=applyScaler(X_test, scaler)
print(X_test)
y_test=pd.read_csv(analysisDir+"/y_test.csv")
print(y_test)
y_test.loc[y_test['CSI'] < 0.9] = 0 
y_test.loc[y_test['CSI'] >= 0.9] = 1

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
y_train=encoded_Y

encoder = LabelEncoder()
encoder.fit(y_test)
encoded_Y = encoder.transform(y_test)
y_test=encoded_Y

names = ["Random Forrest"]

classifiers = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    #DecisionTreeClassifier(max_depth=2,random_state=0),
    RandomForestClassifier(max_depth=5, n_estimators=100, max_features=17),
    # MLPClassifier(alpha=1, max_iter=1000),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()
    ]

for name, clf in zip(names, classifiers):
    score = cross_val_score(clf, X_train, y_train, cv=50)
    print(score)
    print(score.mean())
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    y_pred_prb=clf.predict_proba(X_test)
    print(y_pred)
    print(y_pred_prb)
    accuracy=accuracy_score(y_test, y_pred, normalize=True)
    print("\nAccuracy (train) for %s: %0.1f%% \n" % (name, accuracy * 100))
    logFile = analysisDir+"/classifier.log"
    f = open(logFile, "a")
    f.write("\nAccuracy (train) for %s: %0.1f%% \n" % (str(score), accuracy * 100))
    f.write(str(y_pred_prb))
    f.close()
