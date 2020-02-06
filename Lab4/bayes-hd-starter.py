# bayes-hd-starter.py
# parsons/25-feb-2017
#
# The input data is the processed Cleveland data from the "Heart
# Diesease" dataset at the UCI Machine Learning repository:
#
# https://archive.ics.uci.edu/ml/datasets/Heart+Disease
#
# The code to load a csv file is based on code written by Elizabeth Sklar for Lab 1.


import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#
# Define constants
#

datafilename = 'processed.cleveland.data' # input filename
age      = 0                              # column indexes in input file
sex      = 1
cp       = 2
trestbps = 3
chol     = 4
fbs      = 5
restecg  = 6
thalach  = 7
exang    = 8
oldpeak  = 9
slope    = 10
ca       = 11
thal     = 12
num      = 14 # this is the thing we are trying to predict

# Since feature names are not in the data file, we code them here
feature_names = [ 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num' ]

num_samples = 303 # size of the data file. 
num_features = 13

#
# Open and read data file in csv format
#
# After processing:
# 
# data   is the variable holding the features;
# target is the variable holding the class labels.

try:
    with open( datafilename ) as infile:
        indata = csv.reader( infile )
        data = np.empty(( num_samples, num_features ))
        target = np.empty(( num_samples,), dtype=np.int )
        i = 0
        for j, d in enumerate( indata ):
            ok = True
            for k in range(0,num_features): # If a feature has a missing value
                if ( d[k] == "?" ):         # we do't use that record.
                    ok = False
            if ( ok ):
                data[i] = np.asarray( d[:-1], dtype=np.float64 )
                target[i] = np.asarray( d[-1], dtype=np.int )
                i = i + 1
except IOError as iox:
    print('there was an I/O error trying to open the data file: ' + str( iox ))
    sys.exit()
except Exception as x:
    print('there was an error: ' + str( x ))
    sys.exit()

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#for i in range(len(X_train)):
#    a = X_train[i]
#    print(str(next(reversed(a))))

# Here is are the sets of feastures:
data
# Here is the diagnosis for each set of features:
target

#print (y_train)
#print (len(X_train))
lst = []
for i in range(len(y_train)-1):
    if target[i] == 0:
        lst.append(i)
#print(lst)
lstfbs = []

for i in range(len(lst)):
    a = X_train[i]
    if a[5] == 0:
        lstfbs.append(a)

for i in range(len(lstfbs)):
    print (i)

for j in range(len(lst)):
    print (j)

print(i/j)
# How many records do we have?
num_samples = i
print("Number of samples:", num_samples)

