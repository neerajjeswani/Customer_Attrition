#Import Libraries
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#Load Data Set
X_train = np.genfromtxt("/Users/neerajjeswani/Desktop/Telco/X_train.csv", delimiter=",")
X_val = np.genfromtxt("/Users/neerajjeswani/Desktop/Telco/X_val.csv", delimiter=",")
Y_train = np.genfromtxt("/Users/neerajjeswani/Desktop/Telco/Y_train.csv", delimiter=",")
Y_val = np.genfromtxt("/Users/neerajjeswani/Desktop/Telco/Y_Val.csv", delimiter=",")

####### DECISION TREE #######

##Tables for classification errors
error_val =[]
error_train =[]
tree_depth =[]
auc_train = []
auc_val = []

min_error_train = 1
tree_depth_train = 0
min_error_val = 1
tree_depth_val = 0

for i in range(1,45):
    d_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 7, max_depth = i)
    d_tree.fit(X_train, Y_train)
    
    ##Training Classification error
    Y_pred = d_tree.predict(X_train)
    error = 1 - accuracy_score(Y_train,Y_pred)
    error_train.append(error)
    model_roc_auc = roc_auc_score(Y_train,Y_pred)
    auc_train.append(model_roc_auc)
    
    if (min_error_train>error):
        min_error_train=error
        tree_depth_train = i
    
    ##Validation Classification error
    Y_pred = d_tree.predict(X_val)
    error = 1 - accuracy_score(Y_val,Y_pred)
    error_val.append(error)
    model_roc_auc = roc_auc_score(Y_val,Y_pred)
    auc_val.append(model_roc_auc)
    
    if (min_error_val>error):
        min_error_val=error
        tree_depth_val = i
    
    tree_depth.append(i)

#AUC and Confusion Matrix for the tree with the minimum error for validation data set
d_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 7, max_depth = 6)
d_tree.fit(X_train, Y_train)

Y_pred = d_tree.predict(X_train)
error = 1 - accuracy_score(Y_train,Y_pred)
error_train.append(error)
model_roc_auc = roc_auc_score(Y_train,Y_pred)
confusion_matrix(Y_train,Y_pred)

Y_pred = d_tree.predict(X_val)
error = 1 - accuracy_score(Y_val,Y_pred)
error_train.append(error)
model_roc_auc = roc_auc_score(Y_val,Y_pred)
confusion_matrix(Y_val,Y_pred)

####### RANDOM FOREST #######
    
from sklearn.ensemble import RandomForestClassifier

##Tables for classification errors
error_train =[]
error_val =[]
auc_train = []
auc_val = []

min_error_train = 1
tree_depth_train = 0
min_error_val = 1
tree_depth_val = 0

for k in range(1,45):
    
    rf = RandomForestClassifier(criterion='entropy', n_estimators=k, max_features = 'sqrt')
    rf.fit(X_train, np.ravel(Y_train,order='C'))

    ##Training Classification error
    Y_pred = rf.predict(X_train)
    error = round(1 - accuracy_score(Y_train,Y_pred),4)
    error_train.append(error)
    model_roc_auc = roc_auc_score(Y_train,Y_pred)
    auc_train.append(model_roc_auc)
    
    if (min_error_train>error):
        min_error_train=error
        tree_depth_train = i

    ##Validation Classification error
    Y_pred = rf.predict(X_val)
    error = round(1 - accuracy_score(Y_val,Y_pred),4)
    error_val.append(error)
    model_roc_auc = roc_auc_score(Y_val,Y_pred)
    auc_val.append(model_roc_auc)
    
    if (min_error_val>error):
        min_error_val=error
        tree_depth_val = i

#AUC and Confusion Matrix for the tree with the minimum error for validation data set
rf = RandomForestClassifier(criterion='entropy', n_estimators=44, max_features = 'sqrt')
rf.fit(X_train, np.ravel(Y_train,order='C'))

Y_pred = rf.predict(X_train)
error = round(1 - accuracy_score(Y_train,Y_pred),4)
error_train.append(error)
model_roc_auc = roc_auc_score(Y_train,Y_pred)
confusion_matrix(Y_train,Y_pred)

Y_pred = rf.predict(X_val)
error = round(1 - accuracy_score(Y_val,Y_pred),4)
error_val.append(error)
model_roc_auc = roc_auc_score(Y_val,Y_pred)
confusion_matrix(Y_val,Y_pred)

####### LOGISTIC REGRESSION #######
   
from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression()
LogReg.fit(X_train, Y_train)
Log = LogReg.fit(X_train, Y_train)
y_pred = LogReg.predict(X_train)

error = 1 - LogReg.score(X_train, Y_train)
print(error)
model_roc_auc = roc_auc_score(Y_train,y_pred) 
print(model_roc_auc)
confusion_matrix(Y_train,y_pred)

y_pred = LogReg.predict(X_val)

error = 1 - LogReg.score(X_val, Y_val)
print(error)
model_roc_auc = roc_auc_score(Y_val,y_pred) 
print(model_roc_auc)
confusion_matrix(Y_val,y_pred)

coefficients_log = LogReg.coef_

####### NAIVE BAYES #######
   
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, Y_train)

y_pred = model.predict(X_train)

error = 1- model.score(X_train, Y_train)
print(error)
model_roc_auc = roc_auc_score(Y_train,y_pred) 
print(model_roc_auc)
confusion_matrix(Y_train,y_pred)

y_pred = model.predict(X_val)
error = 1- model.score(X_val, Y_val)
print(error)
model_roc_auc = roc_auc_score(Y_val,y_pred) 
print(model_roc_auc)
confusion_matrix(Y_val,y_pred)

####### K NEAREST NEIGHBOR #######
   
from sklearn.neighbors import KNeighborsClassifier

neighbor = [3,4,5,7]

for i in neighbor:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    
    Y_pred = knn.predict(X_train)
    error = 1 - accuracy_score(Y_train,Y_pred)
    print(error)
    model_roc_auc = roc_auc_score(Y_train,Y_pred) 
    print(model_roc_auc)
    confusion_matrix(Y_train,Y_pred)
    
    Y_pred = knn.predict(X_val)
    error = 1 - accuracy_score(Y_val,Y_pred)    
    print(error)
    model_roc_auc = roc_auc_score(Y_val,Y_pred) 
    print(model_roc_auc)
    confusion_matrix(Y_val,Y_pred)

####### XGBOOST #######

from xgboost import XGBClassifier

error_val =[]
error_train =[]
tree_depth =[]
auc_train = []
auc_val = []

min_error_train = 1
tree_depth_train = 0
min_error_val = 1
tree_depth_val = 0

for i in range(1,45):
    clf =XGBClassifier(n_estimators=i,seed=777)

    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_train)
    error = 1 - accuracy_score(Y_train,Y_pred)
    error_train.append(error)
    model_roc_auc = roc_auc_score(Y_train,Y_pred)
    auc_train.append(model_roc_auc)
    
    if (min_error_train>error):
        min_error_train=error
        tree_depth_train = i
    
    ##Validation Classification error
    Y_pred = clf.predict(X_val)
    error = 1 - accuracy_score(Y_val,Y_pred)
    error_val.append(error)
    model_roc_auc = roc_auc_score(Y_val,Y_pred)
    auc_val.append(model_roc_auc)
    
    if (min_error_val>error):
        min_error_val=error
        tree_depth_val = i
    
    tree_depth.append(i)

#AUC and Confusion Matrix for the tree with the minimum error for validation data set
clf =XGBClassifier(n_estimators=29,seed=777)

clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_train)
error = 1 - accuracy_score(Y_train,Y_pred)
error_train.append(error)
model_roc_auc = roc_auc_score(Y_train,Y_pred)
confusion_matrix(Y_train,Y_pred)

Y_pred = clf.predict(X_val)
error = 1 - accuracy_score(Y_val,Y_pred)
error_val.append(error)
model_roc_auc = roc_auc_score(Y_val,Y_pred)
confusion_matrix(Y_val,Y_pred)
 
