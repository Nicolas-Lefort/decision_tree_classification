# Decision tree classification
# data also available from https://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis
# Dataset for Sensorless Drive Diagnosis Data Set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from utils import measure_error

# feature names for dataframe
col_names = []
for i in range(1,49):
    col_names.append("feature_"+str(i))
col_names.append("labels")
# import data
df = pd.read_csv('Sensorless_drive_diagnosis.txt', sep=" ", names=col_names, header=None)
# define target feature
target = col_names[-1]
# remove missing values
df.dropna(inplace=True)
# labels are already encoded, no further data preprocessing necessary here for decision tree
# define x and y
X, y = df.drop(columns=target), df[target]
# split train test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # shuffle True default
# define the model
param_grid = {'max_depth':range(1, 10),
              'max_features': range(1, 10)}

GR = GridSearchCV(estimator=DecisionTreeClassifier(criterion="gini", random_state=42),
                  param_grid=param_grid,
                  scoring='accuracy',
                  n_jobs=-1)  # use all processors
# fit to train data
GR = GR.fit(X_train, y_train)
# predict test and train data
y_train_pred = GR.predict(X_train)
y_test_pred = GR.predict(X_test)
# check variance
train_test_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),
                                 measure_error(y_test, y_test_pred, 'test')],
                                axis=1)
print(train_test_error)
# recover best model parameters
print(GR.best_params_)
# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
disp.plot(cmap=plt.cm.Blues)
plt.show()