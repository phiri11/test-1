import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score

# this code creates a descriptive analysis of the dataframe
df = pd.read_csv('ts2.csv')
#obtengo x=descriptores, y=label
df.head()       
df.dtypes
encoder = LabelEncoder()
encoderClass = encoder.fit_transform(df['Class'])
df['Class'] = encoderClass
encoderClass
encoder.classes_
df.head()
df.dtypes
#now I assign descriptors and category
x = df.iloc[-1,1]
y = df.iloc[-1,2]
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# summarize the shape of the training and testing sets
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# create the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
# fit the training data to the classifier
X_train.array.reshape(1,-1)
classifier.fit(X_train, y_train)
# predict the labels of the test data
y_pred = classifier.predict(X_test)
# generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
# create the classifier
classifier = KNeighborsClassifier(n_neighbors=5)
# fit the training data to the classifier
classifier.fit(X_train, y_train)
# predict the labels of the test data
y_pred = classifier.predict(X_test)
# generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
