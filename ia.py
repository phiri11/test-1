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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

# this code creates a descriptive analysis of the dataframe
df = pd.read_csv('ts2.csv')
#obtengo x=descriptores, y=label
df.head()      
df.dtypes
#I pass the column 'Class' from object to int32
encoder = LabelEncoder()
encoderClass = encoder.fit_transform(df['Class'])
df['Class'] = encoderClass
encoderClass
encoder.classes_
df.head()
df.dtypes
df.describe()
print(df.groupby('Class').size())
y = df['Class'].values
X = df[['Amplitude','AndersonDarling','Autocor_length','Beyond1Std','Gskew','LinearTrend','MaxSlope','Mean','Meanvariance','MedianAbsDev','MedianBRP','PercentAmplitude','PeriodLS']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# i find the best parameters for the classifier
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])
# summarize the shape of the training and testing sets

n_neighbors = 15 #I obtained this value with the function above, where I identify with the graph that is displayed, that the most effective assertiveness is 15 and the worst would be 2
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print('Precisión del clasificador K-NN entrenamiento: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Precisión del clasificador K-NN Test: {:.2f}'
     .format(knn.score(X_test, y_test)))

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# de sklearn
def plot_cm(y_true, y_pred, figsize=(8,8)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = '0.0%'
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Valores Verdaderos'
    cm.columns.name = 'Prediccion'
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
plot_cm(y_test,pred)

#clasificador random forest
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Precisión del clasificador Random Forest entrenamiento: {:.2f}'
        .format(rf.score(X_train, y_train)))
print('Precisión del clasificador Random Forest Test: {:.2f}'
        .format(rf.score(X_test, y_test)))
plot_cm(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)



