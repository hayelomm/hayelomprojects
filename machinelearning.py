import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

dataset = pd.read_csv('CombinedNormal&DDOS_Dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#dataset.info()

#spliting data to training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)



classifiers = [GaussianNB(), 
               LogisticRegression(solver='lbfgs'),
               DecisionTreeClassifier(), 
               KNeighborsClassifier(n_neighbors=8),
               RandomForestClassifier()]
for cls in classifiers:
    cls.fit(X_train, y_train)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,10))

for cls, ax in zip(classifiers, axes.flatten()):
    plot_confusion_matrix(cls, 
                          X_test, 
                          y_test, 
                          ax=ax, 
                          cmap='Blues',
                         display_labels=['DDOS','Normal'])
    ax.title.set_text(type(cls).__name__)
plt.tight_layout()  
plt.show()

print("======================Classification Report of GNB=============================")
gnb = GaussianNB()
gnb = gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:",result2)
print("===================Classification Report of LR================================")
lr = LogisticRegression(solver='lbfgs')
lr = lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:",result2)
print("===================Classification Report of DT=================================")
dt = DecisionTreeClassifier()
dt = dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:",result2)
print("========================Classification Report of KNN==========================")
knn = KNeighborsClassifier(n_neighbors=8)
knn = knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:",result2)
print("===================Classification Report of RF================================")
rf = RandomForestClassifier()
rf = rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test, y_pred)
print("Accuracy:",result2)
