# Titanic-survival-data
This is a model to predict the survival of a given person using the titanic dataset.


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, roc_curve, roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


df = pd.read_csv(r'C:\Users\user\Downloads\train.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'SibSp', 'Parch', 'Fare']].values
y = df['Survived'].values
kf = KFold(n_splits=5, shuffle = True)
# DecisionTree model
print('for Decision Tree model')
model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 27  )
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.predict([[3, True, 22, 1, 0, 7.25]]))
print('accuracy:', model.score(X_test, y_test))
print(accuracy_score(y_test, y_pred))
print('precision:', precision_score(y_test, y_pred))
print('recall:', recall_score(y_test, y_pred))
print('f1 score:', f1_score(y_test, y_pred))
# LogisticRegression model
print()
print('For Logistic Regression model')
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X_test)
print(model.predict([[3, True, 22, 1, 0, 7.25]]))
print('accuracy:', model.score(X_test, y_test))
print(accuracy_score(y_test, y_pred))
print('precision:', precision_score(y_test, y_pred))
print('recall:', recall_score(y_test, y_pred))
print('f1 score:', f1_score(y_test, y_pred))


def score_model(X, y, kf):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression(solver = 'liblinear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    print('accuracy:', np.mean(accuracy_scores))
    print('precision:', np.mean(precision_scores))
    print('recall score:', np.mean(recall_scores))
    print('f1_score:', np.mean(f1_scores))
print()    
print('Logistic REgression with 5 Fold cross validation')
score_model(X, y, kf)

