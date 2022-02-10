# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:28:39 2022

@author: Pierre
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("train.csv", index_col = "PassengerId")
    
    X = df.drop("Survived", axis = 1)
    y = df['Survived']
    
    X = X.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
    
    X = X.fillna(X.mean())
    
    encoder = LabelEncoder()
    
    X['Sex'] = encoder.fit_transform(X['Sex'])
    X['Embarked'] = encoder.fit_transform(X['Embarked'])
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

    return X_train, X_test, y_train, y_test


def fit_a_model(model_name, X_train, y_train):
    if model_name == 'Logistic Regression':
        model = LogisticRegression().fit(X_train, y_train)
        
        return model
    if model_name == 'KNN':
        model = KNeighborsClassifier().fit(X_train, y_train)
        
        return model
    if model_name == 'Decision Tree':
        model = DecisionTreeClassifier().fit(X_train, y_train)
        
        return model