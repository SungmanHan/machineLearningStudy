# -*- coding: utf-8 -*-

# winequality-white.csv 데이터를 AdaBoostClassifier, 
# GradientBoostingClassifier로 분석한 후, 결과를 확인하세요.

import pandas as pd

fname='../../../data/winequality-white.csv'
wine = pd.read_csv(fname, sep=';')

X = wine.iloc[:,:-1].values
y = wine.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

ada_model = AdaBoostClassifier(
        DecisionTreeClassifier(
                max_depth=5, 
                random_state=1), 
                n_estimators=10000,
                random_state=1).fit(X_train, y_train)

gb_model = GradientBoostingClassifier(
        max_depth=5,        
        max_features=0.5,
        n_estimators=10000,
        random_state=1).fit(X_train, y_train)

print('학습 평가(ada) : ', 
      ada_model.score(X_train, y_train))
print('학습 평가(gb) : ', 
      gb_model.score(X_train, y_train))

print('테스트 평가(ada) : ', 
      ada_model.score(X_test, y_test))
print('테스트 평가(gb) : ', 
      gb_model.score(X_test, y_test))





















