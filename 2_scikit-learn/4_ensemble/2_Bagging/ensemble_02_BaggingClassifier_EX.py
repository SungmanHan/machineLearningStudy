# -*- coding: utf-8 -*-

# winequality-white.csv 데이터를 
# BaggingClassifier 클래스를 이용하여 
# 분석한 후, 결과를 확인하세요.

import pandas as pd

fname = '../../../data/winequality-white.csv'
wine = pd.read_csv(fname, sep=';')

X = wine.iloc[:,:-1].values
y = wine.quality.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(
        max_depth=7,
        random_state=1).fit(X_train, y_train)

print('학습 평가(dt) : ', 
      dt_model.score(X_train, y_train))

from sklearn.ensemble import BaggingClassifier
ensemble = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(
                random_state=1),
                n_estimators=10000,
                max_samples=0.2,
                max_features=0.2,
                random_state=1,
                n_jobs=-1).fit(X_train, y_train)

print('학습 평가(ensemble) : ', 
      ensemble.score(X_train, y_train))

print('테스트 평가(dt) : ', 
      dt_model.score(X_test, y_test))
print('테스트 평가(ensemble) : ', 
      ensemble.score(X_test, y_test))