# -*- coding: utf-8 -*-

# winequality-white.csv 데이터를 SVC로 분석한 후, 
# 결과를 확인하세요.

import pandas as pd

fname='../../data/winequality-white.csv'
wine = pd.read_csv(fname, sep=';')

X = wine.iloc[:,:-1]
y = wine.iloc[:, -1]

pd.options.display.max_columns=100
#print(X.describe())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, 
        stratify=y.values, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC

model = SVC(gamma='scale').fit(X_train, y_train)

print('학습 평가 : ', 
      model.score(X_train, y_train))

print('테스트 평가 : ', 
      model.score(X_test, y_test))

from sklearn.ensemble import BaggingClassifier

ensemble = BaggingClassifier(
        SVC(gamma='scale'),
        n_estimators=10000,
        max_samples=0.3,
        max_features=0.3,
        random_state=1,
        n_jobs=-1).fit(X_train, y_train)

print('학습 평가(ensemble) : ', 
      ensemble.score(X_train, y_train))

print('테스트 평가(ensemble) : ', 
      ensemble.score(X_test, y_test))













