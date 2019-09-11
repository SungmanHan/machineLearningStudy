# -*- coding: utf-8 -*-

# LinearSVC 클래스를 사용하여 load_wine 데이터를 분석하고
# 정확도 및 정밀도, 재현율을 확인하세요.

import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()

X = pd.DataFrame(wine.data)
y = pd.Series(wine.target)

print(X.info())
print(X.describe())

print(y.value_counts())
print(y.value_counts() / len(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        stratify=y.values, random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import LinearSVC

model_l2 = LinearSVC(C=1,
        max_iter=10000).fit(X_train, y_train)

model_l1 = LinearSVC(C=1, penalty='l1', dual=False,
        max_iter=10000).fit(X_train, y_train)

print('학습 평가(l2) : ', 
      model_l2.score(X_train, y_train))
print('학습 평가(l1) : ', 
      model_l1.score(X_train, y_train))

print('테스트 평가(l2) : ', 
      model_l2.score(X_test, y_test))
print('테스트 평가(l1) : ', 
      model_l1.score(X_test, y_test))

from sklearn.metrics import classification_report

pred_train_l2 = model_l2.predict(X_train)
pred_test_l2 = model_l2.predict(X_test)

pred_train_l1 = model_l1.predict(X_train)
pred_test_l1 = model_l1.predict(X_test)

print('classification_report - 학습(l2)')
print(classification_report(y_train, pred_train_l2))

print('classification_report - 테스트(l2)')
print(classification_report(y_test, pred_test_l2))

print('classification_report - 학습(l1)')
print(classification_report(y_train, pred_train_l1))

print('classification_report - 테스트(l1)')
print(classification_report(y_test, pred_test_l1))










