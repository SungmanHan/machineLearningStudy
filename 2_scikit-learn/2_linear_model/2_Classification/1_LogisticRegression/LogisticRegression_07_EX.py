# -*- coding: utf-8 -*-

# LogisticRegression 클래스를 사용하여 
# 사이킷 런에서 제공하는 load_iris 데이터를 분석하고
# 정확도 및 정밀도, 재현율을 확인하세요.

import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

X = pd.DataFrame(iris.data)
y = pd.Series(iris.target)

print(X.info())
print(X.describe())

print(y.value_counts())
print(y.value_counts() / len(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        stratify=y.values, random_state=1)

from sklearn.linear_model import LogisticRegression

model_ovr = LogisticRegression(C=1000,
        solver='lbfgs', 
        multi_class='ovr').fit(X_train, y_train)

model_multi = LogisticRegression(C=100,
        solver='lbfgs', max_iter=10000,
        multi_class='multinomial').fit(X_train, y_train)

print('학습 평가(ovr) : ', 
      model_ovr.score(X_train, y_train))
print('학습 평가(multinomial) : ', 
      model_multi.score(X_train, y_train))

print('테스트 평가(ovr) : ', 
      model_ovr.score(X_test, y_test))
print('테스트 평가(multinomial) : ', 
      model_multi.score(X_test, y_test))

from sklearn.metrics import classification_report

pred_train_ovr = model_ovr.predict(X_train)
pred_test_ovr = model_ovr.predict(X_test)

pred_train_multi = model_multi.predict(X_train)
pred_test_multi = model_multi.predict(X_test)

print('classification_report - 학습(ovr)')
print(classification_report(y_train, pred_train_ovr))

print('classification_report - 테스트(ovr)')
print(classification_report(y_test, pred_test_ovr))

print('classification_report - 학습(multinomial)')
print(classification_report(y_train, pred_train_multi))

print('classification_report - 테스트(multinomial)')
print(classification_report(y_test, pred_test_multi))











