# -*- coding: utf-8 -*-

# Data 디렉토리에 저장된 diabetes.csv 파일의 데이터를
# 분석하여 정확도 및 정밀도, 재현율을 출력하세요.
# (LogisticRegression 클래스를 활용하되, C의 값과 penalty를
# 제어하여 결과를 확인)

import pandas as pd

fname = '../../../../data/diabetes.csv'
diabetes = pd.read_csv(fname, header=None)

X = diabetes.iloc[:,:-1]
y = diabetes.iloc[:, -1]

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        stratify=y.values, random_state=1)

print(X_train.shape[0], X_test.shape[0])

print(y.value_counts() / len(y))

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
        solver='lbfgs', C=1.0, 
        max_iter=10000).fit(X_train, y_train)

print('학습 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

model = LogisticRegression(penalty='l1',
        solver='liblinear', C=100.0, 
        max_iter=10000).fit(X_train, y_train)

print('학습 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

from sklearn.metrics import confusion_matrix

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print("confusion_matrix - 학습 데이터")
print(confusion_matrix(y_train, pred_train))

print("confusion_matrix - 테스트 데이터")
print(confusion_matrix(y_test, pred_test))

from sklearn.metrics import classification_report

print("classification_report - 학습 데이터")
print(classification_report(y_train, pred_train))

print("classification_report - 테스트 데이터")
print(classification_report(y_test, pred_test))









