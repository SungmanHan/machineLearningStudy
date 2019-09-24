# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:11:06 2019

@author: powergen
"""

# 일반적인 머신러닝 단계
# - 성능향상을 위한 데이터 전처리 추가

# 1. 데이터의 적재 및 분할
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# 데이터 적제
cancer = load_breast_cancer()
# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
                                        cancer.data,
                                        cancer.target,
                                        stratify=cancer.target,
                                        random_state=1)
# 2. 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 머신러닝 모델의 객체 생성 및 학습
from sklearn.svm import SVC
model = SVC(C=1.0,gamma='scale').fit(X_train_scaled,y_train)

# 4. 학습된 머신러닝 모델 객체의 평가
print('학습 평가 :\n',model.score(X_train_scaled,y_train))
print('학습 평가 :\n',model.score(X_test_scaled,y_test))

from sklearn.metrics import classification_report
pred_train = model.predict(X_train_scaled)
pred_test = model.predict(X_test_scaled)

print('학습 데이터의 정밀도, 재현율, F1')
print(classification_report(y_train,pred_train))

print('테스트 데이터의 정밀도, 재현율, F1')
print(classification_report(y_test,pred_test))