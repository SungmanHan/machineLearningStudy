# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:11:06 2019

@author: powergen
"""

# 일반적인 머신러닝 단계
# - 성능향상을 위한 데이터 전처리 추가
# - 하이퍼 파라메터의 검색 단계 추가
# - 파이프 라인을 사용한 데이터의 전치리 및 학습을 자동화


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

# 2. 데이터 전처리 및 머신러닝 모델 학습을 위한 파이프라인 생성
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

# Pipeline 클래스
# - 데이터의 전처리 과정과 머신러닝 모델의 학습 및 예측 과정을 동시에 실행할 수 있는 기능을 제공 
pipe = Pipeline([('scaler',MinMaxScaler()),
                 ('svm',SVC(C=1.0,gamma='scale'))
                 ])

pipe.fit(X_train,y_train)


# 3. 학습된 머신러닝 모델 객체의 평가
from sklearn.metrics import classification_report
pred_train = pipe.predict(X_train)
pred_test = pipe.predict(X_test)

print('학습 평가 :\n',pipe.score(X_train,y_train))
print('학습 평가 :\n',pipe.score(X_test,y_test))
print('학습 데이터의 정밀도, 재현율, F1')
print(classification_report(y_train,pred_train))
print('테스트 데이터의 정밀도, 재현율, F1')
print(classification_report(y_test,pred_test))