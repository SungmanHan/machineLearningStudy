# -*- coding: utf-8 -*-

import pandas as pd

# 유방암 데이터 셋(이진분류용 데이터 셋 - 악성/양성)
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data)
y = pd.Series(cancer.target)

# 전체 데이터의 샘플 개수
# 각 샘플의 컬럼 개수
# 각 컬럼의 결측 데이터 유무
print(X.info())

# 각 컬럼 데이터의 스케일 정보를 확인
# - 스케일 조정이 필요한 데이터 셋임을 
# 확인할 수 있음
pd.options.display.max_columns=100
print(X.describe())

# 이진 분류 데이티 셋으로 0과 1로 
# 라벨 데이터가 구성되어 있음
print(y.value_counts())
print(y.value_counts() / len(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=0.3, stratify=y.values,
        random_state=42)

# 분할된 데이터의 개수 확인
print(X_train.shape[0], X_test.shape[0])

# 예측기 클래스의 생성과 학습
# 선형 방정식을 기반으로 데이터를 분류할 수 있는 
# LogisticRegression 클래스
# - 클래스의 이름이 Regression 으로 종료되지만
#   회귀분석이 아닌 분류용 클래스임
from sklearn.linear_model import LogisticRegression

# LogisticRegression 클래스의 하이퍼 파라메터 solver
# 학습을 위해서 사용되는 알고리즘을 선택할 수 있는 파라메터
# 기본 값은 liblinear
# - 작은 데이터 셋에 잘 동작하는 알고리즘으로 L1, L2 정규화를 지원
# sag, saga
# - 대용량의 데이터를 빠르게 학습할 수 있는 알고리즘
# - 확률적 경사하강법을 기반으로 분류할 수 있는 알고리즘
# 다중 클래스의 분류 모델은 newton-cg, sag, saga 과 lbfgs 를 
# 사용해야함
# newton-cg, lbfgs, sag 알고리즘은 L2 정규화만 지원
# liblinear, saga 알고리즘은 L1 정규화도 지원함
model = LogisticRegression(
        solver='lbfgs', 
        max_iter=5000).fit(X_train, y_train)

# 분류를 위한 예측기 클래스의 score 메소드는 
# 정확도의 값을 반환
print('훈련 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

# 분류 모델의 classes_ 멤버는 분류해야할 
# 클래스(값)을 저장하는 멤버 변수입니다.
print(model.classes_)

# 예측 확률을 반환하는 predict_proba 메소드
# - 각 클래스 별 확률의 값을 반환
# - 모델이 저장하고 있는 classes_ 정보의
#   인덱스와 연결해서 확인해야함
print(model.predict(X_test[:3]))
print(model.predict_proba(X_test[:3]))
print(y_test[:3])

# LogisticRegression의 decision_function의 결과
# - 학습에 의해서 찾은 기울기(가중치)와 편향(절편)을 
#   사용하여 예측된 값을 반환
# - 음성 데이터인 경우 음수의 값, 
#   양성 데이터인 경우 양수의 값
# - decision_function의 결과 값을 이해하는 방법
#   음수의 값이 작아질수록 음성데이터일 확률이 높아짐
#   양수의 값이 커질수록 양성데이터일 확률이 높아짐
# - 분류를 위한 선으로부터 이동된 거리를 의미
print(model.decision_function(X_test[-3:]))
print(model.predict_proba(X_test[-3:]))

# 머신러닝 모델이 계산한 기울기, 절편의 값을 사용하여
# 결과를 예측할 수 있음
# (decision_function 과 동일한 결과를 반환)
import numpy as np
print(np.sum(model.coef_ * X_test[-3:], axis=1) + model.intercept_)














