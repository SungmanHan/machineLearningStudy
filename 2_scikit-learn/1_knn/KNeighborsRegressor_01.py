# -*- coding: utf-8 -*-

import numpy as np

# 최근접 이웃 알고리즘을 사용한 회귀분석

# 학습 데이터 X
# - 키와 성별에 대한 정보
X_train = np.array([
        [158, 1],
        [170, 1],
        [183, 1],
        [191, 1],
        [155, 0],
        [163, 0],
        [180, 0],
        [158, 0],
        [170, 0]])

# 학습 데이터 y
# - 몸무게 정보
y_train = np.array([64, 86, 84, 81, 49, 59, 
                    67, 54, 67])

# 최근접 이웃 알고리즘을 사용하여 
# 수치 값(연속된 값)을 예측할 수 있는
# KNeighborsRegressor 클래스
from sklearn.neighbors import KNeighborsRegressor

# 탐색할 인접 데이터의 개수
# - 회귀 분석의 경우 홀수의 값을 지정할 필요는 없음
# - 평균 수치를 반환하기 때문에...
K=2

model=KNeighborsRegressor(
        n_neighbors=K, 
        n_jobs=-1).fit(X_train, y_train)

# 머신러닝 모델의 평가
# - score 메소드
# - 회귀분석 모델의 score 메소드는 
# R2(결정계수) 값을 반환

# R2(결정계수) 계산 공식
# 1 - (실제 정답과 모델이 예측한 값의 차이의 제곱 값 합계) / 
# (실제 정답과 정답의 평균 값 차이의 제곱값 합계)

# 분석 결과의 이해
# 일반적으로 R2(결정계수)는 -1 ~ 1 사이의 값을 반환
# 1에 가까울수록 좋은 예측을 보이는 모델이라는 의미
# 0에 가까울수록 정답 데이터의 평균치 정도를 
# 예측하는 모델이라는 의미
# -값이 커질수록 평균 수치조차 예측하지 못하는 모델
print('학습 평가 : ', model.score(X_train, y_train))

# R2(결정계수) 스코어를 반환할 수 있는 r2_score 함수
from sklearn.metrics import r2_score

predicted = model.predict(X_train)
print('결정계수(R2) : ', r2_score(y_train, predicted))

# 회귀모델에서 사용하는 평가함수
# - 평균절대오차
# - 실제 정답과 예측값의 차에 대해서 절대값을 취한 후 
#   평균값을 반환
# - (실제정답-예측값)의 절대값의 합계를 평균
# - mean_absolute_error 함수를 사용
# - mean_absolute_error(실제정답, 예측값)
# - 실제 정답에 대한 오차의 수치 값을 확인할 수 있음
from sklearn.metrics import mean_absolute_error

predicted = model.predict(X_train)
print('평균절대오차(MAE) : ', 
      mean_absolute_error(y_train, predicted))

# 회귀모델에서 사용하는 평가함수
# - 평균제곱오차
# - 실제 정답과 예측값의 차에 대해서 제곱한 후 
#   평균값을 반환
# - (실제정답-예측값)의 제곱의 합계를 평균
# - mean_squared_error 함수를 사용
# - mean_squared_error(실제정답, 예측값)
# - 일반적으로 딥러닝의 학습을 위한 오차함수로 사용
from sklearn.metrics import mean_squared_error

predicted = model.predict(X_train)
print('평균제곱오차(MSE) : ', 
      mean_squared_error(y_train, predicted))















