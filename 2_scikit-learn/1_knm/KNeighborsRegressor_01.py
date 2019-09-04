# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 20:10:43 2019

@author: powergen
"""
import numpy as np

# 최그접 이웃 알고리즘을 사용한 회귀 분석

# 학습 데이터 X : 키와 성별에 대한 정보
X_train = np.array([
      [158, 1],
      [170, 1],
      [183, 1],
      [191, 1],
      [155, 0],
      [163, 0],
      [180, 0],
      [158, 0],
      [170, 0]
    ])

# 학습 데이터 y : 몸무게 정보
y_train = np.array([
      64,86,84,81,49,59,67,54,67
    ])

# 수치 값(연속된 값)을 예측할 수 있는 KNeighborsRegressor 클래스
from sklearn.neighbors import KNeighborsRegressor
# 회기 분석 시 짝수를 사용해도 됨
K = 2
model = KNeighborsRegressor(n_neighbors=K,n_jobs=-1).fit(X_train,y_train)

# 머신러닝 머델의 평가
# score를 사용
# 분류 시 정확도 반환
# 회기는 R2(결정계수) 1 ~ -1 값을 반환
# (실제 정답과 모델이 에측한 값의 차이의 제곱 ) / 

# 1에 가까울수록 좋은 예측을 보이는 모델
# 0에 가까울수록 정답 데이터의 평균치 정도의 예측을 보이는 모델
# - 값이 커질수록 평균 수치조차 에측하지 못하는 모델
print(model.score(X_train,y_train))


from sklearn.metrics import r2_score

pewsixrws = model.predict(X_train)
print('결정계수(R2) : ',r2_score(y_train,pewsixrws))

# 회귀모델에서 사용하는 평가함수
# - 평균절대오차
# - 실제 정답과 에측값의 차에 대해서 절대값을 취한 후 평균값을 반환
# - (실제정답-예측값)의 절대값의 함계를 평균
# mean_absolute_error 함수를 사용
# mean_absolute_error (실제정답, 에측값)
# - 실제 정답에 대한 오차의 수치 값을 확인할 수 있음 
from sklearn.metrics import mean_absolute_error
predicted = model.predict(X_train)
print('평군절대오차(MAE) :', mean_absolute_error(y_train,predicted))

# 회귀모델에서 사용하는 평가함수
# - 평균제곱오차
# - 실제 정답과 에측값의 차에 대해서 제곱한 후 평균값을 반환
# - (실제정답 - 예측값)의 제곱의 합계를 평균
from sklearn.metrics import mean_squared_error
predicted = model.predict(X_train)
print('평군제곱오차(MSE) :',mean_squared_error(y_train,predicted))
















