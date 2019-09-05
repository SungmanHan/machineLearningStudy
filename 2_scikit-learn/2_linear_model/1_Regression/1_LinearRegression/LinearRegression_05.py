# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

X = np.array([6, 8, 10, 14, 18]).reshape(-1,1)
y = np.array([7, 9, 13, 17.5, 18.7])

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)

# LinearRegression 클래스의 비용(손실)함수
# (머신러닝 모델의 비용(손실)함수는 
# 머신러닝 모델의 학습 완성도를 판단하는 기준)
# (성능이 좋은 머신러닝 모델은 비용(손실)함수의 
# 결과가 작음)
# 잔차 : 훈련데이터를 통해서 예측된 결과과 
# 실제 정답 사이의 오차
# LinearRegression 클래스의 모델은 잔차의 합계를 
# 최소화할 수 있는 가중치, 절편을 찾아내는 것이 최종 목표

# 오차 계산 방법
# 모델의 예측 결과와 실제 정답 사이의 오차 값을 계산한 후
# 제곱한 값의 합계를 구합니다. 
# 그리고 합계의 평균을 오차로 사용
# (모델의 예측 결과 - 실제 정답) ** 2 의 평균값

# 1. 예측 결과를 반환
predicted = model.predict(X)
print('예측 결과 : ', predicted)

# 2. 모델의 예측 결과와 실제 정답 사이의 오차 값을
# 계산한 후, 제곱
loss = (y - predicted) ** 2
print('loss : ', loss)

# 3. 제곱된 값의 합계를 구한 후, 평균 값을 반환
loss_avg = np.mean(loss)
print('모델의 잔차(오차) : ', loss_avg)

from sklearn.metrics import mean_squared_error
print('모델의 평균제곱오차(MSE) : ', 
      mean_squared_error(y, predicted))










