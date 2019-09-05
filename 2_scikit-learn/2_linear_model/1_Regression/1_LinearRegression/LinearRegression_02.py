# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

# 피자 크기(X)에 가격 데이터(y)
# - 사이킷런의 모든 예측기는 입력데이터(X)의 형태를
#   2차원 배열로 가정하고 있기 때문에
#   전달할 입력데이터를 reshape를 통해서
#   형태를 변경
X = np.array([6, 8, 10, 14, 18]).reshape(-1,1)

# 정답 데이터는 일차원 배열로 선언
y = np.array([7, 9, 13, 17.5, 18.7])

# 입력데이터에 대한 정답 데이터(연속된 수치)를
# 선형 방정식을 사용하여 예측할 수 있는 
# LinearRegression 클래스
from sklearn.linear_model import LinearRegression

# LinearRegression 클래스의 객체 생성 매개변수
# copy_X : 입력데이터를 복사 여부
# fit_intercept : 절편의 값을 계산 여부
# normalize : 정규화 여부
# n_jobs : 데이터 분석에 사용할 코어 개수
# (기본값 1, -1로 입력하는 경우 사용가능한 모든 코어를 사용)
model = LinearRegression()

# 사이킷런의 모든 예측기들은 fit 메소드를 제공하며
# fit 메소드는 입력된 X, y 데이터를 학습하는 기능을 제공
# LinearRegression 클래스의 fit 메소드는
# 입력된 데이터에 최적화되는 선형방정식을 계산
# 선형 방정식 : x1 * w1 + x2 * w2 ... + b
# 아래의 경우 : x * w(가중치/기울기) + b(절편/편향)
model.fit(X, y)

print('모델이 계산한 기울기의 값 : ',  model.coef_)
print('모델이 계산한 절편의 값 : ',  model.intercept_)

print('기울기와 가중치를 사용하여 예측')
print(X * model.coef_ + model.intercept_)

print('머신러닝 모델의 predict 메소드를 사용하여 예측')
print(model.predict(X))

# 머신러닝 모델의 평가
print('학습 평가(R2) : ', model.score(X, y))

from sklearn.metrics import mean_absolute_error
print('학습 평가(MAE) : ', 
      mean_absolute_error(y, model.predict(X)))

plt.figure(figsize=(10,7))
plt.title('Pizza Price(inch)')
plt.xlabel('inches')
plt.ylabel('prices')
plt.plot(X, y, 'ko')

# 가격 예측선
# (모델을 통해서 예측된 가격을 사용)
plt.plot(X, model.predict(X), 'b.-')

plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()











