# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

# 피자 크기(X)에 가격 데이터(y)
X = np.array([6, 8, 10, 14, 18])
y = np.array([7, 9, 13, 17.5, 18.7])

# 피자 가격을 예측하기 위한 함수의 정의
def predictPrice(data) :
    # 1차 방정식을 사용하여 예측을 구현
    # data * 기울기 + 절편
    return data * 1.3 + (-0.7)

# 선형 모델을 기반으로 예측할 수 있는 클래스
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression().fit(X.reshape(-1,1), y)

plt.figure(figsize=(10,7))
plt.title('Pizza Price(inch)')
plt.xlabel('inches')
plt.ylabel('prices')
plt.plot(X, y, 'ko')

plt.plot(X, predictPrice(X), 'r--')
plt.plot(X, lr_model.predict(X.reshape(-1,1)), 'b.-')

plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()













