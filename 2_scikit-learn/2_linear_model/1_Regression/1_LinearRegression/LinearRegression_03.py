# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

X = np.array([6, 8, 10, 14, 18]).reshape(-1,1)
y = np.array([7, 9, 13, 17.5, 18.7])

X_2 = X ** 2
X_3 = X ** 3

X = np.c_[X, X_2, X_3]


# 위의 X, y를 학습하여 (LinearRegression 클래스 사용)
# X 데이터가 3 ~ 20 인 경우의 y의 값을
# 그래프로 출력하세요(산점도차트로 출력)

# 모델의 생성과 학습
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)

print(model.coef_)
print(model.intercept_)

# 테스트 데이터 셋 생성
X_test = np.arange(3, 21).reshape(-1,1)

X_test_2 = X_test ** 2
X_test_3 = X_test ** 3

X_test = np.c_[X_test, X_test_2, X_test_3]

print('예측 결과 : ', model.predict(X_test))

plt.figure(figsize=(10,7))
plt.title('Pizza Price(inch)')
plt.xlabel('inches')
plt.ylabel('prices')

#plt.plot(X, y, 'kx')
#plt.plot(X_test, model.predict(X_test), 'ro')

plt.plot(X[:,0], y, 'kx')
plt.plot(X_test[:,0], model.predict(X_test), 'ro')

plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()












