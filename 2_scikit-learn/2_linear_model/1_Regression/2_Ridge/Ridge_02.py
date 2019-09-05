# -*- coding: utf-8 -*-

# boston housing data set(회귀분석용 데이터 셋)
from sklearn.datasets import load_boston

# 사이킷 런에서 제공되는 데이터를
# X(data키)와 y(target키)로 전달받는 방법
# - numpy 배열 타입이 반환됨
X, y = load_boston(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, random_state=1)

from sklearn.linear_model import LinearRegression, Ridge

lr_model = LinearRegression().fit(X_train, y_train)

alpha=1.7
ridge_model = Ridge(alpha=alpha).fit(X_train, y_train)

print('LR 학습 결과 : ', lr_model.score(X_train, y_train))
print('Ridge 학습 결과 : ', ridge_model.score(X_train, y_train))

print('LR 테스트 결과 : ', lr_model.score(X_test, y_test))
print('Ridge 테스트 결과 : ', ridge_model.score(X_test, y_test))

import numpy as np
from matplotlib import pyplot as plt

coef_range = np.arange(1, X_train.shape[1] + 1)

plt.plot(coef_range, lr_model.coef_, 'ro')
plt.plot(coef_range, ridge_model.coef_, 'bx')

plt.axhline(0, color='y', linestyle='--')
plt.show()


















