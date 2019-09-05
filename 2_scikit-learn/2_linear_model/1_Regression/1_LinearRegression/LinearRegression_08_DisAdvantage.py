# -*- coding: utf-8 -*-

# LinearRegression의 단점
# - 데이터의 규모가 커지게 되면 계산이 불가능해짐
# - 학습 데이터에 대해서 오버피팅(과적합)되는 경향이 보임

from sklearn.datasets import load_boston

# 사이킷 런에서 제공되는 데이터를
# X(data키)와 y(target키)로 전달받는 방법
# - numpy 배열 타입이 반환됨
X, y = load_boston(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)

print('학습 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

import numpy as np
from matplotlib import pyplot as plt

# X데이터의 특성(컬럼) 개수 확인
print(X_train.shape[1])
# LinearRegression 객체의 가중치 변수 확인
print(model.coef_)

# X축의 데이터
coef_range = np.arange(1, model.coef_.shape[0] + 1)

plt.plot(coef_range, model.coef_, 'ro')
plt.axhline(0, color='y', linestyle='--')
plt.show()

















