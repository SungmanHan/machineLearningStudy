# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

X = np.arange(1, 11).reshape(-1, 1)
y = np.array([1 if data > 5 else 0 for data in X])

print(X)
print(y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs').fit(X, y)

print('ÇÐ½À Æò°¡ : ', model.score(X, y))

plt.figure(figsize=(10, 7))
for index, data in enumerate(X) :
    plt.scatter(data, data, c='k',
                marker='x' if y[index] == 0 else 'D')

# 라벨이 0 인 데이터의 경우 0 미만의 값이 나오며,ㅡ
# 라벨이 1 인 데이터의 경우 0 이상의 값이 반환되도록
# 가중치와 절편의 값이 계산된 것을 확인할 수 있음
plt.plot(X, X * model.coef_ + model.intercept_, 'r--')    

plt.grid(True)
plt.show()
















