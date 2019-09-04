# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:48:48 2019

@author: powergen
"""

import numpy as np

X = np.arange(1,11).reshape(-1,1)

y = np.arange(1,11) * 10

from sklearn.neighbors import KNeighborsRegressor

K = 2
model = KNeighborsRegressor(n_neighbors=K).fit(X,y)

print(model.predict([[5.3]]))
print(model.predict([[7.9]]))
print(model.predict([[150.3]]))
print(model.predict([[100000]]))

# 선형 모델을 기반으로 예측할 수 있는 클래스
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression(n_jobs=-1).fit(X,y)

print(lr_model.predict([[5.3]]))
print(lr_model.predict([[7.9]]))
print(lr_model.predict([[150.3]]))
print(lr_model.predict([[100000]]))
