# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:11:55 2019

@author: powergen
"""

import pandas as pd

# 1. 데이터 로드
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

# 2. 데이터 추출 
# 특성 추출
X = pd.DataFrame(diabetes.data)
# 라벨 추출
y = pd.Series(diabetes.target)

# 3. 데이터 확인
pd.options.display.max_columns = 10

# 결측 데이터 여부 확인
print(X.info())

# 특성 데이터의 스케일 확인
print(X.describe())

# 회귀 분석용 데이터일 경우 라벨 데이터 스케일도 확인
print(y.describe())

from matplotlib import pyplot as plt
y.hist()
plt.show()

# 4. 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                       X.values,
                                       y.values,
                                       test_size = 0.2,
                                       random_state = 1
                                    )

# 5. 학습
from sklearn.neighbors import KNeighborsRegressor

R = 0
T = 0
RK = 0

for i in range(1,101,1) :
  K = i
  model = KNeighborsRegressor(n_neighbors=K,n_jobs=-1).fit(X_train,y_train)
  TT = model.score(X_train,y_train)
  TR = model.score(X_test,y_test)
  if R == 0 :
    T = TT
    R = TR
    RK = K
  elif R < TR :
    T = TT
    R = TR
    RK = K
  
# 6. 결과
print('-' * 30)
print('최적 K : ',RK)
print('학습 평가 : ',T)
print('테스트 평가 : ',R)

from sklearn.metrics import r2_score
pewsixrws = model.predict(X_train)
print(' 학습 결정계수(R2) : ',r2_score(y_train,pewsixrws))
pewsixrws = model.predict(X_test)
print(' 테스트 결정계수(R2) : ',r2_score(y_test,pewsixrws))

from sklearn.metrics import mean_absolute_error
predicted = model.predict(X_train)
print('학습 평군절대오차(MAE) :', mean_absolute_error(y_train,predicted))
predicted = model.predict(X_test)
print('테스트 평군절대오차(MAE) :', mean_absolute_error(y_test,predicted))

from sklearn.metrics import mean_squared_error
predicted = model.predict(X_train)
print('학습 평군제곱오차(MSE) :',mean_squared_error(y_train,predicted))
predicted = model.predict(X_test)
print('테스트 평군제곱오차(MSE) :',mean_squared_error(y_test,predicted))
