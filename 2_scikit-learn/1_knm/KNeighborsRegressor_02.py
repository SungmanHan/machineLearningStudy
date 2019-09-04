# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 20:52:36 2019

@author: powergen
"""
import pandas as pd

from sklearn.datasets import load_boston

# 1. 데이터 로드
bostan = load_boston()
# - 특성 데이터 추출
X = pd.DataFrame(bostan.data)
# - 라벨 데이터 추출
y = pd.Series(bostan.target)

# 데이터 확인
pd.options.display.max_columns = 13

# 결측 데이터 여부 확인
print(X.info())

# 특성 데이터의 스케일 확인
print(X.describe())

# 회귀 분석용 데이터일 경우 라벨 데이터 스케일도 확인
print(y.describe())

# 3. 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                                X.values,
                                                y.values,
                                                test_size=0.25,
                                                random_state=1
                                                )

# 수치 값(연속된 값)을 예측할 수 있는 KNeighborsRegressor 클래스
from sklearn.neighbors import KNeighborsRegressor
# 회기 분석 시 짝수를 사용해도 됨
R = 0
T = 0
RK = 0
for i in range(1,20,1) :
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


print('최적 K : ',RK)
print('학습 평가 : ',model.score(X_train,y_train))
print('테스트 평가 : ',model.score(X_test,y_test))

from sklearn.metrics import r2_score

pewsixrws = model.predict(X_train)
print('결정계수(R2) : ',r2_score(y_train,pewsixrws))

from sklearn.metrics import mean_absolute_error
predicted = model.predict(X_train)
print('평군절대오차(MAE) :', mean_absolute_error(y_train,predicted))

from sklearn.metrics import mean_squared_error
predicted = model.predict(X_train)
print('평군제곱오차(MSE) :',mean_squared_error(y_train,predicted))

