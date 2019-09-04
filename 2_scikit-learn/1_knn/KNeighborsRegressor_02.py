# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_boston

# 사이킷 런의 데이터 로딩
boston = load_boston()

# 특성 데이터 추출
X = pd.DataFrame(boston.data)

# 라벨 데이터 추출
y = pd.Series(boston.target)

# 특성 데이터의 개수 및 
# 결측 데이터 여부 확인
print(X.info())

# 특성 데이터의 스케일 확인
print(X.describe())

# 라벨 데이터의 스케일 확인
# - 회귀분석을 위한 데이터 셋일 
#   경우에만 확인
print(y.describe())

# 데이터 분할(학습/테스트)
# (회귀분석을 위한 데이터 셋의 경우 
# y 데이터의 비율을 유지하기 위한 
# stratify 속성은 사용하지 않음)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=0.2, random_state=1)

# 데이터 분석을 위한 모델의 생성 및 학습
from sklearn.neighbors import KNeighborsRegressor

K=4
model=KNeighborsRegressor(
        n_neighbors=K, 
        n_jobs=-1).fit(X_train, y_train)

# 머신러닝 모델의 평가
print('학습 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

# R2(결정계수) 스코어를 반환할 수 있는 r2_score 함수
from sklearn.metrics import r2_score

predicted = model.predict(X_train)
print('결정계수(R2) : ', r2_score(y_train, predicted))

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

predicted = model.predict(X_train)
print('학습-평균절대오차(MAE) : ', 
      mean_absolute_error(y_train, predicted))
print('학습-평균제곱오차(MSE) : ', 
      mean_squared_error(y_train, predicted))

predicted = model.predict(X_test)
print('테스트-평균절대오차(MAE) : ', 
      mean_absolute_error(y_test, predicted))
print('테스트-평균제곱오차(MSE) : ', 
      mean_squared_error(y_test, predicted))

















