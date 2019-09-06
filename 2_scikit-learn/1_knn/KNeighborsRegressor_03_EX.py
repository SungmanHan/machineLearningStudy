# -*- coding: utf-8 -*-

# 사이킷 런의 load_diabetes 데이터를 
# 최근접 이웃 알고리즘으로 분석한 후, 
# 모델의 평가점수(R2) 및, 평균제곱오차, 
# 평균절대오차 값을 확인하세요.

import pandas as pd
# 사이킷 런에서 제공하는 회귀분석용 데이터
from sklearn.datasets import load_diabetes

# 사이킷 런의 데이터 로딩
diabetes = load_diabetes()

# 회귀분석용 데이터의 경우 target_names 키가 없음
# - target 데이터 자체가 정답이기 때문에...
print(diabetes.keys())

# 특성 데이터 추출
X = pd.DataFrame(diabetes.data)

# 라벨 데이터 추출
y = pd.Series(diabetes.target)

# 특성 데이터의 개수 및 결측 데이터 확인
print(X.info())

# 특성 데이터의 스케일 확인
pd.options.display.max_columns=100
print(X.describe())

# 라벨 데이터의 스케일 확인
# - 회귀분석을 위한 데이터 셋일 경우에만 확인
print(y.describe())

# 데이터 분할(학습/테스트)
# (회귀분석을 위한 데이터 셋의 경우 
# y 데이터의 비율을 유지하기 위한 
# stratify 속성은 사용하지 않음)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=0.3, random_state=1)

# 데이터 분할 확인
print('학습 데이터 개수 : ', X_train.shape[0])
print('테스트 데이터 개수 : ', X_test.shape[0])

# 데이터 분석을 위한 모델의 생성 및 학습
from sklearn.neighbors import KNeighborsRegressor

# 검색할 이웃의 개수를 변수로 선언
K = 8

# 모델 객체의 생성과 학습을 동시에 진행할 수 있음
# (사이킷 런의 모든 예측기 클래스의 fit 메소드는
# X 데이터의 형태를 2차원으로 
# y 데이터의 형태를 1차원으로 간주함)
model = KNeighborsRegressor(
        n_neighbors=K, n_jobs=-1).fit(
                X_train, y_train)

# 모델의 평가
# score 메소드
# - 분류 예측기 : 정확도를 반환
# - 회귀 예측기 : R2-score(결정계수)를 반환
# (일반적으로 -1 ~ 1 사이의 값이 반환)
print('학습 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

predicted = model.predict(X_train)
print('학습 - 평균절대오차(MAE) : ', 
      mean_absolute_error(y_train, predicted))
print('학습 - 평균제곱오차(MSE) : ', 
      mean_squared_error(y_train, predicted))

predicted = model.predict(X_test)
print('테스트 - 평균절대오차(MAE) : ', 
      mean_absolute_error(y_test, predicted))
print('테스트 - 평균제곱오차(MSE) : ', 
      mean_squared_error(y_test, predicted))







