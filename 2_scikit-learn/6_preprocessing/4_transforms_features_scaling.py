# -*- coding: utf-8 -*-

# 특성 데이터의 스케일링 처리
# 각각의 특성 데이터들은 고유의 특성과 분포를 가지고 있음
# 각각의 특성 데이터의 값을 그대로 사용하게 되면 
# 학습 속도가 저하되거나 학습이 되지 않는 문제가 
# 발생할 수 있음
# (최근접 이웃 알고리즘 및 SVM 알고리즘)
# 이러한 경우 사이킷 런의 Scaler 클래스를 이용하여 
# 각각의 특성 데이터들을 일정 범위로 스케일링할 수 있음

# 주의사항
# 일부 알고리즘(결정트리, 랜덤포레스트)의 경우
# 특성 데이터를 스케일링 처리를 하게 되면
# 성능이 감소될 수 있음

import numpy as np

# 각 특성의 스케일이 차이가 존재하는 데이터
data = np.array([[-1,2],[-0.5,6],[0,10],[1,18]])

import pandas as pd

X = pd.DataFrame(data)
# 스케일 확인
print(X.describe())

# 각 수치데이터의 최소/최대값을 기준으로 정규화를
# 처리할 수 있는 MinMaxScaler 클래스
# - 각 열의 데이터를 0 ~ 1 사이로 압축하는 역할
# - 반드시 수치 데이터만을 전달해야 함
from sklearn.preprocessing import MinMaxScaler
scaler_mm = MinMaxScaler().fit(X.values)
print(X.values)
print(scaler_mm.transform(X.values))

# 표준정규분포를 따르는 값으로 변환시킬 수 있는 
# StandardScaler 클래스
# - 각 열의 데이터들이 최소값과 최대값이 한정되지 않음
# - 일반적으로 StandardScaler 정규화를 처리하는 것이
#   데이터 분석 성능이 높아짐
from sklearn.preprocessing import StandardScaler
scaler_ss = StandardScaler().fit(X.values)
print(X.values)
print(scaler_ss.transform(X.values))

# 중앙값(median)과 IQR(interquartile range)를 
# 사용하여 아웃라이어의 영향을 최소화하며 
# 변환할 수 있는 RobustScaler 클래스
from sklearn.preprocessing import RobustScaler
scaler_rs = RobustScaler().fit(X.values)
print(X.values)
print(scaler_rs.transform(X.values))






















