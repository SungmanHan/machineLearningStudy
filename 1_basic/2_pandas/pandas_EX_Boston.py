# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_boston

# boston 집 가격 데이터
# 회귀분석을 위한 데이터셋
# 회귀 : 라벨 데이터가 연속된 수치로 구성되어 있는 데이터셋
# 분류 : 라벨 데이터가 범주형으로 구성되어 있는 데이터셋
boston = load_boston()

print(boston.keys())

# 데이터 셋에 대한 설명을 확인할 수 있는 DESCR 키 값
print(boston.DESCR)

# 학습을 위한 특성 데이터 추출(2차원)
X = pd.DataFrame(boston.data)
# 특성 데이터의 컬럼명을 지정
X.columns = boston.feature_names

# 학습을 위한 라벨 데이터 추출(1차원)
y = pd.Series(boston.target)

print(X)
# 특성 데이터의 개수 및 타입, 결측 데이터 여부 확인
print(X.info())
# 특성 데이터의 개수가 13이므로 기본 출력의 특성 개수를
# 100개로 확장시킴
pd.options.display.max_columns=100
# 특성데이터의 기본 통계 정보를 확인
# - 값의 스케일이 차이가 있음을 확인
print(X.describe())

# 회귀분석을 위한 데이터 셋이므로
# value_counts 메소드를 사용하여 정보를 확인할 수 없음
print(y.value_counts())
# 회귀분석을 위한 데이터 셋이므로
# 값의 기본 통계를 확인할 수 있음
print(y.describe())

y.hist()















