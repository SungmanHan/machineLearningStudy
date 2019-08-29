# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

data = {
    "year" : [2017, 2018, 2019, 2020],
    "GDP Rate" : [2.8, 3.1, 3.0, np.NaN], 
    "GDP" : ['1.859M', '1.859M', '2.237M', np.NaN]
}

df = pd.DataFrame(data)

# info 메소드는 현재 데이터프레임 내부에
# 존재하는 데이터 전체 개수
# 각 컬럼의 존재하는 실제 데이터 개수
# (NULL이 아닌 데이터)
# 각 컬럼의 데이터 타입 정보를 출력
print(df.info())

# 데이터프레임에 저장된 수치데이터에 대한
# 간략한 통계 정보를 출력할 수 있는 
# describe 메소드
# (수치데이터에 대한 정보만 출력)
print(df.describe())

# 각각의 열에 대해서 통계 메소드를 제공
print(df['year'].sum())
print(df['year'].mean())
print(df['year'].max())
print(df['year'].min())

print("=" * 17)
# 특정 열에 저장된 데이터의 개수를 반환
# (각각의 데이터 개수를 반환)
print(df.GDP.value_counts())

print("=" * 17)
# value_counts 함수의 결과를
# 전체 데이터의 개수로 나누어서
# 비율을 확인할 수 있음
print(df.GDP.value_counts() / len(df))

from sklearn.datasets import load_breast_cancer

_, y = load_breast_cancer(return_X_y=True)

label = pd.Series(y)











































