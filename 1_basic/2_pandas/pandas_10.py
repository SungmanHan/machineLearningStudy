# -*- coding: utf-8 -*-

import pandas as pd

# 데이터프레임의 병합
# merge 함수의 사용

df1 = pd.DataFrame({
    '이름': ['영희', '철수', '철수'],
    '국어순위': [1, 2, 3]})

df2 = pd.DataFrame({
    '성명': ['영희', '영희', '철수'],
    '수학순': [4, 5, 6]})

# 두개의 데이터프레임을 결합하기 위한 컬럼명이 서로 다른 경우
# left_on, right_on 인수를 사용하여 기준열을 명시할 수 있음
print(pd.merge(df1, df2, left_on='이름', right_on="성명"))

df3 = pd.merge(df1, df2, left_on='이름', right_on="성명")

print(df3[['이름','국어순위','수학순']])

df3 = df3[['이름','국어순위','수학순']]

print(df3)










