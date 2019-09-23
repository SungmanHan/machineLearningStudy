# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:11:34 2019

@author: powergen
"""
# 머신러닝에서 문자 형태로 저장된
# 특성 데이터(샘플/입력/X데이터)를 사용하여
# 학습하는 방법

# 1. 문자형의 데이터를 수치데이터로 변경
# - 문자열로 구성된 각 단어를 구분하여 라벨 값을 지정
# - 문자열을 구성하고 있는 각 단어의 빈도수를 계산
# - 문자열의 형태를 각 단어의 빈도수로 변경

# DictVectorizer class
# - 문자열을 구성하는 각 단어의 수를 세어놓은
# - 딕셔너리 타입에서 BOW 벡터를 생성
# - BOW : Bag Of Words

from sklearn.feature_extraction import DictVectorizer

dicts = [{'A':1,'B':2},{'B':3,'C':1,'D':2}]

vertorizer = DictVectorizer()
dict_transform = vertorizer.fit_transform(dicts)

# 희소 행렬 데이터를 확인할 수 있다.
print('변환 결과(희소 행렬) : \n',dict_transform)
print('변환 결과(벡터 행렬) : \n',dict_transform.toarray())

print('구성 단어의 이름 : \n',vertorizer.feature_names_)
print('변환 : \n',vertorizer.transform({'A':2,'D':1}))
print('변환 : \n',vertorizer.transform({'A':2,'D':1}).toarray())
  


