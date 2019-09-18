# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:59:06 2019

@author: powergen
"""

import pandas as pd

fname = '../../data/diabetes.csv'

diabetes = pd.read_csv(fname)

diabetes.columns = ['A','B','C','D','E','F','G','H','LABEL']

print(diabetes.info())

diabetes.iloc[[0,1],0] = None
print(diabetes.info())

print(diabetes.head(3))

mean_A = diabetes.A.mean()
print('A열 평균 데이터 : ',mean_A)

diabetes.A.fillna(mean_A,inplace=True)

print(diabetes.info())
print(diabetes.head(3))








