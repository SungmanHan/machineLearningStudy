# -*- coding: utf-8 -*-

import pandas as pd

fname = '../../data/score.csv'
scores = pd.read_csv(fname)

X = scores.values
y = scores.score.values

print(X)
print(y)

# 사용자 정의 전처리 클래스의 정의
from sklearn.base import BaseEstimator, TransformerMixin
class DataSelector (BaseEstimator, TransformerMixin) : 
    def __init__(self) :
        pass
    def fit(self, X, y=None) :
        return self
    def transform(self, X) :
        return X.iloc[:,2:].values

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([('selector', DataSelector()),
                 ('scaler', MinMaxScaler()),
                 ('lr_model', LinearRegression())])
    
pipe.fit(scores, y)

print('평가 : ', pipe.score(scores, y))

















