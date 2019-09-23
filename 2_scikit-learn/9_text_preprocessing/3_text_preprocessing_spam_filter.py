# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:59:51 2019

@author: powergen
"""


smsCSVPath = '../../data/sms.csv'

import pandas as pd

sms = pd.read_csv(smsCSVPath)

print(sms.info())
print(sms.describe())

X_raw = sms.message
y = sms.label

# y데이터의 편향 확인
print(y.value_counts())
print(y.value_counts() / len(y))

from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw.values,y.values,stratify=y.values,random_state=1)

print(X_train_raw.shape[0], X_test_raw.shape[0])

# 문자열 데이터를 수치데이터로 변환
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer().fit(X_train_raw)

print("토큰 개수 : \n",len(vectorizer.vocabulary_))

X_train = vectorizer.transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
              solver='lbfgs',
              C=1.0,
              n_jobs=-1,
              random_state=1).fit(X_train,y_train_raw)


# 모델 평가
print('학습 평가 : ',model.score(X_train,y_train_raw))
print('테스트 평가 : ',model.score(X_test,y_test_raw))



