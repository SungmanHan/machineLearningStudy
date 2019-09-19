# -*- coding: utf-8 -*-

# diabetes.csv 데이터를 교차 검증을 사용하여 
# SVC, LogisticRegression, RandomForestClassifier 
# 모델의 결과를 확인하세요. 
# (KFOLD를 사용하여 교차검증을 수행하세요)

import pandas as pd

fname='../../data/diabetes.csv'
diabetes = pd.read_csv(fname, header=None)

X = diabetes.iloc[:,:-1]
y = diabetes.iloc[:, -1]

pd.options.display.max_columns=100
print(X.describe())

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

model_lr = LogisticRegression(
        solver='lbfgs', C=1.0, n_jobs=-1, random_state=1)
scores = cross_val_score(model_lr, X.values, y.values,
                         cv=kfold, n_jobs=-1)

print('교차검증 점수(LR) : \n', scores)
print('교차검증 평균 점수(LR) : ', scores.mean())

model_svc = SVC(gamma='scale', C=1.0, random_state=1)
scores = cross_val_score(model_svc, X.values, y.values,
                         cv=kfold, n_jobs=-1)

print('교차검증 점수(SVC) : \n', scores)
print('교차검증 평균 점수(SVC) : ', scores.mean())

model_rf = RandomForestClassifier(
        n_estimators=10000,        
        max_features=0.3,
        n_jobs=-1,
        random_state=1)
scores = cross_val_score(model_rf, X.values, y.values,
                         cv=kfold, n_jobs=-1)

print('교차검증 점수(RF) : \n', scores)
print('교차검증 평균 점수(RF) : ', scores.mean())

























