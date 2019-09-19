# -*- coding: utf-8 -*-

# winequality-white.csv 데이터를 
# 교차 검증을 사용하여 SVC, LogisticRegression,
# GradientBoostingClassifier 모델의
# 결과를 확인하세요. 

import pandas as pd

fname='../../data/winequality-white.csv'
wine=pd.read_csv(fname, sep=';')

X = wine.iloc[:,:-1]
y = wine.iloc[:, -1]

print(X.describe())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X.values)

X_not_scaled = X.values
X_scaled = scaler.transform(X.values)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

model_svc = SVC(C=1.0, gamma='scale', random_state=1)
scores_svc = cross_val_score(
        model_svc, X_scaled, y.values, 
        cv=5, n_jobs=-1)
print('교차검증 점수(SVC) : \n', scores_svc)
print('교차검증 평균 점수(SVC) : ', scores_svc.mean())

model_lr = LogisticRegression(solver='lbfgs', 
        C=1.0, multi_class='multinomial', 
        random_state=1, n_jobs=-1)
scores_lr = cross_val_score(
        model_lr, X_scaled, y.values, 
        cv=5, n_jobs=-1)
print('교차검증 점수(LR) : \n', scores_lr)
print('교차검증 평균 점수(LR) : ', scores_lr.mean())

model_gb = GradientBoostingClassifier(
        n_estimators=10000,
        subsample=0.3,
        max_depth=5,
        max_features=0.5,
        random_state=1)
scores_gb = cross_val_score(
        model_gb, X_not_scaled, y.values, 
        cv=5, n_jobs=-1)
print('교차검증 점수(GB) : \n', scores_gb)
print('교차검증 평균 점수(GB) : ', scores_gb.mean())















