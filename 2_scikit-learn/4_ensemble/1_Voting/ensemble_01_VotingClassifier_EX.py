# -*- coding: utf-8 -*-

# winequality-red.csv 데이터를 LogisticRegression, 
# KNeighborsClassifier, DecisionTreeClassifier를 
# 조합한 VotingClassifier로 분석한 후, 
# 결과를 확인하세요.

import pandas as pd

fname = '../../../data/winequality-red.csv'
wine = pd.read_csv(fname, sep=';')

X = wine.iloc[:,:-1].values
y = wine.quality.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1)

# 데이터 분석을 위한 개별 머신러닝 모델의 생성과 학습
from sklearn.neighbors import KNeighborsClassifier
kn_model = KNeighborsClassifier(
        n_neighbors=5).fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(
        solver='lbfgs', multi_class='multinomial',
        max_iter=10000).fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(
        max_depth=7, 
        random_state=1).fit(X_train, y_train)

print('학습 평가(kn) : ', 
      kn_model.score(X_train, y_train))
print('학습 평가(lr) : ', 
      lr_model.score(X_train, y_train))
print('학습 평가(dt) : ', 
      dt_model.score(X_train, y_train))

# 앙상블 모형 객체 생성 및 학습
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(
        estimators=[('kn', kn_model),
                    ('lr', lr_model),
                    ('dt', dt_model)],
                    n_jobs=-1).fit(X_train, y_train)

print('학습 평가(ensemble) : ', 
      ensemble.score(X_train, y_train))

print('테스트 평가(kn) : ', 
      kn_model.score(X_test, y_test))
print('테스트 평가(lr) : ', 
      lr_model.score(X_test, y_test))
print('테스트 평가(dt) : ', 
      dt_model.score(X_test, y_test))
print('테스트 평가(ensemble) : ', 
      ensemble.score(X_test, y_test))
