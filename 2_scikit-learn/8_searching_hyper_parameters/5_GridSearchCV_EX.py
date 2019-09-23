# -*- coding: utf-8 -*-

# winequality-red.csv 데이터를 
# GridSearchCV 클래스를 활용하여 분석하세요.
# 활용할 예측기는 GradientBoostingClassifier 입니다.

import pandas as pd

fname='../../data/winequality-red.csv'
wine = pd.read_csv(fname, sep=';')

X = wine.iloc[:,:-1]
y = wine.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, 
        test_size=0.2, random_state=10)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

"""
criterion='friedman_mse', init=None,
learning_rate=0.1, loss='deviance', max_depth=3,
max_features=None, max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0, n_estimators=100,
n_iter_no_change=None, presort='auto',
random_state=None, subsample=1.0, tol=0.0001,
validation_fraction=0.1, verbose=0,
warm_start=False
"""
# 검색할 하이퍼 파라메터의 목록을 딕셔너리 변수로
# 저장(하이퍼 파라메터의 이름이 키값으로 저장됨)
param_grid = {'learning_rate':[0.001, 0.01, 0.1],
              'max_depth':[1,2,3,4,5],
              'max_features':[0.3,0.4,0.5,0.7],
              'n_estimators':[100, 500, 1000],
              'subsample':[0.3,0.5,1.0]}

# 폴드를 설정
kfold = KFold(n_splits=3, shuffle=True, random_state=1)

grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=1),
        param_grid=param_grid,
        cv=kfold,
        n_jobs=-1).fit(X_train, y_train)

print('테스트 평가 : ', 
      grid_search.score(X_test, y_test))

print('최적의 매개변수 : \n',
      grid_search.best_params_)

print('최적의 교차검증 점수 : \n',
      grid_search.best_score_)

print('최적의 하이퍼 파라메터가 적용된 모델 : \n',
      grid_search.best_estimator_)

















