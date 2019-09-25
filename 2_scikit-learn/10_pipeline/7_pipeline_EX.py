# -*- coding: utf-8 -*-

# Pipeline 클래스와 GridSearchCV 클래스를 활용하여
# winequality-white.csv 파일을 분석한 후 결과를 확인하세요.

import pandas as pd

fname='../../data/winequality-white.csv'
wine=pd.read_csv(fname, sep=';')

pd.options.display.max_columns=100
print(wine.describe())

X=wine.iloc[:,:-1]
y=wine.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        stratify=y.values,
        random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

from sklearn.svm import SVC
base_model=SVC(random_state=1)

from sklearn.ensemble import BaggingClassifier
ensemble_model=BaggingClassifier(
        base_estimator=base_model,
        n_jobs=-1)

from sklearn.pipeline import Pipeline
pipe=Pipeline([('scaler',scaler),
               ('model',ensemble_model)])

from sklearn.model_selection import KFold
kfold=KFold(n_splits=5,shuffle=True,random_state=1)

from sklearn.model_selection import GridSearchCV

param_grid={'model__n_estimators':[100,300,500],
            'model__max_samples':[0.2,0.3,0.5,0.7],
            'model__max_features':[0.2,0.3,0.5,0.7],
            'model__base_estimator__C':[0.0001,0.001,0.01,0.1,1,10,100,1000],
            'model__base_estimator__gamma':[0.0001,0.001,0.01,0.1,1,10,100,1000]}

grid_model=GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=kfold,
        n_jobs=-1).fit(X_train, y_train)

print("Best 교차 검증 점수 : ", 
      grid_model.best_score_)
print("최적의 하이퍼 파라메터 : ", 
      grid_model.best_params_)

# 4. 파이프 라인을 통해서 학습된 머신러닝 모델의 평가
print("학습 평가 : ", 
      grid_model.score(X_train, y_train))
print("테스트 평가 : ", 
      grid_model.score(X_test, y_test))


















