# -*- coding: utf-8 -*-

# 일반적인 머신러닝 단계
# - 파이프 라인을 사용한 데이터의 전처리 과정 및 
#  머신러닝 모델의 학습 과정 자동화
# - 하이퍼 파라메터를 검색(올바른 방식의 교차 검증을 수행)

# 1. 데이터의 적재 및 분할
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# - 데이터의 적재
cancer = load_breast_cancer()

# - 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target,
        stratify=cancer.target,
        random_state=1)

# 2. 데이터의 전처리 및 머신러닝 모델 학습을 위한
# 파이프 라인 객체의 생성 및 학습
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scaler',MinMaxScaler()),
                 ('svm_model', SVC())])

# 3. 하이퍼 파라메터 검색을 통한 최적의 모델 생성
# - 파이프 라인을 사용하여 데이터 전처리 및 
#   머신러닝 모델의 하이퍼 파라메터 검색
from sklearn.model_selection import KFold, GridSearchCV

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

param_grid = {'svm_model__C':[0.0001,0.001,0.01,0.1,1,10,100,1000],
              'svm_model__gamma':[0.0001,0.001,0.01,0.1,1,10,100,1000]}

grid_model = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=kfold,
        n_jobs=-1)

grid_model.fit(X_train, y_train)

print("Best 교차 검증 점수 : ", grid_model.best_score_)
print("최적의 하이퍼 파라메터 : ", grid_model.best_params_)

# 4. 파이프 라인을 통해서 학습된 머신러닝 모델의 평가
print("학습 평가 : ", 
      grid_model.score(X_train, y_train))
print("테스트 평가 : ", 
      grid_model.score(X_test, y_test))

from sklearn.metrics import classification_report

pred_train = grid_model.predict(X_train)
pred_test = grid_model.predict(X_test)

print('학습 데이터의 정밀도, 재현율, F1')
print(classification_report(y_train, pred_train))

print('테스트 데이터의 정밀도, 재현율, F1')
print(classification_report(y_test, pred_test))










