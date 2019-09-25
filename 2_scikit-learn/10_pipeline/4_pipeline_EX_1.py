# -*- coding: utf-8 -*-

# 일반적인 머신러닝 단계
# - 파이프 라인을 사용한 데이터의 전처리 과정 및 
#  머신러닝 모델의 학습 과정 자동화

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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scaler',MinMaxScaler()),
                 ('poly',PolynomialFeatures(
                         degree=2, include_bias=False)),
                 ('lr_model',
                  LogisticRegression(
                          solver='lbfgs',
                          C=0.7,
                          n_jobs=-1,
                          random_state=1))])

# 3. 파이프 라인의 실행을 통한 데이터 전처리 및
#   머신러닝 모델의 학습을 진행
pipe.fit(X_train, y_train)

# 4. 파이프 라인을 통해서 학습된 머신러닝 모델의 평가
print("학습 평가 : ", 
      pipe.score(X_train, y_train))
print("테스트 평가 : ", 
      pipe.score(X_test, y_test))

from sklearn.metrics import classification_report

pred_train = pipe.predict(X_train)
pred_test = pipe.predict(X_test)

print('학습 데이터의 정밀도, 재현율, F1')
print(classification_report(y_train, pred_train))

print('테스트 데이터의 정밀도, 재현율, F1')
print(classification_report(y_test, pred_test))










