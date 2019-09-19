# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

X = pd.DataFrame(iris.data)
y = pd.Series(iris.target)

print(X.describe())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X.values)
X_values = scaler.transform(X.values)

from sklearn.svm import SVC
model = SVC(gamma='scale', C=1.0, random_state=1)

# 교차검증 기능을 제공하는 cross_val_score 함수
# 하이퍼 파라메터
# cross_val_score(예측기 객체, X 데이터, y 데이터, 교차검증개수)
# 반환되는 값
# - 교차검증 개수에 정의된 크기의 예측기 객체가 생성되며
#   각 예측기의 평가 점수가 반환됨
#   (회귀 모델의 경우 R2 스코어가 반환
#   분류 모델의 경우 정확도가 반환됨)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_values, y.values, 
                         cv=10, n_jobs=-1)

print('교차검증 점수 : \n', scores)
print('교차검증 평균 점수 : ', scores.mean())















