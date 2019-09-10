# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data)
y = pd.Series(cancer.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=0.3, stratify=y.values,
        random_state=42)

from sklearn.linear_model import LogisticRegression

C = 1.0
model = LogisticRegression(C=C,
        solver='lbfgs', 
        max_iter=5000).fit(X_train, y_train)

# 분류를 위한 예측기 클래스의 모델 평가 방법

# 1. 정확도 - Accuracy
# - 분류 모델인 경우 score 메소드를 사용하여
# 정확도의 값을 반환받을 수 있음
# - 전체 데이터 중 맞춘 비율(각각의 클래스의 정보는 무시)
print('훈련 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

# 라벨 데이터의 편향 정보를 확인
print(y.value_counts() / len(y))

# 라벨 데이터의 편향이 발생된 경우
# 정확도의 신뢰성이 떨어질 수 있음

# 정밀도(precision) : 특정 클래스로 예측한 결과에서 
# 실제 정답의 비율
# EX) 1로 예측한 전체 개수 100개 중 85개가 정답이었을 경우
# 정밀도는 85%

# 재현율(recall) : 특정 클래스의 전체 개수 중 실제로 
# 예측할 비율
# EX) 라벨이 1인 데이터의 개수가 100개, 
# 실제 정답으로 예측한 개수 87개인 경우 재현율은 87%

from sklearn.metrics import confusion_matrix
# 학습 데이터에 대한 예측 결과
pred_train = model.predict(X_train)
# 테스트 데이터에 대한 예측 결과
pred_test = model.predict(X_test)

print('confusion_matrix - 학습')
print(confusion_matrix(y_train, pred_train))

print('confusion_matrix - 테스트')
print(confusion_matrix(y_test, pred_test))

# 정밀도 확인
# precision_score : 정밀도의 값을 반환하는 함수
from sklearn.metrics import precision_score

"""
[[141   7]
 [  5 245]]
"""
# 클래스 0의 정밀도
# 141 / (141+5)
print(precision_score(y_train, pred_train, pos_label=0))

# 클래스 1의 정밀도
# 245 / (245+7)
print(precision_score(y_train, pred_train, pos_label=1))

# 재현율 확인
# recall_score : 재현율의 값을 반환하는 함수
from sklearn.metrics import recall_score
"""
[[141   7]
 [  5 245]]
"""
# 클래스 0의 재현률
# 141 / (141+7)
print(recall_score(y_train, pred_train, pos_label=0))

# 클래스 1의 재현률
# 245 / (245+5)
print(recall_score(y_train, pred_train, pos_label=1))

from sklearn.metrics import classification_report

print('classification_report - 학습')
print(classification_report(y_train, pred_train))

print('classification_report - 테스트')
print(classification_report(y_test, pred_test))




















