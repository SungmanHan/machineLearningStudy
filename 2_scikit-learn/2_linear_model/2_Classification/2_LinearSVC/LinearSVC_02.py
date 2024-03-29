# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data)
y = pd.Series(cancer.target)

pd.options.display.max_columns=100
print(X.describe())

print(y.value_counts() / len(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        stratify=y.values, random_state=1)

print(X_train.shape[0], X_test.shape[0])

# 정규화를 통한 X 데이터의 스케일 조정
# 최대최소 정규화를 실행하는 예제
# 모든 특성 데이터는 0 ~ 1 사이로 값이 변환
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import LinearSVC

# LinearSVC 클래스는 기본 제약조건으로 L2 정규화를 지원
# 제약조건에 관련된 하이퍼 파라메터는 C 변수이며,
# 기본값은 1로 설정되어 있습니다.
# C의 값을 높일수록 제약의 강도가 낮아지며
# (학습 데이터를 더 많이 맞출 수 있음 - 과적합시킬 수 있음)
# C의 값은 낮출수록 제약의 강도가 높아집니다.
# (학습 데이터를 많이 맞추지 못하지만 테스트 데이터에 
# 대한 일반화 성능이 높아짐)
svm001_model = LinearSVC(
        C=0.01, max_iter=10000).fit(X_train, y_train)
svm_model = LinearSVC(
        C=1.0, max_iter=10000).fit(X_train, y_train)
svm100_model = LinearSVC(
        C=100.0, max_iter=10000).fit(X_train, y_train)

print('훈련 평가(SVM001) : ', 
      svm001_model.score(X_train, y_train))
print('훈련 평가(SVM) : ', 
      svm_model.score(X_train, y_train))
print('훈련 평가(SVM100) : ', 
      svm100_model.score(X_train, y_train))

print('테스트 평가(SVM001) : ', 
      svm001_model.score(X_test, y_test))
print('테스트 평가(SVM) : ', 
      svm_model.score(X_test, y_test))
print('테스트 평가(SVM100) : ', 
      svm100_model.score(X_test, y_test))

from matplotlib import pyplot as plt

plt.figure(figsize=(10, 7))

plt.plot(svm001_model.coef_.T, 'v', label='C=0.01')
plt.plot(svm_model.coef_.T, 'v', label='C=1.0')
plt.plot(svm100_model.coef_.T, 'v', label='C=100.0')

plt.axhline(0, color='y', linestyle='--')

plt.legend()
plt.show()




















