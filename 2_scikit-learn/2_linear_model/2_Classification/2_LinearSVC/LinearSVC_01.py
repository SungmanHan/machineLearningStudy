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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 선형모델을 사용하여 데이터를 분류할 수 있는 
# LogisticRegression 클래스
from sklearn.linear_model import LogisticRegression
# 선형 Support Vector Machine 알고리즘을 구현하고 있는
# LinearSVC 클래스
# (Linear Support Vector Classification)
from sklearn.svm import LinearSVC

lr_model = LogisticRegression(
        solver='lbfgs').fit(X_train, y_train)

# Linear Support Vector Machine 알고리즘으로 구현된 분류 클래스
# Support Vector Machine을 구현하고 있는 SVC에 비해서
# 선형 계산에 특화되어 있어 선형 데이터를 분류하는 경우 더 효율적
# (LinearSVC 클래스의 학습 이후, the number of iterations 메세지가
# 출력되는 경우 학습이 완료되지 않은 상태이므로 max_iter 매개변수를
# 조정하여 성능을 높일 수 있습니다.)
# - 대다수의 케이스는 데이터 전처리를 통해서 해결
svm_model = LinearSVC(
        random_state=1).fit(X_train, y_train)

print('훈련 평가(LR) : ', 
      lr_model.score(X_train, y_train))
print('훈련 평가(SVM) : ', 
      svm_model.score(X_train, y_train))

print('테스트 평가(LR) : ', 
      lr_model.score(X_test, y_test))
print('테스트 평가(SVM) : ', 
      svm_model.score(X_test, y_test))












