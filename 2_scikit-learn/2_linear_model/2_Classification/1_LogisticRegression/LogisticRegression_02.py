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

# - LogisticRegression 클래스는 기본 제약조건으로 
# L2 정규화를 지원
# - 제약조건에 관련된 하이퍼 파라메터는 C 변수이며,
# 기본값은 1.0로 설정되어 있습니다.
# - C의 값을 높일수록 제약의 강도가 낮아지며
# (학습 데이터를 더 많이 맞출 수 있음 - 과적합시킬 수 있음)
# - C의 값은 낮출수록 제약의 강도가 높아집니다.
# (학습 데이터를 많이 맞추지 못하지만 테스트 데이터에 대한 
# 일반화 성능이 높아짐)
C = 0.01
model = LogisticRegression(C=C,
        solver='lbfgs', 
        max_iter=5000).fit(X_train, y_train)

print('훈련 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

model_001 = LogisticRegression(
        C=0.01, solver='lbfgs',
        max_iter=10000).fit(X_train, y_train)

model_1 = LogisticRegression(
        C=1.0, solver='lbfgs',
        max_iter=10000).fit(X_train, y_train)

model_100 = LogisticRegression(
        C=100.0, solver='lbfgs',
        max_iter=10000).fit(X_train, y_train)

print('훈련 평가(C=0.01) : ', 
      model_001.score(X_train, y_train))
print('훈련 평가(C=1) : ', 
      model_1.score(X_train, y_train))
print('훈련 평가(C=100) : ', 
      model_100.score(X_train, y_train))

print('테스트 평가(C=0.01) : ', 
      model_001.score(X_test, y_test))
print('테스트 평가(C=1) : ', 
      model_1.score(X_test, y_test))
print('테스트 평가(C=100) : ', 
      model_100.score(X_test, y_test))

# 제약 조건에 따른 기울기의 변화
from matplotlib import pyplot as plt

plt.figure(figsize=(10, 7))

plt.plot(model_001.coef_.T, 'v', label='C=0.01')
plt.plot(model_1.coef_.T, 'o', label='C=1.0')
plt.plot(model_100.coef_.T, '^', label='C=100.0')

plt.axhline(0, color='y', linestyle='--')

plt.legend()
plt.show()




















