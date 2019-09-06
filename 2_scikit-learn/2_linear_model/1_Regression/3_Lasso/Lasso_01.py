# -*- coding: utf-8 -*-

import pandas as pd

fname = '../../../../data/score.csv'
scores = pd.read_csv(fname)

# pandas DataFrame에서 특정 행, 열을 삭제할 수 있는 drop 메소드
# drop(삭제할 열/행의 이름, 인덱스)
# - axis 매개변수의 값을 1로 지정하는 경우 열을 삭제
# - axis 매개변수를 지정하지 않는 경우 행을 삭제
# - drop 메소드의 사용 시, inplace=True 매개변수를 지정하지 않는 경우
#   삭제된 결과를 반환(실제 데이터에서는 삭제되지 않음)
# - inplace=True 매개변수를 전달하는 경우, 실제 원본 데이터에서
#   특정 열/행을 삭제하고 아무것도 반환하지 않음
scores.drop('name', axis=1, inplace=True)

X = scores.iloc[:, 1:]
y = scores.iloc[:, 0]

X_train = X.values
y_train = y.values

# 선형 모델에 L1 제약 조건을 추가한 Lasso 클래스
# L1 제약 조건 : 모든 특성 데이터 중 특정 특성에 
# 대해서만 가중치의 값을 할당하는 제약조건
# (대다수 특성의 가중치 값은 0으로 제약)
# L1 제약 조건은 특성 데이터가 많은 데이터를 학습하는 경우 
# 빠르게 학습을 할 수 있는 장점을 가짐
# 모든 특성 데이터 중 중요도가 높은 특성을 구분할 수 있음
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# 분석 모델 클래스의 객체 생성
lr_model = LinearRegression().fit(X_train, y_train)
ridge_model = Ridge(alpha=100).fit(X_train, y_train)
# Lasso 클래스의 하이퍼 파라메터 alpha
# alpha의 값이 커질수록 제약을 크게 설정
# (alpha의 값이 커질수돌 대다수의 특성에 대한 가중치의 값이 0으로 수렴)
# alpha의 값이 작아질수록 제약이 약해짐
# (alpha의 값이 작아질수록 적은 수의 특성에 대한 가중치의 값은 0으로 수혐)
# alpha의 값이 작아질수록 LinearRegression 클래스와 동일해짐
lasso_model = Lasso(alpha=10).fit(X_train, y_train)


# 분석 모델 객체의 평가(R2 스코어 확인 - 결정계수)
print("LR 평가 : ", lr_model.score(X_train, y_train))
print("Ridge 평가 : ", ridge_model.score(X_train, y_train))
print("Lasso 평가 : ", lasso_model.score(X_train, y_train))

from matplotlib import pyplot as plt

coef_range = list(range(1, len(ridge_model.coef_) + 1))

plt.plot(coef_range, lr_model.coef_, 'r^')
plt.plot(coef_range, ridge_model.coef_, 'bo')
plt.plot(coef_range, lasso_model.coef_, 'gv')

plt.hlines(0, 1, len(ridge_model.coef_) + 1, 
           colors='y', linestyles='dashed')

plt.show()



















