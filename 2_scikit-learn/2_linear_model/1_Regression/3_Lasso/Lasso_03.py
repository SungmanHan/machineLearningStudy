﻿# -*- coding: utf-8 -*-

import pandas as pd

fname = '../../../../data/extended_boston.csv'
data = pd.read_csv(fname, header=None, index_col=0)

X = data.iloc[:,:-1]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X.values, y.values, 
                     test_size=0.3, random_state=1)

from sklearn.linear_model import LinearRegression, Ridge, Lasso

# 분석 모델 클래스의 객체 생성
lr_model = LinearRegression().fit(X_train, y_train)
ridge_model = Ridge(alpha=0.1).fit(X_train, y_train)

# You might want to increase the number of iterations.
# 메세지가 출력되는 경우 반복의 횟수를 증가시킬 수 있는
# max_iter 하이퍼 파라메터의 값을 증가시켜 처리할 수 있습니다.
lasso_model = Lasso(alpha=0.01, max_iter=10000).fit(X_train, y_train)

# 분석 모델 객체의 평가(R2 스코어 확인 - 결정계수)
print("LR 평가 - train : ", lr_model.score(X_train, y_train))
print("Ridge 평가 - train : ", ridge_model.score(X_train, y_train))
print("Lasso 평가 - train : ", lasso_model.score(X_train, y_train))

print("=" * 30)

print("LR 평가 - test : ", lr_model.score(X_test, y_test))
print("Ridge 평가 - test : ", ridge_model.score(X_test, y_test))
print("Lasso 평가 - test : ", lasso_model.score(X_test, y_test))

from matplotlib import pyplot as plt

plt.figure(figsize=(10,7))

coef_range = list(range(1, len(ridge_model.coef_) + 1))

plt.plot(coef_range, lr_model.coef_, 'r^')
plt.plot(coef_range, ridge_model.coef_, 'bo')
plt.plot(coef_range, lasso_model.coef_, 'gv')

plt.hlines(0, 1, len(ridge_model.coef_) + 1, 
           colors='y', linestyles='dashed')

plt.show()



















