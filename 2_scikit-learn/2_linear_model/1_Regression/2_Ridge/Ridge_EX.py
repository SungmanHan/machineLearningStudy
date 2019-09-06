# -*- coding: utf-8 -*-

# 사이킷 런의 load_diabetes 함수를 사용하여 
# 당뇨병 수치를 예측할 수 있는 모델을
# 작성한 후 테스트하세요.(LinearRegression, Ridge 클래스를 활용)
# - Ridge 클래스의 alpha 값을 조절하여 값의 변화를 확인하세요

import pandas as pd
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

X = pd.DataFrame(diabetes.data)
y = pd.Series(diabetes.target)

print(X.info())
pd.options.display.max_columns=130
print(X.describe())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, random_state=1)

from sklearn.linear_model import LinearRegression, Ridge

lr_model = LinearRegression().fit(X_train, y_train)

alpha=0.5
ridge_model = Ridge(alpha=alpha).fit(X_train, y_train)

# 선형 모델의 경우 데이터의 특성의 개수가 많아질수록
# 학습의 성능이 높아집니다.
# 위의 데이터 셋인 extended_boston.csv 은
# load_boston 데이터 셋을 확장시킨 것으로 
# load_boston 데이터 셋을 학습할 때보다
# 학습의 결과가 높아지는 것을 확인할 수 있습니다.
print('LR 학습 결과 : ', lr_model.score(X_train, y_train))
print('Ridge 학습 결과 : ', ridge_model.score(X_train, y_train))

# LinearRegression 모델은 일반적으로 오버피팅되기 때문에
# 학습 성능에 비해서 테스트 데이터를 많이 맞추지 못합니다.
# 반면 일반화 성능을 높일 수 있는 Ridge 모델은
# L2 제약으로 인하여 학습 성능은 떨어지지만 테스트 데이터에
# 대해서 성능이 높아지는 것을 확인할 수 있습니다.
print('LR 테스트 결과 : ', lr_model.score(X_test, y_test))
print('Ridge 테스트 결과 : ', ridge_model.score(X_test, y_test))

import numpy as np
from matplotlib import pyplot as plt

coef_range = np.arange(1, X_train.shape[1] + 1)

plt.plot(coef_range, lr_model.coef_, 'ro')
plt.plot(coef_range, ridge_model.coef_, 'bx')

plt.axhline(0, color='y', linestyle='--')
plt.show()






