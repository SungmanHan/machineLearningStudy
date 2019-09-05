# -*- coding: utf-8 -*-

import pandas as pd

fname = '../../../../data/score.csv'
scores = pd.read_csv(fname)

X = scores.iloc[:,2:]
y = scores.score

X_data = X.values
y_data = y.values

from sklearn.linear_model import LinearRegression
lr_model = LinearRegression().fit(X_data, y_data)
print('LR평가(R2) : ', lr_model.score(X_data, y_data))

# 선형 모델에 L2 제약 조건을 추가한 Ridge 클래스
# L2 제약 조건 : 모든 특성에 대한 가중치(기울기)의 값을
# 0 주변으로 위치하도록 제어하는 제약조건
# LinearRegression 클래스는 학습 데이터에 최적화되도록
# 학습을 하기때문에 테스트 데이터에 대한 일반화 성능이 감소됩니다.
# 이러한 경우 모든 특정 데이터를 적절히 활용할 수 있도록
# L2 제약 조건을 사용할 수 있으며, L2 제약조건으로 인하여
# 모델의 일반화 성능(테스트 데이터의 성능)이 증가하게 됩니다.
from sklearn.linear_model import Ridge

# Ridge 클래스의 하이퍼 파라메터 alpha
# alpha의 값이 커질수록 제약을 크게 설정
# (alpha의 값이 커질수돌 모든 특성들의 가중치의 값은 
# 0 주변으로 위치함)
# alpha의 값이 작아질수록 제약이 약해짐
# (alpha의 값이 작아질수록 모든 특성들의 가중치의 값은 
# 0에서 멀어짐)
# alpha의 값이 작아질수록 LinearRegression 클래스와 동일해짐
ridge_model = Ridge(alpha=100).fit(X_data, y_data)
print('Ridge평가(R2) : ', ridge_model.score(X_data, y_data))

import numpy as np
from matplotlib import pyplot as plt

# X데이터의 특성(컬럼) 개수 확인
print(X_data.shape[1])
# LinearRegression 객체의 가중치 변수 확인
print(lr_model.coef_)

# X축의 데이터
coef_range = np.arange(1, lr_model.coef_.shape[0] + 1)

plt.plot(coef_range, lr_model.coef_, 'ro')
plt.plot(coef_range, ridge_model.coef_, 'b^')

plt.axhline(0, color='y', linestyle='--')
plt.show()
























