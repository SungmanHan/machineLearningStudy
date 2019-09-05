# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

X = np.array([6, 8, 10, 14, 18]).reshape(-1,1)
y = np.array([7, 9, 13, 17.5, 18.7])

from sklearn.linear_model import LinearRegression

# LinearRegression 클래스는 선형회귀 알고리즘을 
# 적용하여 회귀의 결과를 예측할 수 있는 예측기 클래스
# 선형방정식
# y = X * 기울기(가중치) + 절편(편향)

# LinearRegression 클래스는 학습의 결과인
# 가중치와 절편을 coef_와 intercept_ 변수에 저장
model = LinearRegression()

# 주의사항
# 모델의 학습 전에 coef_와 intercept_ 멤버 변수는
# 생성되지 않으므로 아래의 코드는 에러가 발생됨
#print(f'기울기 : {model.coef_}')
#print(f'절편 : {model.intercept_}')

model.fit(X, y)

# LinearRegression 클래스는 입력된 X 데이터의
# 각각의 특성에 관련된 기울기 값을 계산
# - 예를들어 하나의 샘플이 3개의 특성(컬럼/열)을
# 가지는 경우 기울기 값은 3개가 생성됩니다.
print(f'기울기 : {model.coef_}')

# 절편의 값은 특성의 개수와 관계없이
# 1개만 생성됩니다.
print(f'절편 : {model.intercept_}')

# predict 메소드를 사용하여 예측하는 예제
# - predict 메소드는 입력 매개변수로 2차원을 입력
# - predict 메소드의 반환 값은 1차원
#  (입력된 X의 행의 수가 같은 1차원 배열이 반환됨)
print('머신러닝 모델의 predict 메소드를 사용하여 예측')
print(model.predict(X))

# 모델이 계산한 기울기와 절편의 값을 사용하여
# 예측하는 코드
print('기울기와 가중치를 사용하여 예측')
print(X * model.coef_ + model.intercept_)














