# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_iris

# 1. 사이킷 런에서 제공하는 데이터를 로딩
iris = load_iris()

# 2. 특성(X), 라벨(y)를 분류

# 입력(특성) 데이터
# - 머신러닝에서 사용되는 변수 중,
#   대문자로 작성된 변수는 일반적으로
#   다차원 데이터를 저장하는 변수를 의미함
X = pd.DataFrame(iris.data)

# 라벨(정답) 데이터
# - 머신러닝에 의해서 예측해야하는 정답
# - 사이킷 런에서는 라벨 데이터가 1차원 배열의 
#   형태로 제공되어야만 함
# - 소문자로 작성된 변수는 스칼라, 
#   1차원의 데이터를 의미함
y = pd.Series(iris.target)

# 3. 데이터의 확인(이해)

# 입력 데이터의 샘플 개수 및 각 특성들의 타입,
# 결측 데이터의 유무를 확인
print(X.info())

# 입력 데이터의 통계 수치 확인
# - 사분위 수를 활용하여 각 특성 데이터의
#   분포를 확인하는 것이 중요함
# - 각 특성 데이터의 스케일을 확인하는 것이 중요함
print(X.describe())

# 라벨(정답) 데이터의 편향 정도 확인
# - 분류용 데이터 셋의 경우 반드시 확인
# - 회귀용 데이터 셋의 경우 X
print(y.value_counts())
print(y.value_counts() / len(y))

# 4. 학습 전 데이터의 분할
# - 학습 데이터, 테스트 데이터, 검증 데이터
# - 학습 데이터 : 머신러닝 모델이 학습할 데이터
# - 테스트 데이터 : 학습이 종료된 머신러닝 모델이
#  정답을 예측하기 위한 데이터
#  (머신러닝 모델의 일반화 정도를 판단하는 기준이 됨)
# - 검증 데이터 : 딥러닝 모델과 같이 단계별로 학습을
#   진행하는 경우 일정 단계에서 검증을 위한 목적으로
#   사용되는 데이터
#   학습데이터의 정확도와 검증데이터 정확도의 추이를
#   비교하여 학습 도중 과적합 여부를 판단
# - 학습(70%), 테스트(20%), 검증(10%)

# 일반적인 데이터의 분할 방법
# - 인덱스 정보를 기반으로 데이터를 분할하는 방법
size = X.shape[0]
print(f'입력 데이터의 전체 샘플 개수 : {size}')

X_train = X.iloc[:int(size * 0.7)]
X_test = X.iloc[int(size * 0.7):int(size * 0.9)]
X_valid = X.iloc[int(size * 0.9):]

y_train = y.iloc[:int(size * 0.7)]
y_test = y.iloc[int(size * 0.7):int(size * 0.9)]
y_valid = y.iloc[int(size * 0.9):]

print(f'{len(X_train)}, {len(X_test)}, {len(X_valid)}')
print(f'{len(y_train)}, {len(y_test)}, {len(y_valid)}')

# 순차적으로 데이터를 분할하는 경우의 문제점
# - 라벨 데이터의 편향 현상이 발생할 수 있음
print('학습 라벨 데이터 : \n', y_train)
print('테스트 라벨 데이터 : \n', y_test)
print('검증 라벨 데이터 : \n', y_valid)

# 사이킷 런의 데이터 분할을 위해서 제공되는
# train_test_split 함수
from sklearn.model_selection import train_test_split

# train_test_split 함수의 사용법
# train_test_split(X, y, 추가적인 파라메터정보...)

# 부가적인 파라메터 정보

# test_size : 테스트 데이터 셋의 비율(실수의 값 사용)
# - 0.3이 입력되는 경우, 학습데이터 70%. 
#   테스트데이터 30%가 반환
# - test_size 를 지정하지 않은 경우
#   학습데이터 75%. 테스트데이터 25%가 반환

# random_state : 난수 발생의 seed 값을 의미
# - 동일한 데이터와 동일한 random_state 정보가 대입되면
#   항상 동일한 데이터 셋이 반환되도록 보장할 수 있음
#   (다수 번의 학습 시 비교를 수월하게 진행할 수 있음)

# stratify : 데이터 셋이 분류 데이터인 경우에만 사용
# - 라벨 데이터의 비율을 전체 데이터 셋에 근사하여 반환

# train_test_split 함수의 반환 값
# X_train(학습할 입력데이터), X_test(테스트할 입력데이터), 
# y_train(학습할 라벨데이터), y_test(테스트할 라벨데이터) 
# = train_test_split(...)

# 실제 사용 예
# - pandas 데이터프레임에서 numpy 배열을 반환받는 방법
# - values 속성을 사용하여 numpy 배열을 반환받을 수 있음
# - Series 타입도 동일하게 numpy 배열을 반환함
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=0.1, random_state=1,
        stratify=y.values)

print(f'{len(X_train)}, {len(X_test)}')

print(X_train[:2])
print(y_train[:2])

print('테스트 라벨 데이터 : \n', y_test)















