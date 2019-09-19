# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_breast_cancer

# 머신러닝을 사용해서 데이터를 분석하는 과정

# 1. 데이터 셋의 로딩
# - 파일로부터 로딩
# - 임의의 값을 생성
# - 사전에 준비된 데이터를 사용
X, y = load_breast_cancer(return_X_y=True)

# 2. 데이터의 분할
# - 훈련 및 테스트 셋으로 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=100)

# 3. 데이터의 전처리
# - 데이터의 형태를 확인한 후, 필요한 경우
# 전처리 과정을 수행
pd.options.display.max_columns=100
X_df = pd.DataFrame(X)
print(X_df.info())
print(X_df.describe())

# - 스케일 조정 작업 수행
# - 스케일의 조정은 학습데이터를 
# 기준으로 처리해야합니다.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)

# 주의사항 
# 테스트 데이터의 경우 fit 메소드의 호출 없이
# 변환과정만 수행해야 합니다.
# (학습 데이터를 기준으로 변환)
X_test = scaler.transform(X_test)

# 4. 머신러닝 알고리즘을 사용하여 데이터를 학습
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
        solver='lbfgs', 
        random_state=1,
        n_jobs=-1).fit(X_train, y_train)

# 5. 머신러닝 모델의 평가
# - 모델의 평가에는 테스트 세트를 사용
print('학습 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))
















