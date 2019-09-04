# -*- coding: utf-8 -*-

# data 디렉토리에 저장된 diabetes.csv 파일의 데이터를
# KNN 알고리즘을 사용하여 분석한 후, 
# 정확도와 각 클래스 별 확률을 출력하세요.

import pandas as pd

fname='../../data/diabetes.csv'
# csv 파일의 첫번째 행으로 각 컬럼의 제목(헤더)이 
# 존재하지 않는 경우
# 1번째 행의 데이터가 제목으로 지정되기 때문에
# 아래와 같이 header 정보를 None으로 지정합니다.
diabetes = pd.read_csv(fname, header=None)
print(diabetes)

# pandas 옵션을 사용하여 생략되는 컬럼의 정보를
# 확인할 수 있음(출력할 최대 컬럼의 개수를 조절)
pd.options.display.max_columns=100

# 데이터 확인 (앞의 5개)
print(diabetes.head())

# 데이터 확인 (뒤의 5개)
print(diabetes.tail())

# 특성(입력) 데이터 추출
X = diabetes.iloc[:,:-1]

# 라벨(정답) 데이터 추출
y = diabetes.iloc[:, -1]

# 특성 데이터의 확인
# 1. 전체 데이터 개수, 특성(컬럼)의 개수
#    결측 데이터의 유무 확인
print(X.info())

# 2. 특성 데이터의 스케일 확인
print(X.describe())

# 3. 특성 데이터의 분포도 확인
from matplotlib import pyplot as plt
X.hist()
plt.show()

# 라벨 데이터의 확인
# - 정답의 편향(분포)을 확인
print(y.value_counts())
print(y.value_counts() / len(y))

# 데이터 분할(학습/테스트)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=0.2, stratify=y.values,
        random_state=1)

# 학습 및 테스트 데이터의 분할된 개수 확인
print('학습 데이터 개수 : ', len(X_train))
print('테스트 데이터 개수 : ', X_test.shape[0])

# 데이터 분석을 위해 사용할 머신러닝 모델의 생성 및 학습
from sklearn.neighbors import KNeighborsClassifier

# 머신러닝 모델의 하이퍼 파라메터 정보를 변수로 저장
K=9

# 모델 객체의 생성과 학습
# n_jobs : 최근접 이웃을 검색할 때, 사용가능한
# 모든 프로세서를 활용하기 위한 하이퍼 파라메터
# (-1 설정하는 경우 모든 프로세서를 사용)
#model = KNeighborsClassifier(n_neighbors=K, n_jobs=-1)
#model.fit(X_train, y_train)

# 머신러닝 모델 객체의 생성과 학습을 동시에 진행할 수 있음
# (사이킷 런의 모든 예측기 클래스의 fit 메소드는
# X 데이터의 형태를 2차원으로 
# y 데이터의 형태를 1차원으로 간주함)
model = KNeighborsClassifier(
        n_neighbors=K, n_jobs=-1).fit(X_train, y_train)

# 머신러닝 모델의 평가
# score 메소드
# - 분류 예측기 : 정확도를 반환
# - 회귀 예측기 : R2Score(결정계수)를 반환
print('학습 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

# 사이킷 런의 평가 함수를 사용한 정확도 확인
# - 예측 값에 대한 정확도 확인
# - accuracy_score(실제정답, 예측값)
from sklearn.metrics import accuracy_score

# 학습 데이터의 예측 결과를 사용하여 정확도 확인
pred_train = model.predict(X_train)
print('학습 데이터에 대한 정확도 : ',
      accuracy_score(y_train, pred_train))

# 테스트 데이터의 예측 결과를 사용하여 정확도 확인
pred_test = model.predict(X_test)
print('테스트 데이터에 대한 정확도 : ',
      accuracy_score(y_test, pred_test))





























