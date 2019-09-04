# -*- coding: utf-8 -*-

import pandas as pd

# 사이킷 런에서 제공하는 유방암 데이터 셋
# 음성/양성 데이터로 분류된 데이터 셋을 제공(0/1)
# 이진 분류를 위한 데이터 셋
from sklearn.datasets import load_breast_cancer

# 데이터 로딩
cancer = load_breast_cancer()

# 입력 데이터(특성) 로딩
X = pd.DataFrame(cancer.data)

# 라벨 데이터(정답) 로딩
y = pd.Series(cancer.target)

# 입력 데이터 확인
# - 특성 개수 및 결측 데이터 유무 확인
print(X.info())

# - 각 특성 데이터의 통계 정보 확인
# - 스케일 정보를 확인
pd.options.display.max_columns=30
print(X.describe())

# 라벨 데이터의 분포 확인(편향정보)
# - 분류 데이터 셋인 경우 반드시 확인해야함
print(y.value_counts() / len(y))

# 1. 데이터 분할
# - 학습/테스트 데이터 분할
# - 분류를 위한 데이터 셋의 경우 반드시
#   stratify 매개변수를 선언하여
#   학습과 테스트 데이터에 전체 라벨 데이터와 동일한 비율로
#   라벨 데이터가 배치될 수 있도록 제어
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=0.3, stratify=y.values,
        random_state=1)

# 데이터 분할 확인
print(X_train.shape[0], X_test.shape[0])

# 최근접 이웃 알고리즘을 구현하고 있는 예측기 클래스의 로딩
# 예측기(Estimator) 클래스 : 사이킷 런에서 제공하는
# 데이터 학습 및 예측을 위한 알고리즘이 구현된 클래스를 의미

# KNeighborsClassifier 클래스 : K-최근접 이웃 알고리즘을
# 구현한 클래스
# - 학습이 빠른 예측기 클래스
#   (학습 데이터를 단순히 저장하기 때문에 속도가 빠름)
# - 예측할 데이터가 입력되면 사전에 저장된 학습 데이터와
#   거리를 비교하여 분류를 수행할 수 있는 예측기
#   (학습 속도는 빠른 반면, 예측 속도는 상대적으로 느림)
from sklearn.neighbors import KNeighborsClassifier

# 예측기 클래스의 객체(모델)을 생성
# 사이킷 런의 모든 예측기 클래스들은
# 구현하고 있는 알고리즘에 따라서 
# 서로 다른 하이퍼 파라메터를 가짐
# 하이퍼 파라메터 : 예측기 클래스의 객체(모델)를 생성할 때
# 사용자가 지정하는 파라메터를 의미
# (하이퍼 파라메터에 값에 따라서 모델의 성능이 변화됨)
# (사이킷 런을 사용하여 머신러닝을 수행할 때 
# 적절한 하이퍼 파라메터를 찾는것이 최종적인 목표가 됨)

# KNeighborsClassifier 예측기의 중요 하이퍼 파라메터
# n_neighbors : 예측할 데이터(테스트 데이터)에서 가장 가까운
# 주변 이웃의 개수(모델이 검색하는 이웃의 개수)
# - 분류를 위한 경우 홀수의 값을 지정하여 다수결이 항상
#   이뤄질 수 있도록 함
# n_jobs : 모델을 사용하여 학습 및 예측할 때
# 현재 머신(컴퓨터)에서 사용가능한 모든 CPU를 사용할 수 
# 있는 하이퍼 파라메터 
# (-1 로 지정하는 사용가능한 모든 CPU를 사용하여 분석)
K=9
model = KNeighborsClassifier(n_neighbors=K, n_jobs=-1)
print(model)

# 예측기 객체(모델)의 데이터 학습을 위한 메소드
# 사이킷 런의 모든 예측기 클래스들은 데이터 학습을 위한
# fit 메소드가 제공
# - 모델변수명.fit(입력데이터(X), 라벨데이터(y))의 
#   형태로 사용
# 주의사항
# - X(입력데이터)는 반드시 2차원으로 입력해야함
# - y(라벨데이터)는 반드시 1차원으로 입력해야함
model.fit(X_train, y_train)

# 예측기 객체(모델)의 성능 평가
# 사이킷 런의 모든 예측기 클래스들은 학습된 이후, 
# 성능 평가를 위한 score 메소드를 제공
# score 메소드는 예측기의 분류에 따라서 서로 다른 값을 반환
# 분류를 위한 예측기(Classifier) : 정확도를 반환
# 회귀를 위한 예측기(Regressor) : R2 score(결정계수)를 반환
accuracy = model.score(X_train, y_train)
print('학습 평가 : ', accuracy)

accuracy = model.score(X_test, y_test)
print('테스트 평가 : ', accuracy)

# 예측기 객체(모델)를 사용하여 예측 값을 반환받는 방법
# 사이킷 런의 모든 예측기 클래스들은 입력된 데이터 대한
# 예측값을 반환할 수 있는 predict 메소드가 제공됨
# - predict 메소드는 fit 메소드와 마찬가지로
#  반드시 2차원 데이터를 입력해야함
# - predict 메소드는 입력된 데이터에 대한 예측값을
#  1차원의 형태로 반환
predicted = model.predict(X_test[:5])
print('예측 : \n', predicted)
print('정답 : \n', y_test[:5])

# 예측기 객체(모델)을 사용한 예측 확률값 반환 방법
# 사이킷 런의 모든 분류를 위한 예측기 클래스들은
# 예측 확률 값을 반환할 수 있는 
# predict_proba 메소드를 제공
# predict_proba 메소드는 분류할 각 클래스 별
# 확률 값을 반환함
# (predict 메소드는 predict_proba 메소드의 결과에서
# 가장 높은 확률 값을 갖는 클래스의 값을 반환하는 메소드)
predicted_proba = model.predict_proba(X_test[:5])
print('예측 확률: \n', predicted_proba)
print('정답 : \n', y_test[:5])














