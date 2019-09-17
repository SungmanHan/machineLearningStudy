# -*- coding: utf-8 -*-

# SVM(Support Vector Machine)은 러시아 과학자 Vladimir Vapnik이
# 1970년대 후반에 제안한 알고리즘
# 분류(classification)문제에서 우수한 일반화(generalization) 
# 성능이 입증되어 머신러닝 알고리즘에서 범용적으로 활용
# SVM 기반의 머신러닝 알고리즘들은 일반화 성능이 다른 분류 모델과 
# 비교할 때 더 좋거나 대등한 것으로 알려져 있다.

# SVM 기반의 머신러닝 알고리즘들은 선형 또는 비선형 분류 뿐만아니라 
# 회귀, 이상치 탐색에도 사용할 수 있음
# 복잡한 분류 문제에도 성능이 우수하며, 중간 크기의 데이터셋에 적합함

# SVM 기반의 머신러닝 알고리즘들은 데이터를 분류하기 위한
# 최대 여백(margin - 마진)의 값을 찾는 방법을 제공함
# 여백(마진)을 최대화하는 결정 초평면(decision hyperplane)을 찾는 것이 
# SVM 기반 모델들의 목표

# 사이킷 런에서는 LinearSVC, SVC, SVR 클래스 등이 제공됨

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42)

# SVM 알고리즘을 기반으로 하는 예측기들은
# 특성 데이터 스케일에 많은 영향을 받습니다.
# 만약 특성 데이터의 스케일이 서로 다른 영역에 위치한 경우
# 공간의 분할이 어려워지므로 학습이 올바르게 진행되지 않습니다.
# SVM 알고리즘 기반의 예측기를 사용하는 경우
# 반드시 데이터의 전처리를 수행해야만 올바르게 학습이 이뤄집니다.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC

model = SVC(gamma='scale').fit(X_train, y_train)

print('학습 평가 : ', 
      model.score(X_train, y_train))

print('테스트 평가 : ', 
      model.score(X_test, y_test))
































