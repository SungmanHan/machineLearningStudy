# -*- coding: utf-8 -*-

# 랜덤포레스트(Random Forest)는 의사결정나무(Decision Tree)를
# 개별 모형으로 사용하는 앙상블 방법
# (배깅(Bagging)을 적용한 의사결정나무의 앙상블 모델)

# 랜덤포레스트는 전체 입력 데이터 중 일부 열의 데이터만 선택하여 사용
# 하지만 노드 분리 시, 모든 독립 변수들을 비교하여
# 최선의 독립 변수를 선택하는 것이 아니라 독립 변수
# 차원을 랜점하게 감소시킨 다음 그 중에서 독립 변수를 선택
# - 개별 모형들 사잉의 상관관계를 감소기켜 모형 성느으이 변동을 최소하 할 수 있음

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1,stratify=y)

