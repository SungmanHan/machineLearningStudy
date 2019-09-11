# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42)

# 결정트리 알고리즘을 구현하고 있는 
# DecisionTreeClassifier 클래스
# 트리 구조를 사용하여 데이터를 분류, 
# 예측할 수 있는 클래스
from sklearn.tree import DecisionTreeClassifier

# 결정트리의 노드
# root node : 트리를 구성하는 최상단의 노드
# leaf node : 트리를 구성하는 최하단의 노드
# pure node : leaf node 중, 하나의 클래스로 분류된 노드

# DecisionTreeClassifier 클래스를 사용하여 모델을 생성할 때
# random_state 하이퍼 파라메터를 적용하여 동일한 특성으로
# 트리구조를 생성하도록 제어할 수 있음
# (random_state 를 지정하지 않으면 매 학습 시 마다
# 서로 다른 특성으로 트리를 생성함 - 학습 결과가 다를 수 있음)
model = DecisionTreeClassifier(
        random_state=1).fit(X_train, y_train)

print('학습 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))










