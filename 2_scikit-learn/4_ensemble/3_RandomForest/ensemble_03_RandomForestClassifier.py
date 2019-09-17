# -*- coding: utf-8 -*-

# 랜덤포레스트(Random Forest)는 의사결정나무(Decision Tree)를 
# 개별 모형으로 사용하는 앙상블 방법
# (배깅(bagging)을 적용한 의사결정나무(decision tree)의 앙상블 모델)

# 랜덤포레스트는 전체 입력 데이터 중 일부 열의 데이터만 선택하여 사용
# 하지만 노드 분리 시, 모든 독립 변수들을 비교하여 
# 최선의 독립 변수를 선택하는 것이 아니라 독립 변수 
# 차원을 랜덤하게 감소시킨 다음 그 중에서 독립 변수를 선택
# - 개별 모형들 사이의 상관관계를 감소시켜 
#   모형 성능의 변동을 최소화 할 수 있음

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(
        max_depth=5, 
        random_state=1).fit(X_train, y_train)

print('학습 평가(dt) : ', 
      dt_model.score(X_train, y_train))

from sklearn.ensemble import RandomForestClassifier

# 랜덤포레스트 모델을 생성할 때 주의해야할 하이퍼 파라메터 정보
# n_estimators : 하위 결정트리 모델의 개수를 의미
# (개수가 많을수록 성능이 향상됨)
# max_features : 실수 값을 입력 0 ~ 1 사이의 값을 입력(기본값은 auto)
# - 기본 값 auto의 의미는 모든 특성 개수의 제곱근의 값을 의미함
# - max_features 의 값을 높일 수록 하위 결정트리의 노드 분기시
#   많은 특성을 고려함
#   (대다수의 하위 결정 트리들이 비슷한 형태를 가짐 - 상관관계가 높아짐)
# - max_features 의 값을 낮게 지정할수록 하위 결정트리의 노드 분기 시
#   적은 특성을 고려함
#   (대다수의 하위 결정 트리들이 서로 다른 결과를 반환 - 상관관계가 낮아짐)
# max_depth : 각 하위 결정 트리의 최대 깊이값을 지정
# - 값을 높일수록 학습 성능이 높아짐.(테스트 데이터에 대해서 일반화 성능이 
#   낮아질 수 있음)
# - 값을 낮게 할수록 학습 성능이 낮아지고, 일반화 성능이 향상됨
#   낮아질 수 있음) 
# 일반화 성능을 올리기 위해서 사용될 수 있는 하이퍼 파라메터
# max_leaf_nodes : 트리의 마지막 노드 개수를 제한
# min_samples_split : 각 노드의 분기 시, 최소 샘플의 개수

# n_jobs : 모델에 학습에 사용되는 코어의 개수를 의미
# (-1을 지정하면 사용가능한 모든 코어를 사용)

ensemble = RandomForestClassifier(
        n_estimators=10000,
        max_features=0.2,
        random_state=1,
        n_jobs=-1).fit(X_train, y_train)

print('학습 평가(ensemble) : ', 
      ensemble.score(X_train, y_train))

print('테스트 평가(dt_model) : ', 
      dt_model.score(X_test, y_test))
print('테스트 평가(ensemble) : ', 
      ensemble.score(X_test, y_test))

# 각 독립 변수의 중요도(feature importance)를 계산
import numpy as np
from matplotlib import pyplot as plt

cancer = load_breast_cancer()
def plot_feature_importances(model):
    plt.figure(figsize=(10,7))
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), 
             model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("feature_importances")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)
    plt.show()

plot_feature_importances(dt_model)
plot_feature_importances(ensemble)
























