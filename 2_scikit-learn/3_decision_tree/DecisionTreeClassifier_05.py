# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=30, 
        random_state=1).fit(X_train, y_train)

print('학습 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

# 결정트리 기반의 예측기 클래스들은 중요 특성들에 대한
# 정보를 제공할 수 있습니다.
# (중요 특성 : 트리 구조에 사용된 각 특성)
# (데이터 분류를 가장 효율적으로 처리한 특성에 높은 값이
# 대입됨)
print('특성 중요도 : \n', model.feature_importances_)

import numpy as np
from matplotlib import pyplot as plt

plt.figure(figsize=(10,7))

cancer = load_breast_cancer()

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), 
             model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("feature_importances")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)













