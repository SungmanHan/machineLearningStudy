# -*- coding: utf-8 -*-

# winequality-red.csv 데이터를 
# RandomForestClassifier 클래스를 이용하여 
# 분석한 후, 결과를 확인하세요.

import pandas as pd

fname='../../../data/winequality-red.csv'
wine = pd.read_csv(fname, sep=';')

X = wine.iloc[:,:-1].values
y = wine.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(
        max_depth=12, 
        random_state=1).fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

ensemble = RandomForestClassifier(
        n_estimators=10000,        
        max_features=0.2,
        random_state=1,
        n_jobs=-1).fit(X_train, y_train)

print('학습 평가(dt) : ', 
      dt_model.score(X_train, y_train))
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
























