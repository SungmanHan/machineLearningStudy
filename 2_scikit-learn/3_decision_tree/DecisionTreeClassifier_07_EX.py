# -*- coding: utf-8 -*-

# DecisionTreeClassifier 클래스를 사용하여 
# load_wine 데이터를 분석하고
# 정확도 및 정밀도, 재현율을 확인하세요.
# (DecisionTree의 그래프, 특성 중요도를 시각화하여 확인하세요)  

import pandas as pd
from sklearn.datasets import load_wine 

wine = load_wine()

X = pd.DataFrame(wine.data)
y = pd.Series(wine.target)

print(X.info())

# 스케일의 조정이 필요한 데이터 셋이지만
# 트리 기반의 알고리즘을 사용하는 경우
# 전처리를 하지 않아도 분석할 수 있음
pd.options.display.max_columns=13
print(X.describe())

print(y.value_counts())
print(y.value_counts() / len(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, stratify=y.values, 
        random_state=42)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(
        random_state=1).fit(X_train, y_train)

# 모델의 정확도 확인
print('학습 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

# 모델의 정밀도/재현율 확인
from sklearn.metrics import classification_report

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print('classification_report - 학습')
print(classification_report(y_train, pred_train))

print('classification_report - 테스트')
print(classification_report(y_test, pred_test))

# 특성 중요도 확인
print('특성 중요도 : \n', model.feature_importances_)

import numpy as np
from matplotlib import pyplot as plt

plt.figure(figsize=(10,7))

def plot_feature_importances(model, feature_names):
    n_features = len(feature_names)
    plt.barh(range(n_features), 
             model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("feature_importances")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)

plot_feature_importances(model, wine.feature_names)

# 그래프 확인
from sklearn.tree import export_graphviz
export_graphviz(model, out_file='./load_wine.dot',                
                feature_names=wine.feature_names, 
                filled=True)

import graphviz
from IPython.display import display

with open('./load_wine.dot', encoding='utf-8') as f:
    dot_graph = f.read()
    
display(graphviz.Source(dot_graph))




















