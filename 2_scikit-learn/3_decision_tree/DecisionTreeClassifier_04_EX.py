# -*- coding: utf-8 -*-

# Data 디렉토리에 저장된 winequality-white.csv 파일의 데이터를
# DecisionTreeClassifier 를 사용하여 분석한 후
# 정확도 및 정밀도, 재현율을 출력하세요.
# (그래프의 정보를 시각화하여 확인하세요.)

import pandas as pd

fname = '../../data/winequality-white.csv'

# 만약 데이터를 구분하고 있는 구분문자가
# , 가 아닌 경우, sep 매개변수를 사용하여
# 구분문자를 지정할 수 있음
wine = pd.read_csv(fname, sep=';')

print(wine)

pd.options.display.max_columns = 100
print(wine.info())

X = wine.iloc[:,:-1]
y = wine.iloc[:, -1]

print(X.info())
print(X.describe())

print(y.value_counts())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, stratify=y.values, 
        random_state=42)

print(X_train.shape[0], X_test.shape[0])

# X 데이터의 스케일 조정
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

lr_model = LogisticRegression(
        solver='lbfgs', max_iter=10000,
        multi_class='multinomial').fit(X_train_scaled, y_train)

ls_model = LinearSVC(
        max_iter=10000).fit(X_train_scaled, y_train)

# 결정트리 알고리즘은 데이터 분할을 처리할 때
# 하나의 특성만을 고려하기 때문에 스케일의 조정이 필요없음
dt_model = DecisionTreeClassifier(
        max_depth=10,
        random_state=1).fit(X_train, y_train)

print('학습 평가(LR) : ', 
      lr_model.score(X_train_scaled, y_train))
print('학습 평가(LS) : ', 
      ls_model.score(X_train_scaled, y_train))
print('학습 평가(DT) : ', 
      dt_model.score(X_train, y_train))

print('테스트 평가(LR) : ', 
      lr_model.score(X_test_scaled, y_test))
print('테스트 평가(LS) : ', 
      ls_model.score(X_test_scaled, y_test))
print('테스트 평가(DT) : ', 
      dt_model.score(X_test, y_test))

from sklearn.metrics import confusion_matrix

pred_train = dt_model.predict(X_train)
pred_test = dt_model.predict(X_test)

print('confusion_matrix - 학습')
print(confusion_matrix(y_train, pred_train))

print('confusion_matrix - 테스트')
print(confusion_matrix(y_test, pred_test))

from sklearn.metrics import classification_report

pred_train = dt_model.predict(X_train)
pred_test = dt_model.predict(X_test)

print('classification_report - 학습')
print(classification_report(y_train, pred_train))

print('classification_report - 테스트')
print(classification_report(y_test, pred_test))

from sklearn.tree import export_graphviz

export_graphviz(dt_model, out_file='./wine.dot',                
                feature_names=X.columns,
                filled=True)

import graphviz
from IPython.display import display

with open('./wine.dot', encoding='utf-8') as f:
    dot_graph = f.read()
    
display(graphviz.Source(dot_graph))















