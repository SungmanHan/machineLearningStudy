# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:04:15 2019

@author: powergen
"""
import pandas as pd

# 데이터 로딩
path_diabetes = '../../data/diabetes.csv'

o_diabetes = pd.read_csv(path_diabetes,header=None)

r_diabetes = pd.DataFrame(o_diabetes)

pd.options.display.max_columns = 9

# 데이터 확인
print('-'*20)
print(r_diabetes.head())
print('-'*20)
print(r_diabetes.tail())
print('-'*20)
print('info : \n ',r_diabetes.info())
print('-'*20)
print('describe : \n ',r_diabetes.describe())
print('-'*20)

# 입력 데이터(특성) 로딩
X = pd.DataFrame(r_diabetes.iloc[:,:-1])

# 라벨 데이터(정답) 로딩
y = pd.Series(r_diabetes.iloc[:,-1])

print('-'*20)

# 특성 데이터 의 스케일 확인
from matplotlib import pyplot as plt
X.hist()
plt.show()

print(y.value_counts())
print(y.value_counts() / len(y))


# 데이터 분리 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=0.25, stratify=y.values,
        random_state=1)

print('-'*20)
# 데이터 분할 확인
print('데이터 분할 확인 : \n ',X_train.shape[0], X_test.shape[0])

# 학습
from sklearn.neighbors import KNeighborsClassifier

R = 0
RK = 0

for i in range(1,101,1):
  K = i
  model = KNeighborsClassifier(
                      n_neighbors=K, 
                      n_jobs=-1
                      ).fit(X_train, y_train)
  accuracy = model.score(X_train, y_train)
  accuracy_t = model.score(X_test, y_test)
  predicted = model.predict(X_test[:10]) * 100
  predicted_proba = model.predict_proba(X_test[:10]) * 100
  predicted_proba = model.predict_proba(X_test[:10]) * 100
  if R == 0 :
    R = accuracy_t
    RK = K
  else :
    if R < accuracy_t : 
      R = accuracy_t
      RK = K

# 결과
model = KNeighborsClassifier(n_neighbors=RK, n_jobs=-1)
model.fit(X_train, y_train)
accuracy = model.score(X_train, y_train)
accuracy_t = model.score(X_test, y_test)
predicted_proba = model.predict_proba(X_test[:10]) * 100
print('학습 데이터 Score : ',accuracy*100)
print('테스트 데이터 Score : ',accuracy_t*100)
print('예측 확률: \n', predicted_proba)
print('정답 : \n', y_test[:10])
print('최적 K :',RK)

from sklearn.metrics import accuracy_score as accs

pred_train = model.predict(X_train)
pred_train_test = model.predict(X_test)

print('학습 데이터에 대한 정확도 :  \n',accs(y_train,pred_train))
print('테스트 데이터에 대한 정확도 :  \n',accs(y_test,pred_train_test))

