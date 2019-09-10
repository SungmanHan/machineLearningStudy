# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data)
y = pd.Series(cancer.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X.values, y.values, 
                     stratify=y.values,
                     random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import LinearSVC

model = LinearSVC(
        C=1.0, max_iter=10000).fit(X_train, y_train)

print('훈련 평가 : ', 
      model.score(X_train, y_train))

print('테스트 평가 : ', 
      model.score(X_test, y_test))

print('모델의 예측 결과 : \n',
      model.predict(X_test[:5]))

print('실제 정답 : \n', y_test[:5])

# LinearSVC 클래스는 확률 값을 반환하는 
# predict_proba 메소드가 제공되지 않음
#print('모델의 예측 확률 : \n',
#      model.predict_proba(X_test[:5]))

# decision_function 메소드를 사용하여 
# 예측 결과의 과정을 이해할 수 있음
# 0 을 기준으로 작다면 음성, 
# 크다면 양성으로 예측함
pred = model.decision_function(X_test[:5])
print('모델의 예측 값 : \n', pred)
















