# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

# 신장과 몸무게의 데이터를 저장하고 있는 X 데이터
X_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67]
])
# 신장과 몸무게 정보를 이용하여 예측해야하는 성별 정보
y_train = ['male', 'male', 'male', 'male', 'female', 
           'female', 'female', 'female', 'female']

#plt.figure(figsize=(13,7))
#plt.title('Data Sets')
#plt.xlabel('Height')
#plt.ylabel('Weight')
#
#for i, data in enumerate(X_train) :
#    plt.scatter(data[0], data[1], c='k',
#                marker='x' if y_train[i] == 'male' else 'D')
#
#plt.grid(True)
#plt.show()

# 사이킷 런의 KNN 알고리즘을 적용하여 예측하는 예제
X_test = np.array([[155,70]])

# KNN 알고리즘 구현하고 있는 분류를 위한 
# KNeighborsClassifier 클래스
from sklearn.neighbors import KNeighborsClassifier

# 하이퍼 파라메터 정보 설정
# - 이웃 탐색을 위한 최근접 이웃의 개수를 변수로 선언
K=3

# KNeighborsClassifier의 객체(모델)을 생성 시
# 최근접 이웃의 개수를 생성자로 전달
# (기본값은 5)
model = KNeighborsClassifier(n_neighbors=K, n_jobs=-1)

# fit 메소드를 사용하여 학습 데이터를 학습
# 별도의 학습을 진행하지 않고 저장함(게으른 학습 방법)
model.fit(X_train, y_train)

# predict 메소드를 사용하여 예측 결과를 반환받음
predicted = model.predict(X_test)
print(predicted)

# 각 클래스별 예측 확률을 확인
predicted_proba = model.predict_proba(X_test)
print(predicted_proba)

# predict_proba를 사용하는 경우 확률 값 만이 
# 반환되기 때문에 실제 라벨(정답)을 확인하려는 경우
# classes_ 를 사용해야합니다.
print(model.classes_)
print(model.classes_[
        predicted_proba[0].argsort()[::-1][0]])























