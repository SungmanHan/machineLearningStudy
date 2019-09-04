# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 라벨(정답)이 0인 X 데이터의 생성
class_0 = np.array([[1,2],[3,5],[2,7],[5,5],[4,5]])
# 라벨(정답)이 0인 X 데이터의 생성
class_1 = class_0 + 10

# 테스트 데이터 생성
test = np.array([[10,10]])

# 산점도 차트 출력
# 라벨이 0인 데이터를 파란색으로
# 1인 데이터를 초록색으로 출력
# - 테스트 데이터는 빨간색으로 출력
plt.scatter(class_0[:,0], class_0[:,1], c='b')
plt.scatter(class_1[:,0], class_1[:,1], c='g')
plt.scatter(test[:,0], test[:,1], c='r')
plt.show()

# 라벨(정답) 데이터 생성
label = np.r_[np.full([5], 'BLUE'),
              np.full([5], 'GREEN')]

# X 데이터의 통합
X = np.r_[class_0, class_1]

# X 데이터의 통합(pandas 모듈 사용)
X_blue = pd.DataFrame(class_0)
X_green = pd.DataFrame(class_1)

X = pd.concat([X_blue, X_green], 
              ignore_index=True)

# 라벨 데이터의 저장(pandas 모듈 사용)
y = pd.Series(label)

# 최근접 이웃 알고리즘을 구현하고 있는
# 사이킷 런의 예측기 클래스 사용
from sklearn.neighbors import KNeighborsClassifier

# 머신러닝 예측기의 객체 생성
model = KNeighborsClassifier(n_neighbors=1)

# 머신러닝 예측기 객체에게 데이터를 학습시킴
# - fit 메소드는 예측기 객체를 학습시킬 때 호출하는 메소드
# - 첫번째 매개변수는 입력데이터(2차원 데이터)
# - 두번째 매개변수는 라벨데이터(1차원 데이터)
# - 버전에 따라서 pandas 모듈의 객체의 사용이
#   불가할 수 있기 때문에 numpy 배열을 입력
model.fit(X.values, y.values)

# 머신러닝 예측기를 사용하여 테스트 데이터를 예측
predicted = model.predict(test)

# 예측 결과 출력
print('test -> ', predicted)














