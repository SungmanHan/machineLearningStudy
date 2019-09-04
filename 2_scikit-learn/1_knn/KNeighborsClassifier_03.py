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

# 사이킷 런을 사용하지 않고, 최근접 이웃 알고리즘을 
# 구현하여 테스트 결과를 반환받는 예제
X_test = np.array([[155,70]])

# 최근접 이웃 알고리즘이 거리를 계산할 때 사용하는 방법
# - 유클리드 거리 계산법을 적용함
# - 테스트 데이터와 학습 데이터 사이의 유클리드 거리 계산
# - (T1, T2)와 (R1, R2) 사이의 유클리드 거리 계산법
# - A = ((T1 - R1)의 제곱 + (T2 - R2)의 제곱) 
# - 유클리드 거리 = A의 제곱근
distances_step1 = X_train - X_test
print(distances_step1)

distances_step2 = distances_step1 ** 2
print(distances_step2)

# np.sum 함수 : 매개변수의 총 합계를 구하는 함수
# axis 매개변수의 값을 지정하지 않으면 전체 합계가 반환
# axis = 0 인경우 열의 함계를 반환
# axis = 1 인경우 행의 함계를 반환
distances_step3 = np.sum(distances_step2, axis=1)
print(distances_step3)

# np.sqrt 함수 : 매개변수의 제곱근 값을 반환
distances = np.sqrt(distances_step3)
print('테스트 데이터에 대한 거리 계산 결과 : \n',
      distances)

# 테스트 데이터와 가장 가까운 학습 데이터 3개의 인덱스 추출
# numpy 배열의 argsort 메소드
# 배열 내부를 오름차순으로 정렬했을때의 인덱스 값을 반환
# 내림차순의 경우 argsort()[::-1][:3]
nearest_indices = distances.argsort()[:3]
print(nearest_indices)

# 테스트 데이터와 가장 가까운 학습 데이터 3개의 
# 인덱스를 활용하여 각 학습 데이터의 정답 데이터를 추출
# np.take 함수는 1번째 매개변수로 전달된 배열로부터
# 2번째 매개변수로 전달된 인덱스에 해당되는 요소를 반환
nearest_genders = np.take(y_train, nearest_indices)
print(nearest_genders)

# 내부 데이터의 개수를 확인할 수 있는 Counter 클래스
# (중복을 제거시킨 각 값의 개수를 확인할 수 있음)
from collections import Counter
counter = Counter(nearest_genders)
print(counter)

print(counter.most_common())
print(counter.most_common()[0][0])

# 예측 확률 값을 출력
K = 3
print(counter.most_common()[0][1] / K)
print(counter.most_common()[1][1] / K)

























