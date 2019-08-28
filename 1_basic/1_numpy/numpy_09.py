# -*- coding: utf-8 -*-

# 파이썬의 리스트 변수 생성
list_1 = [1,2,3]
list_2 = [4,5,6]

# 파이썬의 리스트는 사칙 연산이 허용되지 않습니다.
# 에러 발생
#print(list_1 + 10)

# 파이썬 리스트에 대해서 사칙 연산을 적용하는 예제
# - 반복문을 활용한 리스트의 생성
print([data + 10 for data in list_1])

print(list_1 + list_2)

print([data1 + data2 for data1, data2 in zip(list_1,list_2)])

import numpy as np

numpy_array_1 = np.array([1,2,3])
numpy_array_2 = np.array([4,5,6])
 
# numpy 배열의 각 요소의 합계
numpy_array_r = numpy_array_1 + numpy_array_2
# numpy_array_r = np.add(numpy_array_1, numpy_array_2)
print(numpy_array_r)

# numpy 배열의 각 요소의 차
numpy_array_r = numpy_array_1 - numpy_array_2
# numpy_array_r = np.subtract(numpy_array_1, numpy_array_2)
print(numpy_array_r)

# numpy 배열의 각 요소의 곱
numpy_array_r = numpy_array_1 * numpy_array_2
# numpy_array_r = np.multiply(numpy_array_1, numpy_array_2)
print(numpy_array_r)

# numpy 배열의 각 요소의 나눗셈
numpy_array_r = numpy_array_1 / numpy_array_2
# numpy_array_r = np.divide(numpy_array_1, numpy_array_2)
print(numpy_array_r)


numpy_array_3 = np.array([1,2,3,4])

# numpy 배열은 브로드캐스팅 연산이 허용되는 타입입니다.
# - 분배법칙이 성립됨
# - 큰 차원의 각 요소에 작은 차원의 값이 각각 연산되는 방식
numpy_array_r = numpy_array_3 + 10
print(numpy_array_r)

# 1차원 배열끼리 연산이 수행되는 경우
# 하나의 요소를 가진 배열이 다수개의 요소를 가진
# 배열에 각각 연산됩니다.
numpy_array_4 = np.array([10])
numpy_array_r = numpy_array_3 + numpy_array_4
print(numpy_array_r)

# 1차원 베열끼리 연산이 수행될 때,
# 두 배열모두 1개 이상의 값을 가지는 경우
# 반드시 두 배열의 크기가 동일해야만 연산이 수행됩니다.
numpy_array_5 = np.array([10,11])
numpy_array_r = numpy_array_3 + numpy_array_5
print(numpy_array_r)

# 2차원 배열과 1차원 배열의 연산이 수행되는 경우
# 2차원의 각 1차원 배열 shape와 동일한 형태의
# 1차원 배열만 연산이 허용됩니다.
numpy_array_6 = np.array([[1,2],[3,4],[5,6]])
numpy_array_7 = np.array([10,20])

numpy_array_r = numpy_array_6 + numpy_array_7
print(numpy_array_r)


















