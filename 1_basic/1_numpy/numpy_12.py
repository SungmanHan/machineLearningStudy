# -*- coding: utf-8 -*-

import numpy as np
 
numpy_array_1 = np.array([1, 2, 3])
numpy_array_2 = np.array([4, 5, 6])

# numpy 배열을 왼쪽에서 오른쪽으로 결합
r = np.r_[numpy_array_1, numpy_array_2]
print(r)


numpy_array_1 = np.array([1, 2, 3])
numpy_array_2 = np.array([4, 5, 6])

# numpy 배열을 왼쪽에서 오른쪽으로 결합
r2 = np.hstack([numpy_array_1, numpy_array_2])
print(r2)

array_1_reshape = numpy_array_1.reshape(-1,1)
array_2_reshape = numpy_array_2.reshape(-1,1)

r = np.r_[array_1_reshape, array_2_reshape]
print(r)

# 2개의 1차원 numpy 배열을 
# 세로로 결합하여 2차원 배열 생성
# (각 열의 결합하여 2차원 배열을 생성)
r = np.c_[numpy_array_1, numpy_array_2]
print(r)

# numpy 배열을 왼쪽에서 오른쪽으로 결합
# 세로로 결합하여 2차원 배열 생성
r = np.column_stack([numpy_array_1, numpy_array_2])
print(r)

r = np.c_[array_1_reshape, array_2_reshape]
print(r)

# numpy 배열의 전치 연산의 결과를 반환하는 T 연산자
r = r.T
print(r)

























