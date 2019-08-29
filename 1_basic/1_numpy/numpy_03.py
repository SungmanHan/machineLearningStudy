# -*- coding: utf-8 -*-

import numpy as np

# 파이썬 리스트 생성
# range 함수(시작값, 종료값, 증가치)
# 아래의 코드는 [1,2,3,4,5,6,7,8,9,10] 리스트를 생성합니다.
python_list = list(range(1,11))

# numpy 배열 생성
numpy_array_1 = np.array(python_list)
print(f"numpy_array_1.shape -> {numpy_array_1.shape}")
print(f"numpy_array_1 -> {numpy_array_1}")

# 배열의 형태를 수정할 수 있는 reshape() 메소드
# 1차원 배열을 2차원 배열로 형태를 변환하는 예제
# -1 매개변수가 사용되는 경우 나머지 매개변수 값에의해서
# 자동으로 계산된 값이 적용됩니다.
# (아래의 예의 경우 열의 수를 2로 고정하여 행의 수는
#  5로 처리됩니다.)
numpy_array_2 = numpy_array_1.reshape(-1, 2)
print(f"numpy_array_1.shape -> {numpy_array_1.shape}")
print(f"numpy_array_2.shape -> {numpy_array_2.shape}")
print(f"numpy_array_2 -> {numpy_array_2}")

# 2차원 배열을 1차원 배열로 형태를 변환하는 예제
numpy_array_3 = numpy_array_2.reshape(-1)
print(f"numpy_array_3.shape -> {numpy_array_3.shape}")
print(f"numpy_array_3 -> {numpy_array_3}")










