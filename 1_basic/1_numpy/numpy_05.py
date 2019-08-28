# -*- coding: utf-8 -*-

import numpy as np
 
# numpy 배열은 파이썬 리스트와 같이
# 슬라이싱 연산이 지원됨
python_list = [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
              ]

numpy_array = np.array(python_list)

# 슬라이싱 연산
# 시작인덱스 : 종료인덱스
# 주의사항 -> 종료인덱스는 포함되지 않음
slice_1 = numpy_array[0:2, 0:2]
print(slice_1)
 
# 슬라이싱 연산시 시작 및 종료인덱스를 생략
# 할 수 있음
# 시작 인덱스를 생략하는 경우 0 으로 시작됨
# 종료 인덱스를 생략하는 경우 마지막 요소까지 
# 포함하여 반환함
slice_2 = numpy_array[1:, 1:]
print(slice_2)

# 슬라이싱 연산에서 -1 값을 활용하는 예제
# -1은 마지막 인덱스를 의미함
# 아래의 예제는 모든 행의 데이터를 추출하되
# 마지막 열의 값을 제외하고 추출함
slice_3 = numpy_array[:, :-1]
print(slice_3)

# 아래와 같이 모든 행에 대해서 특정 열의 정보만
# 추출하는 경우 1차원 배열이 반환됨
slice_4 = numpy_array[:, -1]
print(slice_4)





















