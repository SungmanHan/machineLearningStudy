# -*- coding: utf-8 -*-

import numpy as np

# numpy 배열의 생성 함수
# numpy 모듈의 함수를 사용하여 배열을 생성할 수 있음

# 1. np.zeros(배열의형태) : 입력된 형태의 배열을
# 생성하면서 모든 요소의 값을 0으로 초기화
# - 아래의 코드는 3행 2열의 다차원 배열이 생성되고
#   모든 요소의 값은 0으로 채워짐
numpy_array_0 = np.zeros((3,2))
print("numpy_array_0 : \n{0}".format(numpy_array_0))

# 2. np.ones(배열의형태) : 입력된 형태의 배열을
# 생성하면서 모든 요소의 값을 1으로 초기화
# - 아래의 코드는 2행 2열의 다차원 배열이 생성되고
#   모든 요소의 값은 1으로 채워짐
numpy_array_1 = np.ones((2,2))
print("numpy_array_1 : \n{0}".format(numpy_array_1))

# 3. np.full(배열의형태, 값) : 입력된 형태의 배열을
# 생성하면서 모든 요소의 값을 지정한 값으로 초기화
# - 아래의 코드는 3행 3열의 다차원 배열이 생성되고
#   모든 요소의 값은 5으로 채워짐
numpy_array_2 = np.full((3,3), 5)
print("numpy_array_2 : \n{0}".format(numpy_array_2))

# 4. np.eye(배열의 행의크기) : 
# - 2차원 배열을 생성하는 함수
# - 지정된 행의 크기를 갖는 배열을 생성
# - 대각선 방향으로 1의 값은 갖는 배열 생성(대각행렬)
# - 아래의 코드는 5행 5열의 다차원 배열이 생성되고
#   대각선 방향으로 1의 값이 채워짐
numpy_array_3 = np.eye(5)
print("numpy_array_3 : \n{0}".format(numpy_array_3))









