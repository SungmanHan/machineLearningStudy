# -*- coding: utf-8 -*-

import pandas as pd

fname='../../data/iris.csv'
iris=pd.read_csv(fname)

print(iris)

# pandas 모듈을 사용하여 범주형 데이터를 
# 수치 데이터로 변환 방법
# - factorize 메소드
# - 특정 열에 존재하는 모든 데이터의 중복을 제거한 후,
# 각 값에 대해서 정수값을 매핑하여 반환하는 메소드
# 반환값 -> 정수배열, 각 정수에 해당되는 실제 데이터
encoded, categories = iris.Species.factorize()
print('인코딩된 정수의 배열\n', encoded)
print('인코딩된 각 정수의 문자열 값\n', categories)

# 사이킷 런을 활용한 인코딩 방법
# 범주데이터를 수치데이터로 변형하는 방법
# - LabelEncoder 클래스
# - 문자열이나 정수로된 라벨 값을  0  ~  K−1 까지의 
#   정수로 변환 
# - 변환된 규칙은 classes_ 속성에서 확인할 수 있음
# - 예측 결과에 적용할 수 있도록 역변환을 위한 
#   inverse_transform 메서드를 지원
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder().fit(iris.Species)

# 각 범주형 값에 대해서 정수값을 반환하기 위한
# 배열을 확인
# - ['setosa' 'versicolor' 'virginica']
print(encoder.classes_)

# - 'setosa' 데이터 transform 메소드에 전달되는 경우
#   해당되는 인덱스의 값 0이 반환
print(iris.Species[:3])
print(encoder.transform(iris.Species[:3]))

# - 'virginica' 데이터 transform 메소드에 전달되는 경우
#   해당되는 인덱스의 값 2이 반환
print(iris.Species[-3:])
print(encoder.transform(iris.Species[-3:]))

# 전체 데이터 변환
print(encoder.transform(iris.Species))

# 역변환
print(encoder.inverse_transform([0]))
print(encoder.inverse_transform([0,1,2]))

# 사이킷 런을 활용한 인코딩 방법
# 범주데이터를 수치데이터로 변형하는 방법
# - 원핫인코딩
# - 데이터의 대다수가 0으로 구성이 되며, 
#   특정 인덱스의 값 만을 1로 갖도록 변환하는 인코딩 방법
# - 라벨(정답)이 아닌 특성 데이터에 대해서 사용
from sklearn.preprocessing import OneHotEncoder
# 원핫인코딩을 위한 배열의 재배치
# - 원핫인코딩은 반드시 2차원 배열로 데이터를
#   전달해야 합니다.
# - 아래와 같이 fit 메소드에 pandas의 Series 타입을 
#  입력하면 ValueError가 발생합니다.
#encoder = OneHotEncoder().fit(iris.Species.values)

# - iris 품종에 대한 일차원 배열을 2차원 배열로 변환하여
# 처리합니다.
encoder = OneHotEncoder().fit(
        iris.Species.values.reshape(-1,1))

# OneHotEncoder의 fit 메소드를 통해서
# 학습된 범주데이터의 배열
# - 각 범주의 인덱스 값을 살펴봐야함
print(encoder.categories_)

# setosa에 대한 원핫인코딩 결과를 확인
# - 희소행렬이 반환
# - (행의인덱스, 열의인덱스) 해당위치의값
# - OneHotEncoder 클래스의 sparse 매개변수를 
#   True로 지정하는 경우 희소행렬이 반환
# - OneHotEncoder 클래스의 sparse 매개변수를 
#   False로 지정하는 경우 numpy 배열이 반환
print(encoder.transform(
        iris.Species.values[:50].reshape(-1,1)))

print(encoder.transform(
        iris.Species.values[:50].reshape(-1,1)).toarray())

print(encoder.transform(
        iris.Species.values.reshape(-1,1)).toarray())

# 역변환
# inverse_transform 메서드를 사용하여
# 원핫인코딩의 값을 실제 값으로 반환
# 주의사항 - 2차원 배열로 전달해야함
print(encoder.inverse_transform([[1,0,0]]))
print(encoder.inverse_transform([[0,1,0]]))
print(encoder.inverse_transform([[0,0,1]]))
















