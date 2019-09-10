# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)

# 특성이 2개로 구성된 X 데이터
print(X[:10])
# 클래스(카테고리)가 3개로 구성된 y 데이터
# (0,1,2)
print(y[:10])

from sklearn.linear_model import LogisticRegression

# LogisticRegression 클래스는 이진분류뿐만 아니라
# 다항 분류 데이터 셋에 대해서도 동작할 수 있습니다.

# 다중 클래스를 분류하기 위한 하이퍼파라메터
# multi_class : 'ovr', 'multinomial'

# ovr : One VS Rest(일대다 방식의 분류 방법)
# - 각 클래스 별로 이진분류 예측기를 생성하는 방식
# - 분류해야할 클래스가 많아질수록 학습 시간이 오래 걸림

# multinomial : Softmax 방식 기반의 다중 클래스 분류
# - 분류해야할 클래스의 개수와 관계앖이 1개의 모델이 생성
# - 내부적으로 Matrix 연산을 수행하여 각 클래스 별
#   LogisticRegression의 예측 값으르 계산하여
#  가장 큰 확률 값으로 클래스를 예측하는 방식을 사용
# - One-hot Encoding 기반의 라벨 데이터를 사용하여
#  다중 분류를 수행

model = LogisticRegression(
        solver='lbfgs',
        #multi_class='ovr').fit(X, y)
        multi_class='multinomial').fit(X, y)

print('학습 평가 : ', model.score(X, y))

# 사이킷 런의 분류를 위해서 사용되는 예측기들은 
# 다항 분류를 지원하며, 다항 분류를 위해서 
# 일대다 방식을 적용
# - 각 클래스 별로 이진분류 모델을 생성하여
#   특성 데이터에 대한 각 이진분류 모델의 결과에서 
#   가장 큰 값을 취함
# - 아래의 계수 배열과 같은 경우 3개의 클래스를 
#   예측하기 위해서 2개의 특성을 사용하므로 
#   3 X 2 열의 배열이 생성되며,
#   절편의 값 또한 각 클래스 별로 계산되므로 
#   3 의 크기를 갖는 일차원 배열이 반환
print('기울기(가중치) 배열의 shape : ',
      model.coef_.shape)
print('절편(편향) 배열의 shape : ',
      model.intercept_.shape)

print('기울기 : \n', model.coef_)
print('절편 : \n', model.intercept_)

# 다항 분류의 경우 decision_function 메소드의 결과는
# 각 클래스를 구분하기 위한 방정식의 결과에서
# 양수/음수의 값을 반환함
# (가장 큰 값의 인덱스로 예측함)
print(model.decision_function(X[:3]))
print(y[:3])

# 다항 분류의 경우 predict_proba 메소드의 결과는
# 각 클래스에 해당할 확률 값이 반환되며,
# 그 중 가장 큰 값으로 예측하게 됨
print(model.predict_proba(X[:3]))
print(y[:3])
























