# -*- coding: utf-8 -*-

# 앙상블 방법론(ensemble methods)
# 모형 결합(model combining) : 다수개의 예측기를 결합하여 
# 하나의 예측 모델을 생성하는 방법
# 특정한 하나의 예측 방법이 아닌 복수의 예측 모형을 결합하여 
# 더 나은 성능의 예측을 하려는 시도로 나온 방법

# 단점
# 앙상블 방법은 일반적으로 머신러닝 모델의 계산량이 증가함

# 장점
# - 단일 모형을 사용할 때 보다 성능 분산이 
# 감소(과최적화를 방지)
# - 개별 모형의 성능이 안좋을 경우에는 
# 결합 모형의 성능이 더 향상

# 앙상블의 모형 결합을 위한 방법
# 취합(aggregation), 부스팅(boosting)

# 취합(aggregation) : 사용할 모형의 집합이 이미 결정되어 있는 경우
# - 다수결 (Majority Voting), 배깅 (Bagging), 랜덤포레스트 (Random Forests)
# 부스팅(boosting) : 사용할 모형을 점진적으로 늘려나가려는 경우
# - 에이다부스트 (AdaBoost), 그레디언트 부스트 (Gradient Boost)

# 다수결 (Majority Voting) 처리를 사용한 모델 생성 예제
# hard voting: 단순 투표. 개별 모형의 결과 기준
# soft voting: 가중치 투표. 개별 모형의 조건부 확률의 합 기준

# VotingClassifier 클래스
# - estimators: 
# 예측기 목록, 리스트나 named parameter 형식을 지원
# - voting: 문자열 {hard, soft}
#  디폴트는 hard

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1)

# 데이터 분석을 위한 개별 머신러닝 모델의 생성과 학습
from sklearn.neighbors import KNeighborsClassifier
kn_model = KNeighborsClassifier(
        n_neighbors=5).fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(
        solver='lbfgs', 
        max_iter=10000).fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(
        max_depth=3, 
        random_state=1).fit(X_train, y_train)

print('학습 평가(kn) : ', 
      kn_model.score(X_train, y_train))
print('학습 평가(lr) : ', 
      lr_model.score(X_train, y_train))
print('학습 평가(dt) : ', 
      dt_model.score(X_train, y_train))

# 앙상블 모형 객체 생성 및 학습
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(
        estimators=[('kn', kn_model),
                    ('lr', lr_model),
                    ('dt', dt_model)],
                    n_jobs=-1).fit(X_train, y_train)

print('학습 평가(ensemble) : ', 
      ensemble.score(X_train, y_train))


print('테스트 평가(kn) : ', 
      kn_model.score(X_test, y_test))
print('테스트 평가(lr) : ', 
      lr_model.score(X_test, y_test))
print('테스트 평가(dt) : ', 
      dt_model.score(X_test, y_test))
print('테스트 평가(ensemble) : ', 
      ensemble.score(X_test, y_test))








