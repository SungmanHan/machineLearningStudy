# -*- coding: utf-8 -*-

# 앙상블 방법에서 사용하는 독립적인 예측기의 수가 
# 많을 수록 성능 향상이 일어날 가능성이 높음 
# 다만, 다른 확률 모형을 사용하는데에는 한계가 있기때문에 
# 일반적으로 배깅(bagging) 방법을 사용
# 배깅(bagging) : 같은 예측기를 사용하지만 
# 서로 다른 결과를 출력하는 다수의 예측기를 적용하는 방법
# 동일한 예측기과 데이터를 사용하지만, 
# 부트스트래핑(bootstrapping)과 유사하게 트레이닝 데이터를 
# 랜덤하게 선택해서 다수결 예측기를 적용

# BaggingClassifier 클래스를 사용하여 
# 배깅(bagging)을 적용할 수 있음

# BaggingClassifier 클래스의 하이퍼 파라메터
# base_estimator: 기본 모형
# n_estimators: 모형 갯수. 디폴트 10
# bootstrap: 데이터의 중복 사용 여부. 디폴트 True
# max_samples: 데이터 샘플 중 선택할 샘플의 수 혹은 비율. 
# 디폴트 1.0
# bootstrap_features: 특징 차원의 중복 사용 여부. 디폴트 False
# max_features: 다차원 독립 변수 중 선택할 차원의 수 혹은 
# 비율 디폴트 1.0

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(
        random_state=1).fit(X_train, y_train)

print('학습 평가(dt) : ', 
      dt_model.score(X_train, y_train))

from sklearn.ensemble import BaggingClassifier
ensemble = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(
                max_depth=5, random_state=1),
                n_estimators=100000,
                max_samples=0.25,
                max_features=0.25,
                random_state=1,
                n_jobs=-1).fit(X_train, y_train)

print('학습 평가(ensemble) : ', 
      ensemble.score(X_train, y_train))

print('테스트 평가(dt) : ', 
      dt_model.score(X_test, y_test))
print('테스트 평가(ensemble) : ', 
      ensemble.score(X_test, y_test))













