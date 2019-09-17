# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42)

from sklearn.ensemble import GradientBoostingClassifier

# GradientBoostingClassifier의 모델을 선언하기 위한
# 하이퍼 파라메터
# - 제약조건을 강화하면서 모델의 개수를 늘려나가는 방식을
# 사용하는 것이 일반화 성능의 강황에 도움이 됨

# 1. max_depth
# - 되도록 5 이상은 주지않는 것이 일반적인 설정
model = GradientBoostingClassifier(
        random_state=1, n_estimators=10000,
        max_depth=1).fit(X_train, y_train)

print('학습 평가 : ', 
      model.score(X_train, y_train))
print('테스트 평가 : ', 
      model.score(X_test, y_test))

# 2. learning_rate
# learning_rate의 값은 내부적인 트리를 생성하는 것에 연관되며
# 낮은 learning_rate는 많은 하위 트리를 생성하게 됨
# 테스트를 통한 적절한 값을 찾는 것이 중요함
model = GradientBoostingClassifier(
        random_state=1, n_estimators=10000,
        learning_rate=0.1).fit(X_train, y_train)

print('학습 평가 : ', 
      model.score(X_train, y_train))
print('테스트 평가 : ', 
      model.score(X_test, y_test))

# 3. subsample
# 딥러닝의 Drop-out과 유사한 개념을 적용한 매개변수
# 전체 데이터를 학습하지 않고 일부분의 데이터를 사용하여
# 학습을 진행할 수 있도록 함
# (과적합을 방지하기 위한 제약 조건)
model = GradientBoostingClassifier(
        random_state=1, 
        subsample=0.15).fit(X_train, y_train)

print('학습 평가 : ', 
      model.score(X_train, y_train))
print('테스트 평가 : ', 
      model.score(X_test, y_test))














