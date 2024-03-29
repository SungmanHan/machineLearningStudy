# -*- coding: utf-8 -*-

# 일반적인 머신러닝 단계
# - 성능 향상을 위한 데이터 전처리 단계 추가

# 1. 데이터의 적재 및 분할
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# - 데이터의 적재
cancer = load_breast_cancer()

# - 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target,
        stratify=cancer.target,
        random_state=1)

# 2. 데이터의 전처리 과정 수행
# - 라벨 인코딩, 특성 데이터의 스케일 조정 등의 작업을 수행
# - 사이킷 런의 변환기 클래스를 활용
# - fit 메소드는 반드시 학습 데이터에 대해서만 적용
# - transform 메소드를 사용하여 학습 및 테스트 데이터의 변환을 수행
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 머신러닝 모델 객체의 생성 및 학습
from sklearn.svm import SVC

model = SVC(C=1.0, 
            gamma='scale').fit(X_train_scaled, y_train)

# 4. 학습된 머신러닝 모델 객체의 평가
print("학습 평가 : ", 
      model.score(X_train_scaled, y_train))
print("테스트 평가 : ", 
      model.score(X_test_scaled, y_test))

from sklearn.metrics import classification_report

pred_train = model.predict(X_train_scaled)
pred_test = model.predict(X_test_scaled)

print('학습 데이터의 정밀도, 재현율, F1')
print(classification_report(y_train, pred_train))

print('테스트 데이터의 정밀도, 재현율, F1')
print(classification_report(y_test, pred_test))










