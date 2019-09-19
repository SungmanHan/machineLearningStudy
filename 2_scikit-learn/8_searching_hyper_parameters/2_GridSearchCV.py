# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)

# 데이터를 훈련 및 검증 세트, 테스트 세트로 분할
# - 학습 데이터는 모델의 학습에만 사용
# - 검증 데이터는 학습의 결과를 확인하기 위한 용도로 사용
# - 테스트 데이터는 검증 데이터에 대해서 가장 높은 성적을
# 기록한 모델을 최종적으로 테스트하기 위한 용도로 사용
# 7 : 1 : 2
X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, stratify=y, 
        test_size=0.2, random_state=10)

X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, stratify=y_temp, 
        test_size=0.1, random_state=10)

# 각 하이퍼 파라메터의 조합에 따른 최적의 평가 점수를
# 저장하기 위한 변수
best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100] :
    for C in [0.001, 0.01, 0.1, 1, 10, 100] :
        # 각 매개변수의 조합에 대해서 
        # SVC 모델 객체를 생성하여 훈련
        model = SVC(C=C, gamma=gamma, 
                    random_state=1).fit(
                            X_train, y_train)
        
        # 검증 세트를 사용하여 SVC 모델을 평가
        score = model.score(X_valid, y_valid)
        
        # 평가 점수가 높은 경우 
        # 해당 SVC 모델과 매개변수를 저장
        if score > best_score :
            best_score = score
            best_params = {'C':C,'gamma':gamma,'random_state':1}
        
print('최고 점수 : ', best_score)
print('최적의 파라메터 : \n', best_params)

# 가장 높은 평가점수를 기록한 하이퍼 파라메터를 사용하여
# 모델의 생성 후 훈련 및 검증 세트를 사용하여 학습
best_model = SVC(**best_params).fit(X_temp, y_temp)
print('모델 평가(test) : ', 
      best_model.score(X_test, y_test))

















