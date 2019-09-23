# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, 
        test_size=0.2, random_state=10)

# 각 하이퍼 파라메터의 조합에 따른 최적의 평가 점수를
# 저장하기 위한 변수
best_score = 0

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

for gamma in [0.001, 0.01, 0.1, 1, 10, 100] :
    for C in [0.001, 0.01, 0.1, 1, 10, 100] :
        # 각 매개변수의 조합에 대해서 
        # SVC 모델 객체를 생성
        model = SVC(C=C, gamma=gamma, 
                    random_state=1)
        
        # 교차 검증을 사용하여
        # 머신러닝 모델의 평가 점수를 반환
        kfold = KFold(n_splits=5, shuffle=True,
                      random_state=1)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=kfold, n_jobs=-1)
        
        # 교차 검증 정확도의 평균을 계산합니다
        score = scores.mean()
        
        # 평가 점수가 높은 경우 
        # 해당 SVC 모델과 매개변수를 저장
        if score > best_score :
            best_score = score
            best_params = {'C':C,
                           'gamma':gamma,
                           'random_state':1}
        
print('최고 점수 : ', best_score)
print('최적의 파라메터 : \n', best_params)

# 가장 높은 평가점수를 기록한 하이퍼 파라메터를 사용하여
# 모델의 생성 후 훈련 세트를 사용하여 학습
best_model = SVC(**best_params).fit(X_train, y_train)
print('모델 평가(test) : ', 
      best_model.score(X_test, y_test))

















