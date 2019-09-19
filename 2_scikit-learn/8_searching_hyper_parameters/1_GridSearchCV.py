# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1)

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
        
        # 테스트 세트를 사용하여 SVC 모델을 평가
        score = model.score(X_test, y_test)
        
        # 평가 점수가 높은 경우 
        # 해당 SVC 모델과 매개변수를 저장
        if score > best_score :
            best_score = score
            best_params = {'C':C,'gamma':gamma}
        
print('최고 점수 : ', best_score)
print('최적의 파라메터 : \n', best_params)













