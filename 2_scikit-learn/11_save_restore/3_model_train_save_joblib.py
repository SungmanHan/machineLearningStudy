# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=11)

model = GradientBoostingClassifier(
        n_estimators=1500,
        max_depth=2,
        max_features=0.3,
        subsample=0.5,
        random_state=1).fit(X_train, y_train)

print("학습 평가 : ", 
      model.score(X_train, y_train))
print("테스트 평가 : ", 
      model.score(X_test, y_test))

"""
학습 평가 :  1.0
테스트 평가 :  0.951048951048951
"""

import os
# pip install joblib
import joblib

try :
    # 현재 경로에 save 디렉토리의 존재 여부 확인
    if not (os.path.isdir('./save')) :
        # 만약 save 디렉토리가 없다면 생성
        os.makedirs(os.path.join('./save'))
except OSError as e :
    print('save 디렉토리 생성 에러')
    
joblib.dump(model, './save/save_model_using_joblib.bin')




























