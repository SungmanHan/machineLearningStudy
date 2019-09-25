# -*- coding: utf-8 -*-

import joblib

# 학습된 머신러닝 객체를 참조하는 변수
model = joblib.load('./save/save_model_using_joblib.bin')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=11)

print("학습 평가 : ", 
      model.score(X_train, y_train))
print("테스트 평가 : ", 
      model.score(X_test, y_test))

"""
학습 평가 :  1.0
테스트 평가 :  0.951048951048951
"""