# -*- coding: utf-8 -*-

import joblib
grid_model = joblib.load('./save/save_model_cancer.bin')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1)

print("학습 평가 : ", 
      grid_model.score(X_train, y_train))
print("테스트 평가 : ", 
      grid_model.score(X_test, y_test))

"""
학습 평가 :  0.9953051643192489
테스트 평가 :  0.972027972027972
"""








