# -*- coding: utf-8 -*-

import pickle

# 학습된 머신러닝 객체를 참조하는 변수
model = None
with open('./save/save_model_using_pickle.bin','rb') as f:
    # pickle을 사용하여 파일에 저장된 머신러닝 모델의
    # 객체를 로딩
    model = pickle.load(f)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1)

print("학습 평가 : ", 
      model.score(X_train, y_train))
print("테스트 평가 : ", 
      model.score(X_test, y_test))

"""
학습 평가 :  1.0
테스트 평가 :  0.965034965034965
"""