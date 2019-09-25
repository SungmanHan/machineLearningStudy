# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.svm import SVC
model = SVC(random_state=1)

from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', scaler),
                 ('model', model)])

from sklearn.model_selection import KFold
kfold=KFold(n_splits=5,shuffle=True,random_state=1)

from sklearn.model_selection import GridSearchCV
param_grid={'model__C':[0.001, 0.01,0.1,1,10,100],
            'model__gamma':[0.001, 0.01,0.1,1,10,100]}

grid_model = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=kfold,
        iid=True,
        n_jobs=-1).fit(X_train, y_train)

print("학습 평가 : ", 
      grid_model.score(X_train, y_train))
print("테스트 평가 : ", 
      grid_model.score(X_test, y_test))

"""
학습 평가 :  0.9953051643192489
테스트 평가 :  0.972027972027972
"""

import os
import joblib

try :
    if not (os.path.isdir('./save')) :
        os.makedirs(os.path.join('./save'))
except OSError as e :
    print('save 디렉토리 생성 에러')
    
joblib.dump(grid_model, './save/save_model_cancer.bin')










