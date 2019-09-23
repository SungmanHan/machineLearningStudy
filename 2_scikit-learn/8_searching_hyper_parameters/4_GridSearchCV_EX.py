# -*- coding: utf-8 -*-

excelPath = '../../data/winequality-red.csv'
import pandas as pd
wine = pd.read_csv(excelPath,sep=';')

X = wine.iloc[:,:-1]
y = wine.iloc[:,-1]

print(X.info())
pd.options.display.max_columns = 11
print(X.describe())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X.values)
X = scaler.transform(X.values)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, 
        test_size=0.2, random_state=10)

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5,shuffle=True,random_state=1)


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {'learning_rate':[0.001, 0.01, 0.1, 1, 10],
              'max_depth':[1,2,3,4,5],
              'max_features':[0.2,0.3,0.4,0.5,0.6],
              'n_estimators':[200,300,400,500,600],
              'subsample':[0.2,0.3,0.4,0.5,0.6]}        
grid_search = GridSearchCV(
                  estimator=GradientBoostingClassifier(random_state=1),
                  param_grid=param_grid,
                  cv=kfold,
                  n_jobs=-1).fit(X_train,y_train)

print('best_params : ', grid_search.best_params_)
print('best_score : ', grid_search.best_score_)
print('모델 평가(test) : ', grid_search.score(X_test, y_test))

















