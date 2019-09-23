# -*- coding: utf-8 -*-

# load_iris 데이터를 
# GridSearchCV 클래스를 활용하여 분석하세요.
# 활용할 예측기는 LogisticRegression 입니다.

# solver : ['lbfgs', 'liblinear']
# penalty : ['l1', 'l2']
# C : [ ... ]

# 실행 시, 에러가 발생되는 경우 조건부 매개변수를
# 사용하여 처리하세요.

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, 
        test_size=0.2, random_state=10)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
# 교차검증 점수를 기반으로 최적의 하이퍼 파라메터를 
# 검색할 수 있는 GridSearchCV 클래스
from sklearn.model_selection import GridSearchCV

# 조건부 매개변수를 사용하기 위한 매개변수 그리드 선언
# 사용 방식 
# - [{조건부 매개변수 1}, {조건부 매개변수 2} ... ]
param_grid = [{'solver' : ['lbfgs'],
              'penalty' : ['l2'],
              'C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'solver' : ['liblinear'],
              'penalty' : ['l1', 'l2'],
              'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

# 폴드를 설정
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

# GridSearchCV 클래스의 하이퍼 파라메터 정보
# GridSearchCV(예측기 객체, 테스트 파라메터의 
# 딕셔너리 객체, 
# cv=교차검증 폴드 수,...)
grid_search = GridSearchCV(
        estimator=LogisticRegression(n_jobs=-1, random_state=1),
        param_grid=param_grid,
        cv=kfold,
        n_jobs=-1).fit(X_train, y_train)

print('테스트 평가 : ', 
      grid_search.score(X_test, y_test))

print('최적의 매개변수 : \n',
      grid_search.best_params_)

print('최적의 교차검증 점수 : \n',
      grid_search.best_score_)

























