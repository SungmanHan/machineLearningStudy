# -*- coding: utf-8 -*-

# GridSearchCV 클래스를 사용하여
# SMSSpamCollection 파일을 분석할 수 있는 
# 최적의 모델을 검색하여 분석 결과를 출력하세요.

import pandas as pd

fname = '../../data/SMSSpamCollection'
# 탭 문자를 기준으로 데이터터가 구성되었으므로
# sep 매개변수의 값을 \t 로 지정합니다.
sms = pd.read_csv(fname, header=None, sep='\t')

print(sms[:3])

X = sms.iloc[:,1]
y = sms.iloc[:,0]

from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train, y_test = \
    train_test_split(X, y, stratify=y, random_state=1)

# X 데이터의 전처리(BOW 벡터)
# 학습 데이터의 토큰을 기존으로 테스트 데이터까지 생성
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer().fit(X_train_raw)
#vectorizer = TfidfVectorizer(stop_words='english').fit(X_train_raw)
#vectorizer = TfidfVectorizer(
#        stop_words='english',
#        min_df=3).fit(X_train_raw)

X_train = vectorizer.transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

print('토큰(단어)의 개수 : ', 
      len(vectorizer.vocabulary_))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
param_grid = {'n_estimators' : [1000,2000,3000],
              'max_depth' : [3,5,7,10],
              'max_features' : [0.3,0.5,0.7]}
base_model = RandomForestClassifier(
        random_state=1,
        n_jobs=-1)

grid_model = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=kfold, iid=True,
        n_jobs=-1).fit(X_train, y_train)

# 평가
print("학습 평가 : ", grid_model.score(X_train, y_train))
print("테스트 평가 : ", grid_model.score(X_test, y_test))

# 하이퍼 파라메터 검색 결과 확인
print("하이퍼 파라메터 검색 결과 : ", 
      grid_model.best_params_)
print("최고 점수 : ", 
      grid_model.best_score_)













