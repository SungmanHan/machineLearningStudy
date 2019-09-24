# -*- coding: utf-8 -*-

# sms.csv 파일의 데이터를 분석할 수 있는 모델을 생성하여
# 분석 결과를 확인하세요.

import pandas as pd

fname = '../../data/sms.csv'
sms = pd.read_csv(fname)

X_raw = sms.message
y = sms.label

print(X_raw[:5])
print(y[:5])

print(y.value_counts())
print(y.value_counts() / len(y))

from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train, y_test = \
    train_test_split(X_raw.values, y.values, 
                     stratify=y.values, random_state=0)
    
print(X_train_raw.shape[0], X_test_raw.shape[0])

# 문자열로 구성된 특성(X) 데이터의 전처리 수행
from sklearn.feature_extraction.text import CountVectorizer

# 모든 전처리의 과정은 학습 데이터에 한해서 수행합니다.
# - 모든 데이터를 대상으로 전처리를 수행하는 경우
# 머신러닝 알고리즘이 테스트 데이터를 미리 확인하는 
# 결과를 가져올 수 있기 때문에...
#vectorizer = CountVectorizer().fit(X_raw.values)
#X = vectorizer.transform(X_raw).toarray()
#print(X[:5])

# 학습 데이터에 존재하는 각 단어를 Counting 하도록 전처리
vectorizer = CountVectorizer().fit(X_train_raw)

print("토큰 개수 : ", len(vectorizer.vocabulary_))
print("변환 결과(희소행렬) : ", vectorizer.transform(X_train_raw))
print("변환 결과(밀집행렬) : ", vectorizer.transform(X_train_raw).toarray())

X_train = vectorizer.transform(X_train_raw)

# 전처리를 수행할 때, 테스트 데이터에 대해서는 
# fit 메소드를 사용하지 않습니다.
# (문자열 전처리의 경우 fit 메소드의 결과로 각 단어의 
#  인덱스 값이 변경될 수 있기때문에)
X_test = vectorizer.transform(X_test_raw)

# 스팸(0), 햄(1) 데이터를 분류하기 위한 예측기 객체 생성
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')

# 교차 검증을 통해서 사전 평가
from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
score = cross_val_score(model, X_train, y_train, 
                        cv=kfold, n_jobs=-1).mean()

print('교차검증 결과 : ', score)

# 모델 학습
model.fit(X_train, y_train)

# 모델 평가
print('학습 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

# 라벨 데이터의 편향이 심하므로
# 정밀도, 재현율을 확인해야함
from sklearn.metrics import classification_report
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print('학습 결과 보고서')
print(classification_report(y_train, pred_train))

print('테스트 결과 보고서')
print(classification_report(y_test, pred_test))





























