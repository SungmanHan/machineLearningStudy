# -*- coding: utf-8 -*-

# data 디렉토리에 저장된 SMSSpamCollection 파일을 분석하여
# 결과를 확인하세요
# (말뭉치 변환에 TfidfVectorizer 클래스를 활용하세요.)

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

#vectorizer = TfidfVectorizer().fit(X_train_raw)
#vectorizer = TfidfVectorizer(stop_words='english').fit(X_train_raw)
vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=3).fit(X_train_raw)

X_train = vectorizer.transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

print('토큰(단어)의 개수 : ', 
      len(vectorizer.vocabulary_))

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
        n_estimators=10000,
        max_depth=5,
        max_features=0.5,
        random_state=1,
        n_jobs=-1).fit(X_train, y_train)

# 평가
print("학습 평가 : ", model.score(X_train, y_train))
print("테스트 평가 : ", model.score(X_test, y_test))






















