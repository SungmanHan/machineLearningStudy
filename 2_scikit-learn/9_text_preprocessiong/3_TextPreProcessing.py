# -*- coding: utf-8 -*-

# TfidfVectorizer 클래스
# CountVectorizer와 비슷하지만 TF-IDF 방식으로 
# 단어의 가중치를 조정한 BOW 벡터를 생성
# TF-IDF(Term Frequency – Inverse Document Frequency)
# TF : 특정한 단어의 빈도수
# IDF : 특정한 단어가 들어 있는 문서의 수에 반비례하는 수
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['Hello Python',
          'Hello Scikit learn',
          'This is first document',
          'This is second document',
          'This is third document',
          'The last document']

vectorizer = TfidfVectorizer().fit(corpus)

print('토큰(단어)의 개수 : ', len(vectorizer.vocabulary_))
print('토큰(단어)의 내용 : \n', vectorizer.vocabulary_)

print('변환 결과(말뭉치변환) : \n',
      vectorizer.transform(corpus).toarray())


















