# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data)
y = pd.Series(cancer.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=0.3, stratify=y.values,
        random_state=42)

from sklearn.linear_model import LogisticRegression

# LogisticRegression 클래스는 기본 제약조건으로 
# L2 정규화를 지원
# penalty 하이퍼 파라메터의 값을 l1으로 변경하면 
# 모델의 제약조건을 L1 정규화로 변경할 수 있습니다.
# C의 값을 높일수록 제약의 강도가 낮아지며
# (일부 특성 데이터의 가중치의 값만이 0으로 수렴)
# C의 값은 낮출수록 제약의 강도가 높아집니다.
# (대다수 특성 데이터의 가중치의 값이 0으로 수렴)

# l1 제약조건을 사용하는 경우 solver 매개변수의 값을
# - liblinear : 작은 데이터 셋에 적합한 알고리즘
# - saga : 대용량의 데이터 셋에 적합한 알고리즘
C = 1
model = LogisticRegression(C=C, penalty='l1',
        solver='liblinear', 
        #solver='saga', 
        max_iter=10000).fit(X_train, y_train)

print('훈련 평가 : ', model.score(X_train, y_train))
print('테스트 평가 : ', model.score(X_test, y_test))

# 제약 조건에 따른 기울기의 변화
from matplotlib import pyplot as plt

plt.figure(figsize=(10, 7))
plt.axhline(0, color='y', linestyle='--')

for C, marker in zip([0.01, 1, 100], ['o','^','v']) :
    model = LogisticRegression(C=C, penalty='l1',
        solver='liblinear',         
        max_iter=10000).fit(X_train, y_train)
    
    print(f'C-{C}인 로지스틱 모델의 학습 평가 : ',
          model.score(X_train, y_train))
    print(f'C-{C}인 로지스틱 모델의 테스트 평가 : ',
          model.score(X_test, y_test))
    
    plt.plot(model.coef_.T, marker, label=f'C={C}')
    
plt.legend()
plt.show()















