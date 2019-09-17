# -*- coding: utf-8 -*-

# SVM(Support Vector Machin)은 러시아 과학자 Vladimir Vapnik이 1970년대 후반에 제안한 알고리즘
# 분류(classification)문제에서 우수한 일반화

from sklearn.datasets import load_breast_cancer
X,y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)

#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

from sklearn.svm import SVC
svc_model = SVC(gamma='scale').fit(X_train,y_train)

print('학습 평가 : ',svc_model.score(X_train,y_train))
print('테스트 평가 : ',svc_model.score(X_test,y_test))
