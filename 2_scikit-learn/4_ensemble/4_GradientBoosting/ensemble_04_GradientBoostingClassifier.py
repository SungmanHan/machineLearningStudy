# -*- coding: utf-8 -*-

# 부스팅 (boosting)
# - 성능이 약한 학습기(weak learner) 다수 개를 연결하여 
#   강한 학습기(strong learner)를 만드는 앙상블 학습 방법
# - 부스팅 방법은 앞서 학습된 약한 모델의 약점을 보완해나가면서 
#   성능이 개선된 더 나은 모델로 학습시키는 것

# 부스팅 방법론을 구현하고 있는 모델
# 1. 에이다부스트(AdaBoost, Adaptive Boosting)
# - 에이다부스트(AdaBoost)는 과소적합(underfitted)된 
#  학습 데이터 샘플의 가중치를 높이면서 새로 학습된 모델이 
#  학습하기 어려운 데이터에 더 잘 적합되도록 하는 방식
# - 에이다부스트의 학습 단계
# a. 전제 학습 데이터셋을 이용해 모델을 만든 후, 
#    잘못 예측(분류)된 샘플의 가중치를 상대적으로 증가시킴
# b. 다음 모델을 학습 시킬 때 업데이트된 가중치가 반영된 
#    데이터를 사용하여 모델을 학습 시킨다. 
# c. a와 b의 과정을 반복

# Scikit-learn의 AdaBoostClassifier 클래스를 사용
from sklearn.ensemble import AdaBoostClassifier

# 2. 그래디언트 부스팅(Gradient Boosting)
# - 아다부스트와 마찬가지로 직전에 학습된 모델의 오차를 
# 보완하는 방향으로 모델(분류기, 학습기)을 추가해주는 방법
# - 그래디언트 부스팅은 아다부스트와 달리 학습단계 마다 
# 데이터 샘플의 가중치를 업데이트 하는 것이 아니라 
# 직전 단계 모델의 잔여 오차에 대해 새로운 모델을 학습시키는 방법

# Scikit-learn의 GradientBoostingClassifier 클래스를 사용
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42)

# AdaBoostClassifier 클래스는 앙상블의 구축을 위해서
# 배깅와 유사한 방식을 사용
# 사용자가 직접 내부에서 사용될 모델을 입력합니다.

# AdaBoostClassifier 클래스를 활용하여 학습을 제어하는 방법
# 1. 예측기의 개수를 제어
# 2. 학습률을 통해 제어
from sklearn.linear_model import LogisticRegression
ada_model = AdaBoostClassifier(
        LogisticRegression(
                C=0.1,
                solver='lbfgs',
                max_iter=10000),
                n_estimators=100000,
                learning_rate=1.0,
                random_state=1).fit(X_train, y_train)

print('학습 평가(ada) : ', 
      ada_model.score(X_train, y_train))

# GradientBoostingClassifier 클래스는 내부를 구성하는
# 하위 예측 모델을 결정 트리를 사용합니다.
gb_model = GradientBoostingClassifier(
        n_estimators=1000, max_depth=2, 
        random_state=1).fit(X_train, y_train)

print('학습 평가(gb) : ', 
      gb_model.score(X_train, y_train))

print('테스트 평가(ada) : ', 
      ada_model.score(X_test, y_test))
print('테스트 평가(gb) : ', 
      gb_model.score(X_test, y_test))
























